from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from gigachat import GigaChat

from .config import LLMConfig
from .gigachat_schema import NormalizeTicket


SYSTEM_PROMPT = "Ты обязан вернуть ТОЛЬКО JSON без markdown. Никаких комментариев."


def _validate_mtls_files(ca_bundle_file: str | None, cert_file: str | None, key_file: str | None) -> None:
    required = [ca_bundle_file, cert_file, key_file]
    missing = [str(p) for p in required if not p or not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "mTLS files are missing for GigaChat: " + ", ".join(missing) +
            ". Set llm.ca_bundle_file/cert_file/key_file (or corresponding GIGACHAT_* env overrides)."
        )


class LLMCache:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )

    def get(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO cache(k,v) VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))
        self.conn.commit()


class GigaChatNormalizer:
    def __init__(self, cfg: LLMConfig, taxonomy: dict | list[str], mock: bool = False):
        self.cfg = cfg
        self.cache = LLMCache(cfg.cache_db)
        self.mock = mock
        self.client = None

        if isinstance(taxonomy, dict):
            self.categories = taxonomy.get("category_codes", [])
            self.subcategories_by_category = taxonomy.get("subcategories_by_category", {})
            self.loan_products = taxonomy.get("loan_products", ["NONE"])
        else:
            self.categories = taxonomy
            self.subcategories_by_category = {}
            self.loan_products = ["NONE"]

        if not mock:
            if cfg.mode == "mtls":
                _validate_mtls_files(cfg.ca_bundle_file, cfg.cert_file, cfg.key_file)
                self.client = GigaChat(
                    base_url=cfg.base_url,
                    ca_bundle_file=cfg.ca_bundle_file,
                    cert_file=cfg.cert_file,
                    key_file=cfg.key_file,
                    key_file_password=(os.getenv(cfg.key_file_password_env) if cfg.key_file_password_env else os.getenv("GIGACHAT_KEY_PASSWORD")) or None,
                    verify_ssl_certs=cfg.verify_ssl_certs,
                    timeout=60.0,
                    max_retries=3,
                    retry_backoff_factor=0.5,
                )
            else:
                self.client = GigaChat(
                    base_url=cfg.base_url,
                    ca_bundle_file=cfg.ca_bundle_file,
                    verify_ssl_certs=cfg.verify_ssl_certs,
                    timeout=60.0,
                    max_retries=3,
                    retry_backoff_factor=0.5,
                )

    def _key(self, payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False) + self.cfg.prompt_version
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def normalize(self, payload: dict) -> NormalizeTicket:
        k = self._key(payload)
        cached = self.cache.get(k)
        if cached:
            return NormalizeTicket.model_validate(cached)
        if self.mock:
            txt = payload.get("client_first_message", "")
            is_complaint = any(w in txt.lower() for w in ["жалоб", "не работает", "ошибка", "проблем"])
            resp = NormalizeTicket(
                client_first_message=txt,
                short_summary=txt[:120],
                is_complaint=is_complaint,
                complaint_category="TECHNICAL" if is_complaint else "OTHER",
                complaint_subcategory="payment_error" if is_complaint else None,
                product_area=payload.get("product"),
                loan_product="CONSUMER_LOAN" if ("кредит" in txt.lower()) else "NONE",
                severity="medium" if is_complaint else "low",
                keywords=["ошибка", "оплата", "приложение"] if is_complaint else ["вопрос", "инфо", "уточнение"],
                confidence=0.8,
                notes=None,
            )
            self.cache.set(k, resp.model_dump())
            return resp

        user_prompt = json.dumps(
            {
                "task": "normalize_ticket",
                "rules": {
                    "choose_exactly_one_category": True,
                    "category_must_be_from_allowed": True,
                    "subcategory_should_match_chosen_category": True,
                    "loan_product_rule": "Если обращение про кредитование: loan_product != NONE, иначе loan_product = NONE",
                    "multi_dialog_fields": "Вход может содержать несколько текстовых полей (чат/звонок/комментарий/summary). Используй client_first_message как основной источник, dialog_context как доп.контекст.",
                    "ignore_empty_context_fields": True,
                },
                "allowed_categories": self.categories,
                "allowed_subcategories_by_category": self.subcategories_by_category,
                "allowed_loan_products": self.loan_products,
                "input": payload,
            },
            ensure_ascii=False,
        )
        try:
            response = self.client.chat(
                {
                    "model": self.cfg.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                }
            )
        except Exception as e:
            msg = str(e)
            if "TLSV13_ALERT_CERTIFICATE_REQUIRED" in msg or "certificate required" in msg.lower():
                if self.cfg.mode == "tls":
                    raise RuntimeError(
                        "GigaChat TLS handshake failed: server requires client certificate. "
                        "Switch llm.mode to mtls and set cert_file/key_file, or use an endpoint that does not require mTLS."
                    ) from e
                raise RuntimeError(
                    "GigaChat mTLS handshake failed: server requires client certificate. "
                    "Check cert_file/key_file/ca_bundle_file paths and that env overrides are not pointing to empty files."
                ) from e
            raise
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
            obj = NormalizeTicket.model_validate(parsed)
        except Exception:
            repair_prompt = f"Исправь: верни JSON по схеме, убери запрещенные токены.\n{content}"
            try:
                response2 = self.client.chat(
                    {
                        "model": self.cfg.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": repair_prompt},
                        ],
                    }
                )
            except Exception as e:
                msg = str(e)
                if "TLSV13_ALERT_CERTIFICATE_REQUIRED" in msg or "certificate required" in msg.lower():
                    if self.cfg.mode == "tls":
                        raise RuntimeError(
                            "GigaChat TLS handshake failed on repair request: server requires client certificate. "
                            "Switch llm.mode to mtls or use a non-mTLS endpoint."
                        ) from e
                    raise RuntimeError(
                        "GigaChat mTLS handshake failed on repair request: certificate required by server. "
                        "Verify mTLS cert/key/CA configuration."
                    ) from e
                raise
            obj = NormalizeTicket.model_validate(json.loads(response2.choices[0].message.content))
        self.cache.set(k, obj.model_dump())
        return obj
