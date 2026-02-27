from __future__ import annotations

import hashlib
import logging
import time
import json
import os
import sqlite3
import ssl
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import LLMConfig
from .gigachat_schema import NormalizeTicket


SYSTEM_PROMPT = "Ты обязан вернуть ТОЛЬКО JSON без markdown. Никаких комментариев."

logger = logging.getLogger(__name__)


def _validate_mtls_files(ca_bundle_file: str | None, cert_file: str | None, key_file: str | None) -> None:
    required = [ca_bundle_file, cert_file, key_file]
    missing = [str(p) for p in required if not p or not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "mTLS files are missing for GigaChat: " + ", ".join(missing)
            + ". Set llm.ca_bundle_file/cert_file/key_file (or corresponding GIGACHAT_* env overrides)."
        )


def _safe_cert_summary(cert_file: str | None) -> str:
    if not cert_file:
        return "cert_file=<empty>"
    cert_path = Path(cert_file)
    if not cert_path.exists():
        return f"cert_file={cert_file} (missing)"
    try:
        out = subprocess.run(
            ["openssl", "x509", "-in", str(cert_path), "-noout", "-subject", "-issuer", "-serial"],
            capture_output=True,
            text=True,
            check=True,
        )
        summary = " | ".join(line.strip() for line in out.stdout.splitlines() if line.strip())
        return f"cert_file={cert_path}; {summary}"
    except Exception:
        return f"cert_file={cert_path}; openssl_summary=unavailable"


def _base_url_host_port(base_url: str) -> tuple[str, int | None]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "<unknown-host>"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


def _tls_debug_context(cfg: LLMConfig) -> str:
    proxy_vars = [name for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY") if os.getenv(name)]
    proxy_hint = ",".join(proxy_vars) if proxy_vars else "none"
    return (
        f"mode={cfg.mode}; base_url={cfg.base_url}; verify_ssl_certs={cfg.verify_ssl_certs}; "
        f"proxies={proxy_hint}"
    )




def _coerce_response_fields(parsed: dict, payload: dict) -> dict:
    full_dialog = str(payload.get("full_dialog_text", "") or "")
    dialog_context = str(payload.get("dialog_context", "") or "")
    dialog_text = full_dialog or dialog_context or ""
    category = parsed.get("complaint_category") or parsed.get("category") or "OTHER"
    subcategory = parsed.get("complaint_subcategory") or parsed.get("subcategory")
    product = parsed.get("product_area") or parsed.get("product")
    loan_product = parsed.get("loan_product") or parsed.get("product")
    is_complaint = parsed.get("is_complaint")
    if is_complaint is None:
        is_complaint = str(category).upper() not in {"OTHER", "NONE", "NON_COMPLAINT"}
    severity = parsed.get("severity") or ("medium" if is_complaint else "low")
    if severity not in {"low", "medium", "high"}:
        severity = "medium" if is_complaint else "low"
    keywords = parsed.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        keywords = ["жалоба", "обращение", "сервис"] if is_complaint else ["вопрос", "инфо", "уточнение"]
    keywords = [str(k) for k in keywords][:8]
    while len(keywords) < 3:
        keywords.append("уточнение")
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.6 if is_complaint else 0.5
    confidence = max(0.0, min(1.0, confidence))

    return {
        "client_first_message": dialog_text,
        "short_summary": parsed.get("short_summary") or parsed.get("summary") or dialog_text[:120],
        "is_complaint": bool(is_complaint),
        "complaint_category": str(category),
        "complaint_subcategory": subcategory,
        "product_area": product,
        "loan_product": loan_product or "NONE",
        "severity": severity,
        "keywords": keywords,
        "confidence": confidence,
        "notes": parsed.get("notes"),
    }


def _normalize_code(s: str) -> str:
    v = str(s or "").strip().lower().replace("-", "_").replace(" ", "_")
    out = []
    for ch in v:
        if ch.isalnum() or ch == "_":
            out.append(ch)
    code = "".join(out).strip("_")
    return code or "other"


def _normalize_error_with_tls_hint(cfg: LLMConfig, e: Exception, *, phase: str) -> RuntimeError | None:
    msg = str(e)
    if "TLSV13_ALERT_CERTIFICATE_REQUIRED" not in msg and "certificate required" not in msg.lower():
        return None
    if cfg.mode == "tls":
        if phase == "repair":
            return RuntimeError(
                "GigaChat TLS handshake failed on repair request: server requires client certificate. "
                "Switch llm.mode to mtls or use a non-mTLS endpoint."
            )
        return RuntimeError(
            "GigaChat TLS handshake failed: server requires client certificate. "
            "Switch llm.mode to mtls and set cert_file/key_file, or use an endpoint that does not require mTLS."
        )
    host, port = _base_url_host_port(cfg.base_url)
    cert_summary = _safe_cert_summary(cfg.cert_file)
    if phase == "repair":
        return RuntimeError(
            "GigaChat mTLS handshake failed on repair request: certificate required/rejected by server. "
            "Verify endpoint host mapping, cert chain, and server-side client cert policy. "
            f"Target={host}:{port}. {cert_summary}. Debug: {_tls_debug_context(cfg)}"
        )
    return RuntimeError(
        "GigaChat mTLS handshake failed: server requires/rejects client certificate during TLS. "
        "This usually means the cert was not accepted by endpoint policy (DN/issuer/chain) for this host. "
        "Check endpoint host, mTLS cert mapping/whitelist on server side, full certificate chain, and proxy interference. "
        f"Target={host}:{port}. {cert_summary}. Debug: {_tls_debug_context(cfg)}"
    )

def _build_mtls_ssl_context(
    ca_bundle_file: str | None,
    cert_file: str | None,
    key_file: str | None,
    key_file_password: str | None,
    verify_ssl_certs: bool,
) -> ssl.SSLContext:
    context = ssl.create_default_context(cafile=ca_bundle_file)
    context.check_hostname = verify_ssl_certs
    if not verify_ssl_certs:
        context.verify_mode = ssl.CERT_NONE
    context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file), password=key_file_password)
    return context


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _ChatResp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _HTTPXChatClient:
    def __init__(self, *, base_url: str, verify: bool | str | ssl.SSLContext, timeout: float = 60.0):
        self._client = httpx.Client(base_url=base_url, verify=verify, timeout=timeout, trust_env=False)

    def count_tokens(self, *, model: str, input_text: str) -> int | None:
        try:
            response = self._client.post("/tokens/count", json={"model": model, "input": [input_text]})
            if response.status_code >= 400:
                return None
            data = response.json()
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    return int(first.get("tokens") or first.get("count") or first.get("token_count"))
            if isinstance(data, dict):
                return int(data.get("tokens") or data.get("count") or data.get("token_count"))
            return None
        except Exception:
            return None

    def chat(self, payload: dict) -> _ChatResp:
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _ChatResp(content)


class LLMCache:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
            )

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        with self._lock:
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
            self.taxonomy_raw = taxonomy.get("raw", {})
        else:
            self.categories = taxonomy
            self.subcategories_by_category = {}
            self.loan_products = ["NONE"]
            self.taxonomy_raw = {}

        self.category_mode = getattr(cfg, "category_mode", "taxonomy")
        self.discovered_categories: list[str] = []

        if not mock:
            if cfg.mode == "mtls":
                _validate_mtls_files(cfg.ca_bundle_file, cfg.cert_file, cfg.key_file)
                key_password = (os.getenv(cfg.key_file_password_env) if cfg.key_file_password_env else os.getenv("GIGACHAT_KEY_PASSWORD")) or None
                ssl_context = _build_mtls_ssl_context(
                    ca_bundle_file=cfg.ca_bundle_file,
                    cert_file=cfg.cert_file,
                    key_file=cfg.key_file,
                    key_file_password=key_password,
                    verify_ssl_certs=cfg.verify_ssl_certs,
                )
                self.client = _HTTPXChatClient(base_url=cfg.base_url, verify=ssl_context, timeout=60.0)
            else:
                verify: bool | str = cfg.verify_ssl_certs
                if cfg.ca_bundle_file:
                    verify = cfg.ca_bundle_file
                self.client = _HTTPXChatClient(base_url=cfg.base_url, verify=verify, timeout=60.0)

    def _key(self, payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False) + self.cfg.prompt_version
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _llm_input(self, payload: dict) -> dict:
        return {k: v for k, v in payload.items() if k != "client_first_message"}

    def _single_user_prompt(self, payload: dict) -> str:
        if self.category_mode == "discover":
            return json.dumps(
                {
                    "task": "normalize_ticket_discover_categories",
                    "rules": {
                        "choose_or_create_category": True,
                        "if_possible_reuse_previous_category": True,
                        "if_no_match_create_new_short_category_code": True,
                        "category_code_format": "snake_case_ascii",
                        "multi_dialog_fields": "Используй ВСЕ доступные поля входа (full_dialog_text, dialog_context, signal_fields). Оценивай весь диалог.",
                        "ignore_empty_context_fields": True,
                    },
                    "existing_categories": sorted(set([*self.categories, *self.discovered_categories])),
                    "input": self._llm_input(payload),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "task": "normalize_ticket",
                "rules": {
                    "choose_exactly_one_category": True,
                    "category_must_be_from_allowed": True,
                    "subcategory_should_match_chosen_category": True,
                    "loan_product_rule": "Если обращение про кредитование: loan_product != NONE, иначе loan_product = NONE",
                    "multi_dialog_fields": "Используй ВСЕ доступные поля входа (full_dialog_text, dialog_context, signal_fields). Оценивай весь диалог.",
                    "ignore_empty_context_fields": True,
                },
                "allowed_categories": self.categories,
                "allowed_subcategories_by_category": self.subcategories_by_category,
                "allowed_loan_products": self.loan_products,
                "taxonomy_raw": self.taxonomy_raw,
                "input": self._llm_input(payload),
            },
            ensure_ascii=False,
        )

    def _remember_discovered_category(self, parsed_or_obj) -> None:
        if self.category_mode != "discover":
            return
        cat = None
        if isinstance(parsed_or_obj, dict):
            cat = parsed_or_obj.get("complaint_category") or parsed_or_obj.get("category")
        else:
            cat = getattr(parsed_or_obj, "complaint_category", None)
        if not cat:
            return
        code = _normalize_code(cat)
        if code not in self.discovered_categories:
            self.discovered_categories.append(code)

    def estimate_tokens(self, payload: dict) -> int:
        prompt = self._single_user_prompt(payload)
        if self.cfg.request_metrics_enabled and self.client and hasattr(self.client, "count_tokens"):
            token_input = f"{SYSTEM_PROMPT}\n{prompt}"
            count = self.client.count_tokens(model=self.cfg.model, input_text=token_input)
            if count is not None:
                return max(1, int(count))
        return max(1, len(prompt) // 4)

    def normalize_batch(self, payloads: list[dict]) -> list[NormalizeTicket]:
        if not payloads:
            return []
        if self.mock:
            return [self.normalize(p) for p in payloads]

        results: list[NormalizeTicket | None] = [None] * len(payloads)
        uncached_idx: list[int] = []
        uncached_payloads: dict[int, dict] = {}
        for i, payload in enumerate(payloads):
            cached = self.cache.get(self._key(payload))
            if cached:
                results[i] = NormalizeTicket.model_validate(cached)
            else:
                uncached_idx.append(i)
                uncached_payloads[i] = payload

        if not uncached_payloads:
            return [r for r in results if r is not None]

        max_attempts = 3
        pending = set(uncached_idx)

        for attempt in range(1, max_attempts + 1):
            if not pending:
                break

            batch_indexes = sorted(pending)
            batch_payloads = [uncached_payloads[idx] for idx in batch_indexes]

            user_prompt = json.dumps(
                {
                    "task": "normalize_tickets",
                    "rules": {
                        "choose_exactly_one_category": True,
                        "category_must_be_from_allowed": True,
                        "subcategory_should_match_chosen_category": True,
                        "loan_product_rule": "Если обращение про кредитование: loan_product != NONE, иначе loan_product = NONE",
                        "multi_dialog_fields": "Используй ВСЕ доступные поля входа (full_dialog_text, dialog_context, signal_fields). Оценивай весь диалог.",
                        "ignore_empty_context_fields": True,
                        "return_one_result_per_input": True,
                        "must_return_batch_index": True,
                    },
                    "allowed_categories": self.categories,
                    "allowed_subcategories_by_category": self.subcategories_by_category,
                    "allowed_loan_products": self.loan_products,
                    "taxonomy_raw": self.taxonomy_raw,
                    "inputs": [
                        {"_batch_index": idx, **self._llm_input(payload)}
                        for idx, payload in zip(batch_indexes, batch_payloads)
                    ],
                },
                ensure_ascii=False,
            )

            req_token_count = None
            if self.cfg.request_metrics_enabled and self.client and hasattr(self.client, "count_tokens"):
                token_input = f"{SYSTEM_PROMPT}\n{user_prompt}"
                req_token_count = self.client.count_tokens(model=self.cfg.model, input_text=token_input)
                logger.info("[stage=prepare/llm] tokens per batch request: %s", req_token_count if req_token_count is not None else "n/a")

            chat_started = time.perf_counter()
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
                hinted = _normalize_error_with_tls_hint(self.cfg, e, phase="single")
                if hinted is not None:
                    raise hinted from e
                raise
            elapsed_ms = int((time.perf_counter() - chat_started) * 1000)
            if self.cfg.request_metrics_enabled:
                logger.info(
                    "[stage=prepare/llm] batch request latency_ms=%s tokens=%s size=%s attempt=%s",
                    elapsed_ms,
                    req_token_count if req_token_count is not None else "n/a",
                    len(batch_payloads),
                    attempt,
                )
                logger.info("[stage=prepare/llm] batch request delivered successfully size=%s attempt=%s", len(batch_payloads), attempt)

            parsed = json.loads(response.choices[0].message.content)
            if isinstance(parsed, dict):
                items = parsed.get("items") or parsed.get("results") or parsed.get("tickets") or []
            elif isinstance(parsed, list):
                items = parsed
            else:
                items = []

            if not isinstance(items, list):
                items = []

            assigned: set[int] = set()
            leftovers: list[dict] = []

            for item in items:
                if not isinstance(item, dict):
                    continue
                raw_idx = item.get("_batch_index")
                try:
                    idx = int(raw_idx)
                except Exception:
                    idx = None
                if idx is None or idx not in pending or idx in assigned:
                    leftovers.append(item)
                    continue
                obj = NormalizeTicket.model_validate(_coerce_response_fields(item, uncached_payloads[idx]))
                self._remember_discovered_category(item)
                self.cache.set(self._key(uncached_payloads[idx]), obj.model_dump())
                results[idx] = obj
                assigned.add(idx)

            unassigned_pending = [idx for idx in batch_indexes if idx not in assigned]
            for idx, item in zip(unassigned_pending, leftovers):
                obj = NormalizeTicket.model_validate(_coerce_response_fields(item, uncached_payloads[idx]))
                self._remember_discovered_category(item)
                self.cache.set(self._key(uncached_payloads[idx]), obj.model_dump())
                results[idx] = obj
                assigned.add(idx)

            pending -= assigned
            if pending:
                logger.warning(
                    "[stage=prepare/llm] batch missing %s/%s rows, retrying pending subset (attempt=%s/%s)",
                    len(pending),
                    len(batch_indexes),
                    attempt,
                    max_attempts,
                )

        if pending:
            raise RuntimeError(
                f"Batch response incomplete after retries: unresolved={len(pending)} indexes={sorted(pending)}"
            )

        return [r for r in results if r is not None]

    def normalize(self, payload: dict) -> NormalizeTicket:
        k = self._key(payload)
        cached = self.cache.get(k)
        if cached:
            return NormalizeTicket.model_validate(cached)
        if self.mock:
            txt = str(payload.get("full_dialog_text", "") or payload.get("dialog_context", "") or "")
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

        user_prompt = self._single_user_prompt(payload)
        req_token_count = None
        if self.cfg.request_metrics_enabled and self.client and hasattr(self.client, "count_tokens"):
            token_input = f"{SYSTEM_PROMPT}\n{user_prompt}"
            req_token_count = self.client.count_tokens(model=self.cfg.model, input_text=token_input)
            logger.info("[stage=prepare/llm] tokens per request: %s", req_token_count if req_token_count is not None else "n/a")

        chat_started = time.perf_counter()
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
            hinted = _normalize_error_with_tls_hint(self.cfg, e, phase="single")
            if hinted is not None:
                raise hinted from e
            raise
        elapsed_ms = int((time.perf_counter() - chat_started) * 1000)
        if self.cfg.request_metrics_enabled:
            logger.info("[stage=prepare/llm] request latency_ms=%s tokens=%s", elapsed_ms, req_token_count if req_token_count is not None else "n/a")
            logger.info("[stage=prepare/llm] request delivered successfully")
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
            obj = NormalizeTicket.model_validate(_coerce_response_fields(parsed, payload))
            self._remember_discovered_category(parsed)
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
                hinted = _normalize_error_with_tls_hint(self.cfg, e, phase="repair")
                if hinted is not None:
                    raise hinted from e
                raise
            repaired = json.loads(response2.choices[0].message.content)
            obj = NormalizeTicket.model_validate(_coerce_response_fields(repaired, payload))
            self._remember_discovered_category(repaired)
        self.cache.set(k, obj.model_dump())
        return obj
