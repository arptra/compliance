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
from .gigachat_api import build_gigachat_transport_client
from .gigachat_api.rate_limit import get_rate_limiter
from .gigachat_schema import NormalizeTicket
from .questions_loader import load_questions, save_questions_taxonomy


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




def _coerce_questions_payload(parsed: dict, fallback_code: str = "OTHER") -> dict:
    if not isinstance(parsed, dict):
        return parsed
    out = dict(parsed)
    primary = out.get("primary_category_code") or out.get("complaint_category") or out.get("category") or fallback_code
    out["complaint_category"] = str(primary)
    out["complaint_subcategory"] = None
    if str(primary).upper() == str(fallback_code).upper():
        out["is_complaint"] = False
    else:
        out["is_complaint"] = bool(out.get("is_complaint", True))
    trig = out.get("triggered_codes")
    if isinstance(trig, list):
        note_prefix = f"triggered_codes={trig}"
        out["notes"] = (note_prefix + "; " + str(out.get("notes", "")).strip()).strip("; ")
    return out


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


def _short_category_name_from_question(question_ru: str) -> str:
    q = str(question_ru or "").strip()
    if not q:
        return "Обращение"
    low = q.lower().replace("?", "").strip()
    prefixes = [
        "есть ли жалоба на ",
        "жалоба на ",
        "есть ли проблема с ",
        "проблема с ",
    ]
    for p in prefixes:
        if low.startswith(p):
            low = low[len(p):].strip()
            break
    if not low:
        low = q.replace("?", "").strip()
    words = [w for w in low.split() if w]
    short = " ".join(words[:4]).strip()
    if not short:
        short = q.replace("?", "").strip()
    if not short:
        short = "Обращение"
    return short[0].upper() + short[1:]


def _normalize_subcategory_code(s: str) -> str:
    return _normalize_code(s)


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
        self._api_limiter = get_rate_limiter(f"api:{base_url}")

    def count_tokens(self, *, model: str, input_text: str) -> int | None:
        try:
            response = self._post_api_with_backoff("/tokens/count", json={"model": model, "input": [input_text]})
            if response.status_code >= 400:
                return None
            data = response.json()
            return self._extract_token_count(data)
        except Exception:
            return None

    def _post_api_with_backoff(self, path: str, *, json: dict) -> httpx.Response:
        response = None
        for attempt in range(6):
            self._api_limiter.wait()
            response = self._client.post(path, json=json)
            if response.status_code != 429:
                if response.status_code < 400:
                    self._api_limiter.record_success()
                return response
            self._api_limiter.backoff(response, attempt)
        if response is None:
            raise RuntimeError("GigaChat request was not executed")
        return response

    @classmethod
    def _extract_token_count(cls, data: object) -> int | None:
        if isinstance(data, list):
            counts = [cls._extract_token_count(item) for item in data]
            valid_counts = [item for item in counts if item is not None]
            return sum(valid_counts) if valid_counts else None
        if isinstance(data, dict):
            for key in ("tokens", "count", "token_count", "total_tokens"):
                value = data.get(key)
                if isinstance(value, (int, float, str)) and str(value).strip():
                    try:
                        return int(float(value))
                    except Exception:
                        pass
                nested = cls._extract_token_count(value)
                if nested is not None:
                    return nested
            for key in ("data", "items", "result", "results"):
                nested = cls._extract_token_count(data.get(key))
                if nested is not None:
                    return nested
        return None

    def chat(self, payload: dict) -> _ChatResp:
        response = self._post_api_with_backoff("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _ChatResp(content)

    def list_models(self) -> list[str]:
        response = self._client.get("/models")
        response.raise_for_status()
        payload = response.json()
        items = payload
        if isinstance(payload, dict):
            items = payload.get("data") or payload.get("models") or payload.get("items") or []
        if not isinstance(items, list):
            return []
        models: list[str] = []
        for item in items:
            if isinstance(item, dict):
                model_name = item.get("id") or item.get("name")
                if model_name:
                    models.append(str(model_name))
            elif item:
                models.append(str(item))
        return models


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
        self.discovered_taxonomy_file = Path(getattr(cfg, "discovered_taxonomy_file", "data/interim/discovered_categories.json"))
        self.questions_file = Path(getattr(cfg, "questions_file", "configs/questions_categories.json"))
        self.discovered_categories: list[str] = []
        self.discovered_subcategories_by_category: dict[str, list[str]] = {}
        self.question_items: list[dict] = []
        self.question_file_hash: str = ""
        self.questions_fallback_code = "OTHER"
        self.question_category_map: dict[str, dict] = {}
        self.question_category_map_file = Path("data/interim/questions_category_map.json")
        self.question_map_hash: str = ""
        self._load_discovered_taxonomy()
        self._load_questions_mode()

        if not mock:
            self.client = build_gigachat_transport_client(cfg)

    def _key(self, payload: dict) -> str:
        raw = json.dumps(
            {
                "payload": payload,
                "prompt_version": self.cfg.prompt_version,
                "mode_signature": self._mode_signature(),
                "model": self.cfg.model,
                "temperature": getattr(self.cfg, "temperature", 0.2),
                "top_p": getattr(self.cfg, "top_p", 0.95),
                "max_output_tokens": getattr(self.cfg, "max_output_tokens", 2048),
                "system_prompt": getattr(self.cfg, "system_prompt", SYSTEM_PROMPT),
                "user_prompt_prefix": getattr(self.cfg, "user_prompt_prefix", ""),
                "context_notes": getattr(self.cfg, "context_notes", ""),
                "tagging_prompt_notes": getattr(self.cfg, "tagging_prompt_notes", ""),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _mode_signature(self) -> str:
        if self.category_mode == "discover":
            return f"|mode=discover|file={self.discovered_taxonomy_file}"
        if self.category_mode == "questions":
            return f"|mode=questions|qhash={self.question_file_hash}|qmap={self.question_map_hash}"
        return "|mode=taxonomy"

    def _llm_input(self, payload: dict) -> dict:
        return {k: v for k, v in payload.items() if k != "client_first_message"}

    def _system_prompt(self) -> str:
        value = str(getattr(self.cfg, "system_prompt", SYSTEM_PROMPT) or "").strip()
        return value or SYSTEM_PROMPT

    def _wrap_user_prompt(self, base_prompt: str) -> str:
        context_notes = str(getattr(self.cfg, "context_notes", "") or "").strip()
        user_prompt_prefix = str(getattr(self.cfg, "user_prompt_prefix", "") or "").strip()
        tagging_notes = self._format_labeling_notes(str(getattr(self.cfg, "tagging_prompt_notes", "") or "").strip())
        parts: list[str] = []
        if context_notes:
            parts.append(f"Контекст эксперимента:\n{context_notes}")
        if tagging_notes:
            parts.append(f"Правила тегирования:\n{tagging_notes}")
        if user_prompt_prefix:
            parts.append(user_prompt_prefix)
        parts.append(base_prompt)
        return "\n\n".join(parts)

    @staticmethod
    def _format_labeling_notes(raw: str) -> str:
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                lines: list[str] = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name", "") or "").strip()
                    description = str(item.get("description", "") or "").strip()
                    if name and description:
                        lines.append(f"- {name}: {description}")
                    elif name:
                        lines.append(f"- {name}")
                    elif description:
                        lines.append(f"- {description}")
                if lines:
                    return "\n".join(lines)
        except Exception:
            pass
        return raw

    def _chat_payload(self, user_prompt: str) -> dict:
        payload: dict = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
        }
        temperature = getattr(self.cfg, "temperature", None)
        top_p = getattr(self.cfg, "top_p", None)
        max_output_tokens = getattr(self.cfg, "max_output_tokens", None)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if max_output_tokens is not None:
            payload["max_tokens"] = int(max_output_tokens)
        return payload

    def _single_user_prompt(self, payload: dict) -> str:
        if self.category_mode == "questions":
            existing_categories = sorted({v.get("category_code", "") for v in self.question_category_map.values() if str(v.get("category_code", "")).strip()})
            allowed_categories = [*existing_categories, self.questions_fallback_code] if existing_categories else [self.questions_fallback_code]
            return self._wrap_user_prompt(json.dumps(
                {
                    "task": "questionnaire_normalize_ticket",
                    "rules": {
                        "evaluate_each_question": True,
                        "return_matched_boolean_and_short_rationale": True,
                        "select_primary_question_from_allowed_question_codes": True,
                        "if_primary_question_has_existing_category_reuse_it": True,
                        "if_primary_question_has_no_category_generate_new_short_category_name": True,
                        "category_name_must_be_short": True,
                        "category_name_must_not_equal_question_text": True,
                        "classification_must_be_driven_by_question_and_dialog_semantics": True,
                        "if_no_match_use_other": True,
                        "if_other_then_is_complaint_false": True,
                        "return_fields": ["primary_question_code", "category_name", "triggered_codes", "keywords", "notes"],
                        "return_json_compatible_with_normalize_ticket": True,
                    },
                    "allowed_question_codes": [x["code"] for x in self.question_items],
                    "allowed_categories": allowed_categories,
                    "questions": self.question_items,
                    "existing_question_category_map": self.question_category_map,
                    "input": self._llm_input(payload),
                },
                ensure_ascii=False,
            ))
        if self.category_mode == "discover":
            return self._wrap_user_prompt(json.dumps(
                {
                    "task": "discover_and_normalize_ticket",
                    "rules": {
                        "you_must_discover_categories_yourself": True,
                        "do_not_use_external_taxonomy": True,
                        "reuse_existing_discovered_categories_when_semantically_close": True,
                        "if_no_semantic_match_create_new_category": True,
                        "category_code_format": "snake_case_ascii_short",
                        "subcategory_code_format": "snake_case_ascii_short",
                        "return_category_and_subcategory_codes": True,
                        "multi_dialog_fields": "Используй ВСЕ доступные поля входа (full_dialog_text, dialog_context, signal_fields). Оценивай весь диалог.",
                        "ignore_empty_context_fields": True,
                    },
                    "existing_discovered_taxonomy": {
                        "categories": sorted(set(self.discovered_categories)),
                        "subcategories_by_category": self.discovered_subcategories_by_category,
                    },
                    "input": self._llm_input(payload),
                },
                ensure_ascii=False,
            ))
        return self._wrap_user_prompt(json.dumps(
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
        ))

    def _batch_user_prompt(self, batch_indexes: list[int], batch_payloads: list[dict]) -> str:
        if self.category_mode == "questions":
            existing_categories = sorted({v.get("category_code", "") for v in self.question_category_map.values() if str(v.get("category_code", "")).strip()})
            allowed_categories = [*existing_categories, self.questions_fallback_code] if existing_categories else [self.questions_fallback_code]
            return self._wrap_user_prompt(json.dumps(
                {
                    "task": "questionnaire_normalize_batch",
                    "rules": {
                        "return_one_result_per_input": True,
                        "must_return_batch_index": True,
                        "select_primary_question_from_allowed_question_codes": True,
                        "if_primary_question_has_existing_category_reuse_it": True,
                        "if_primary_question_has_no_category_generate_new_short_category_name": True,
                        "category_name_must_be_short": True,
                        "category_name_must_not_equal_question_text": True,
                        "classification_must_be_driven_by_question_and_dialog_semantics": True,
                        "if_other_then_is_complaint_false": True,
                        "return_result_fields": ["primary_question_code", "category_name", "triggered_codes", "keywords", "notes"],
                        "return_result_compatible_with_normalize_ticket": True,
                    },
                    "allowed_question_codes": [x["code"] for x in self.question_items],
                    "allowed_categories": allowed_categories,
                    "questions": self.question_items,
                    "existing_question_category_map": self.question_category_map,
                    "inputs": [
                        {"_batch_index": idx, "input": self._llm_input(payload)}
                        for idx, payload in zip(batch_indexes, batch_payloads)
                    ],
                },
                ensure_ascii=False,
            ))
        if self.category_mode == "discover":
            return self._wrap_user_prompt(json.dumps(
                {
                    "task": "discover_and_normalize_tickets",
                    "rules": {
                        "you_must_discover_categories_yourself": True,
                        "do_not_use_external_taxonomy": True,
                        "reuse_existing_discovered_categories_when_semantically_close": True,
                        "if_no_semantic_match_create_new_category": True,
                        "category_code_format": "snake_case_ascii_short",
                        "subcategory_code_format": "snake_case_ascii_short",
                        "return_category_and_subcategory_codes": True,
                        "return_one_result_per_input": True,
                        "must_return_batch_index": True,
                        "multi_dialog_fields": "Используй ВСЕ доступные поля входа (full_dialog_text, dialog_context, signal_fields). Оценивай весь диалог.",
                        "ignore_empty_context_fields": True,
                    },
                    "existing_discovered_taxonomy": {
                        "categories": sorted(set(self.discovered_categories)),
                        "subcategories_by_category": self.discovered_subcategories_by_category,
                    },
                    "inputs": [
                        {"_batch_index": idx, **self._llm_input(payload)}
                        for idx, payload in zip(batch_indexes, batch_payloads)
                    ],
                },
                ensure_ascii=False,
            ))
        return self._wrap_user_prompt(json.dumps(
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
        ))

    def _remember_discovered_category(self, parsed_or_obj) -> None:
        if self.category_mode != "discover":
            return
        cat = None
        sub = None
        if isinstance(parsed_or_obj, dict):
            cat = parsed_or_obj.get("complaint_category") or parsed_or_obj.get("category")
            sub = parsed_or_obj.get("complaint_subcategory") or parsed_or_obj.get("subcategory")
        else:
            cat = getattr(parsed_or_obj, "complaint_category", None)
            sub = getattr(parsed_or_obj, "complaint_subcategory", None)
        if not cat:
            return
        code = _normalize_code(cat)
        if code not in self.discovered_categories:
            self.discovered_categories.append(code)
        if sub:
            sub_code = _normalize_subcategory_code(sub)
            self.discovered_subcategories_by_category.setdefault(code, [])
            if sub_code not in self.discovered_subcategories_by_category[code]:
                self.discovered_subcategories_by_category[code].append(sub_code)
        self._save_discovered_taxonomy()

    def _load_discovered_taxonomy(self) -> None:
        if self.category_mode != "discover":
            return
        p = self.discovered_taxonomy_file
        if not p.exists() or p.stat().st_size == 0:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"categories": [], "subcategories_by_category": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
            self.discovered_categories = []
            self.discovered_subcategories_by_category = {}
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {"categories": [], "subcategories_by_category": {}}
        cats = data.get("categories", []) if isinstance(data, dict) else []
        subs = data.get("subcategories_by_category", {}) if isinstance(data, dict) else {}
        self.discovered_categories = [_normalize_code(x) for x in cats if str(x).strip()]
        fixed_subs: dict[str, list[str]] = {}
        if isinstance(subs, dict):
            for k, v in subs.items():
                kc = _normalize_code(k)
                vals = v if isinstance(v, list) else []
                fixed_subs[kc] = [_normalize_subcategory_code(x) for x in vals if str(x).strip()]
        self.discovered_subcategories_by_category = fixed_subs

    def _save_discovered_taxonomy(self) -> None:
        if self.category_mode != "discover":
            return
        p = self.discovered_taxonomy_file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "categories": sorted(set(self.discovered_categories)),
                    "subcategories_by_category": {k: sorted(set(v)) for k, v in self.discovered_subcategories_by_category.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )



    def _question_code_to_text(self) -> dict[str, str]:
        return {str(x.get("code", "")): str(x.get("question_ru", "")) for x in self.question_items if str(x.get("code", "")).strip()}


    def _update_question_map_hash(self) -> None:
        raw = json.dumps(self.question_category_map, ensure_ascii=False, sort_keys=True)
        self.question_map_hash = hashlib.sha1(raw.encode("utf-8")).hexdigest()


    def _load_question_category_map(self) -> None:
        self.question_category_map = {}
        if not self.question_category_map_file.exists():
            self._update_question_map_hash()
            return
        try:
            data = json.loads(self.question_category_map_file.read_text(encoding="utf-8"))
            mapping = data.get("question_category_map", {}) if isinstance(data, dict) else {}
            if isinstance(mapping, dict):
                self.question_category_map = {
                    str(k): {
                        "question_ru": str(v.get("question_ru", "")) if isinstance(v, dict) else "",
                        "category_code": str(v.get("category_code", "")) if isinstance(v, dict) else "",
                        "category_name": str(v.get("category_name", "")) if isinstance(v, dict) else "",
                    }
                    for k, v in mapping.items()
                }
        except Exception:
            self.question_category_map = {}
        self._update_question_map_hash()


    def _save_question_category_map(self) -> None:
        self._update_question_map_hash()
        payload = {
            "file_hash": self.question_file_hash,
            "fallback_code": self.questions_fallback_code,
            "question_category_map": self.question_category_map,
        }
        self.question_category_map_file.parent.mkdir(parents=True, exist_ok=True)
        self.question_category_map_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


    def _resolve_questions_result(self, parsed: dict) -> dict:
        if not isinstance(parsed, dict):
            return {"complaint_category": self.questions_fallback_code, "is_complaint": False}

        out = dict(parsed)
        q_code = str(out.get("primary_question_code") or out.get("question_code") or "").strip()
        if not q_code:
            out["complaint_category"] = self.questions_fallback_code
            out["is_complaint"] = False
            out["complaint_subcategory"] = None
            return out

        q_map = self._question_code_to_text()
        if q_code not in q_map:
            out["complaint_category"] = self.questions_fallback_code
            out["is_complaint"] = False
            out["complaint_subcategory"] = None
            return out

        mapped = self.question_category_map.get(q_code)
        if mapped and str(mapped.get("category_code", "")).strip():
            category_code = str(mapped.get("category_code"))
            category_name = str(mapped.get("category_name") or _short_category_name_from_question(mapped.get("question_ru") or q_map.get(q_code, "")))
        else:
            raw_name = str(out.get("category_name") or out.get("category_label") or out.get("category") or "").strip()
            question_text = q_map.get(q_code, "")
            if not raw_name or raw_name.strip().lower() == question_text.strip().lower():
                category_name = _short_category_name_from_question(question_text)
            else:
                category_name = raw_name
            category_code = _normalize_code(category_name)
            used = {str(v.get("category_code", "")) for v in self.question_category_map.values()}
            if category_code in used:
                category_code = f"{category_code}_{q_code}"
            self.question_category_map[q_code] = {
                "question_ru": q_map.get(q_code, ""),
                "category_code": category_code,
                "category_name": category_name,
            }
            self._save_question_category_map()

        out["complaint_category"] = category_code
        out["complaint_subcategory"] = None
        out["is_complaint"] = True
        trig = out.get("triggered_codes")
        q_ru = q_map.get(q_code, "")
        prefix = f"question_code={q_code}; question_ru={q_ru}; category_name={category_name}"
        if isinstance(trig, list):
            prefix = f"{prefix}; triggered_codes={trig}"
        out["notes"] = (prefix + "; " + str(out.get("notes", "")).strip()).strip("; ")
        return out


    def export_questions_categories_json(self) -> dict:
        return {
            "file_hash": self.question_file_hash,
            "fallback_code": self.questions_fallback_code,
            "questions": self.question_items,
            "question_category_map": self.question_category_map,
        }


    def _load_questions_mode(self) -> None:
        if self.category_mode != "questions":
            return
        loaded = load_questions(self.questions_file)
        self.question_items = loaded.get("items", [])
        self.question_file_hash = str(loaded.get("file_hash", ""))
        self.questions_fallback_code = str(loaded.get("fallback_code", "OTHER"))
        out = Path("data/interim/questions_taxonomy.json")
        need_save = True
        if out.exists():
            try:
                current = json.loads(out.read_text(encoding="utf-8"))
                need_save = str(current.get("file_hash", "")) != self.question_file_hash
            except Exception:
                need_save = True
        if need_save:
            save_questions_taxonomy(
                out,
                items=self.question_items,
                file_hash=self.question_file_hash,
                version=int(loaded.get("version", 1)),
            )
        self._load_question_category_map()

    def estimate_tokens(self, payload: dict) -> int:
        prompt = self._single_user_prompt(payload)
        if self.cfg.request_metrics_enabled and self.client and hasattr(self.client, "count_tokens"):
            token_input = f"{self._system_prompt()}\n{prompt}"
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

            user_prompt = self._batch_user_prompt(batch_indexes, batch_payloads)

            req_token_count = None
            if self.cfg.request_metrics_enabled and self.client and hasattr(self.client, "count_tokens"):
                token_input = f"{self._system_prompt()}\n{user_prompt}"
                req_token_count = self.client.count_tokens(model=self.cfg.model, input_text=token_input)
                logger.info("[stage=prepare/llm] tokens per batch request: %s", req_token_count if req_token_count is not None else "n/a")

            chat_started = time.perf_counter()
            try:
                response = self.client.chat(self._chat_payload(user_prompt))
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
                base_item = item.get("result") if (self.category_mode == "questions" and isinstance(item.get("result"), dict)) else item
                parsed_item = self._resolve_questions_result(base_item) if self.category_mode == "questions" else item
                obj = NormalizeTicket.model_validate(_coerce_response_fields(parsed_item, uncached_payloads[idx]))
                self._remember_discovered_category(item)
                self.cache.set(self._key(uncached_payloads[idx]), obj.model_dump())
                results[idx] = obj
                assigned.add(idx)

            unassigned_pending = [idx for idx in batch_indexes if idx not in assigned]
            for idx, item in zip(unassigned_pending, leftovers):
                base_item = item.get("result") if (self.category_mode == "questions" and isinstance(item.get("result"), dict)) else item
                parsed_item = self._resolve_questions_result(base_item) if self.category_mode == "questions" else item
                obj = NormalizeTicket.model_validate(_coerce_response_fields(parsed_item, uncached_payloads[idx]))
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
            if self.category_mode == "questions":
                if is_complaint and self.question_items:
                    pseudo = {"primary_question_code": self.question_items[0]["code"]}
                    resolved = self._resolve_questions_result(pseudo)
                    category = resolved.get("complaint_category", "OTHER")
                    notes = resolved.get("notes")
                else:
                    category = "OTHER"
                    notes = None
                resp = NormalizeTicket(
                    client_first_message=txt,
                    short_summary=txt[:120],
                    is_complaint=(category != "OTHER"),
                    complaint_category=str(category),
                    complaint_subcategory=None,
                    product_area=payload.get("product"),
                    loan_product="NONE",
                    severity="medium" if (category != "OTHER") else "low",
                    keywords=["жалоба", "вопрос", "сервис"] if (category != "OTHER") else ["вопрос", "инфо", "уточнение"],
                    confidence=0.8,
                    notes=notes,
                )
                self.cache.set(k, resp.model_dump())
                return resp
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
            token_input = f"{self._system_prompt()}\n{user_prompt}"
            req_token_count = self.client.count_tokens(model=self.cfg.model, input_text=token_input)
            logger.info("[stage=prepare/llm] tokens per request: %s", req_token_count if req_token_count is not None else "n/a")

        chat_started = time.perf_counter()
        try:
            response = self.client.chat(self._chat_payload(user_prompt))
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
            if self.category_mode == "questions":
                parsed = self._resolve_questions_result(parsed)
            obj = NormalizeTicket.model_validate(_coerce_response_fields(parsed, payload))
            self._remember_discovered_category(parsed)
        except Exception:
            repair_prompt = f"Исправь: верни JSON по схеме, убери запрещенные токены.\n{content}"
            try:
                response2 = self.client.chat(self._chat_payload(repair_prompt))
            except Exception as e:
                hinted = _normalize_error_with_tls_hint(self.cfg, e, phase="repair")
                if hinted is not None:
                    raise hinted from e
                raise
            repaired = json.loads(response2.choices[0].message.content)
            if self.category_mode == "questions":
                repaired = self._resolve_questions_result(repaired)
            obj = NormalizeTicket.model_validate(_coerce_response_fields(repaired, payload))
            self._remember_discovered_category(repaired)
        self.cache.set(k, obj.model_dump())
        return obj
