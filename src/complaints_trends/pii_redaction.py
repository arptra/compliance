from __future__ import annotations

import re

from .config import PIIConfig


EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+?\d[\d\-() ]{8,}\d)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
ACCOUNT_RE = re.compile(r"\b\d{20}\b")


def redact_pii(text: str, cfg: PIIConfig) -> str:
    out = str(text or "")
    out = EMAIL_RE.sub(cfg.replace_email, out)
    out = PHONE_RE.sub(cfg.replace_phone, out)
    out = URL_RE.sub(cfg.replace_url, out)
    out = CARD_RE.sub(cfg.replace_card, out)
    out = ACCOUNT_RE.sub(cfg.replace_account, out)
    return out
