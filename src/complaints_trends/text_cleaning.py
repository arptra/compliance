from __future__ import annotations

import re
from pathlib import Path


def load_tokens(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {ln.strip().lower() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}


def clean_for_model(text: str, deny_tokens: set[str]) -> str:
    out = str(text or "").lower()
    out = re.sub(r"<[^>]+>", " ", out)
    out = re.sub(r"\b\d+\b", " ", out)
    words = re.findall(r"[\wа-яА-Я]+", out)
    words = [w for w in words if w not in deny_tokens and len(w) > 1]
    return " ".join(words)
