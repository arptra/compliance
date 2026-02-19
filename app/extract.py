from __future__ import annotations

import re
from typing import Any, Dict, List


def _compile_markers(markers: List[str]) -> re.Pattern:
    escaped = [re.escape(m) for m in markers if m]
    if not escaped:
        return re.compile(r"$a")
    return re.compile(r"(?im)\b(" + "|".join(escaped) + r")\b")


def extract_first_client_message(text: str, cfg: Dict[str, Any]) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    role_cfg = cfg.get("input", {}).get("role_parsing", {})
    if not role_cfg.get("enabled", True):
        return text

    client_markers = role_cfg.get("client_markers", ["CLIENT", "КЛИЕНТ", "ПОЛЬЗОВАТЕЛЬ", "USER"])
    stop_markers = role_cfg.get("stop_on_markers", ["OPERATOR", "ОПЕРАТОР", "CHATBOT", "БОТ", "SUPPORT", "АГЕНТ"])
    client_prefix_regexes = role_cfg.get("client_prefix_regexes", [])

    # 1) find first explicit client prefix line
    start = None
    for pat in client_prefix_regexes:
        m = re.search(pat, text)
        if m and (start is None or m.start() < start):
            start = m.end()

    # 2) fallback: first marker occurrence
    if start is None:
        cm = _compile_markers(client_markers).search(text)
        if cm:
            start = cm.end()

    if start is not None:
        tail = text[start:]
        sm = _compile_markers(stop_markers).search(tail)
        return (tail[: sm.start()] if sm else tail).strip(" -:\n\t")

    # fallback
    mode = role_cfg.get("fallback_mode", "first_paragraph")
    if mode == "first_n_chars":
        n = int(role_cfg.get("fallback_first_n_chars", 400))
        return text[:n].strip()

    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return (parts[0] if parts else text).strip()
