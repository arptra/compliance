from __future__ import annotations

import re
from dataclasses import dataclass

from .config import ClientFirstConfig


@dataclass
class Segment:
    role: str
    text: str


def _detect_role(line: str, cfg: ClientFirstConfig) -> str | None:
    u = line.upper()
    for m in cfg.client_markers:
        if m.upper() in u:
            return "client"
    for m in cfg.operator_markers:
        if m.upper() in u:
            return "operator"
    for m in cfg.chatbot_markers:
        if m.upper() in u:
            return "chatbot"
    return None


def _strip_marker(line: str) -> str:
    return re.sub(r"^[\[\(\s]*[A-ZА-Я_ ]{2,20}[:\-]\s*", "", line.strip(), flags=re.IGNORECASE)


def split_dialog(dialog: str, cfg: ClientFirstConfig) -> list[Segment]:
    segments: list[Segment] = []
    for ln in str(dialog).splitlines():
        role = _detect_role(ln, cfg)
        if role:
            segments.append(Segment(role=role, text=_strip_marker(ln)))
    return segments


def extract_client_first_message(dialog: str, cfg: ClientFirstConfig) -> str:
    text = str(dialog or "").strip()
    if not text:
        return ""
    segments = split_dialog(text, cfg)
    client_msgs = [s.text for s in segments if s.role == "client" and s.text.strip()]
    if client_msgs:
        first = client_msgs[0]
        if len(first) < cfg.min_client_len and cfg.take_second_client_if_too_short and len(client_msgs) > 1:
            return client_msgs[1].strip()
        return first.strip()
    if cfg.fallback_mode == "first_paragraph":
        return text.split("\n\n", 1)[0][: cfg.fallback_first_n_chars].strip()
    return text[: cfg.fallback_first_n_chars].strip()
