from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _question_code(index: int, question_ru: str) -> str:
    h = hashlib.sha1(question_ru.encode("utf-8")).hexdigest()[:8]
    return f"q_{index:03d}_{h}"


def load_questions(path: str | Path) -> dict:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("questions JSON must be an object")
    if "version" not in data:
        raise ValueError("questions JSON must contain version")
    categories = data.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("questions JSON must contain non-empty categories list")

    seen: set[str] = set()
    deduped: list[str] = []
    for item in categories:
        if not isinstance(item, dict):
            raise ValueError("each category item must be an object")
        q = item.get("question_ru")
        if not isinstance(q, str) or not q.strip():
            raise ValueError("question_ru must be a non-empty string")
        qs = q.strip()
        if qs in seen:
            continue
        seen.add(qs)
        deduped.append(qs)

    if not deduped:
        raise ValueError("questions list became empty after deduplication")

    items = [{"code": _question_code(i, q), "question_ru": q} for i, q in enumerate(deduped, start=1)]
    file_hash = hashlib.sha1(p.read_bytes()).hexdigest()
    return {
        "version": int(data["version"]),
        "items": items,
        "file_hash": file_hash,
        "fallback_code": "OTHER",
    }


def save_questions_taxonomy(out_path: str | Path, items: list[dict], file_hash: str, version: int) -> None:
    p = Path(out_path)
    payload = {
        "version": int(version),
        "file_hash": str(file_hash),
        "fallback_code": "OTHER",
        "items": [{"code": str(x.get("code", "")), "question_ru": str(x.get("question_ru", ""))} for x in items],
        "mapping": {str(x.get("code", "")): str(x.get("question_ru", "")) for x in items},
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
