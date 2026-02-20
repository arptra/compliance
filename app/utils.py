from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import yaml


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@contextmanager
def timed_step(name: str):
    logger = logging.getLogger("pipeline")
    start = time.time()
    logger.info("Start: %s", name)
    try:
        yield
    finally:
        logger.info("Done: %s (%.2fs)", name, time.time() - start)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(data: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
