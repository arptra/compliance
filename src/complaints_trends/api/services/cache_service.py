from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    expires_at: datetime


class InMemoryCache:
    def __init__(self, ttl_seconds: int = 60) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if not entry:
            return None
        if datetime.now(timezone.utc) >= entry.expires_at:
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = CacheEntry(value=value, expires_at=datetime.now(timezone.utc) + self.ttl)

    def invalidate_prefix(self, prefix: str) -> None:
        for key in list(self._store.keys()):
            if key.startswith(prefix):
                self._store.pop(key, None)
