from __future__ import annotations

import os
import threading
import time


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return max(0.0, float(raw))
    except Exception:
        return default


def rate_limit_delay(response: object, attempt: int) -> float:
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
    if retry_after:
        try:
            return max(0.2, min(30.0, float(retry_after)))
        except Exception:
            pass
    fallback_delays = (1.5, 3.0, 6.0, 10.0, 15.0)
    return fallback_delays[min(attempt, len(fallback_delays) - 1)]


class AdaptiveRateLimiter:
    def __init__(self, *, min_interval: float, max_interval: float) -> None:
        self._min_interval = min_interval
        self._max_interval = max(max_interval, min_interval)
        self._current_interval = min_interval
        self._next_allowed_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            scheduled_at = max(now, self._next_allowed_at)
            self._next_allowed_at = scheduled_at + self._current_interval
            delay = scheduled_at - now
        if delay > 0:
            time.sleep(delay)

    def backoff(self, response: object, attempt: int) -> None:
        delay = rate_limit_delay(response, attempt)
        with self._lock:
            now = time.monotonic()
            self._current_interval = min(self._max_interval, max(self._current_interval * 1.6, delay))
            self._next_allowed_at = max(self._next_allowed_at, now + delay)
        time.sleep(delay)

    def record_success(self) -> None:
        with self._lock:
            if self._current_interval > self._min_interval:
                self._current_interval = max(self._min_interval, self._current_interval * 0.85)


_LIMITERS: dict[str, AdaptiveRateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_rate_limiter(key: str) -> AdaptiveRateLimiter:
    min_interval = _env_float("GIGACHAT_MIN_REQUEST_INTERVAL_SECONDS", 1.0)
    max_interval = _env_float("GIGACHAT_MAX_REQUEST_INTERVAL_SECONDS", 20.0)
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = AdaptiveRateLimiter(min_interval=min_interval, max_interval=max_interval)
            _LIMITERS[key] = limiter
        return limiter
