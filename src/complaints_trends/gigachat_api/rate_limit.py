from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import Callable, TypeVar


logger = logging.getLogger("uvicorn.error.gigachat")
ResponseT = TypeVar("ResponseT")


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


def retry_after_value(response: object) -> str | None:
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") if hasattr(headers, "get") else None
    if value in (None, ""):
        return None
    return str(value).strip()[:128] or None


@dataclass(frozen=True)
class _RequestLease:
    serial: bool


class AdaptiveRateLimiter:
    def __init__(self, *, key: str, min_interval: float, max_interval: float) -> None:
        self._key = key
        self._min_interval = min_interval
        self._max_interval = max(max_interval, min_interval)
        self._current_interval = min_interval
        self._next_allowed_at = 0.0
        self._retry_not_before = 0.0
        self._serial_mode = False
        self._active_requests = 0
        self._next_waiter_id = 0
        self._wait_queue: deque[int] = deque()
        self._condition = threading.Condition()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "key": self._key,
                "mode": "GLOBAL_SERIAL_QUEUE" if self._serial_mode else "PARALLEL",
                "active": self._active_requests,
                "queued": len(self._wait_queue),
                "concurrency": 1 if self._serial_mode else None,
                "retry_in_seconds": max(0.0, self._retry_not_before - time.monotonic()),
            }

    def _acquire(self) -> _RequestLease:
        thread_name = threading.current_thread().name
        with self._condition:
            waiter_id = self._next_waiter_id
            self._next_waiter_id += 1
            self._wait_queue.append(waiter_id)

            while True:
                now = time.monotonic()
                at_front = bool(self._wait_queue and self._wait_queue[0] == waiter_id)
                serial_slot_available = not self._serial_mode or self._active_requests == 0
                ready_at = max(self._next_allowed_at, self._retry_not_before)
                if at_front and serial_slot_available and now >= ready_at:
                    self._wait_queue.popleft()
                    lease = _RequestLease(serial=self._serial_mode)
                    self._active_requests += 1
                    self._next_allowed_at = now + self._current_interval
                    active = self._active_requests
                    queued = len(self._wait_queue)
                    mode = "serial" if lease.serial else "parallel"
                    self._condition.notify_all()
                    break

                timeout = None
                if at_front and serial_slot_available:
                    timeout = max(0.001, ready_at - now)
                self._condition.wait(timeout=timeout)

        logger.info(
            "[GIGACHAT_POOL] event=request_dispatched limiter=%s mode=%s "
            "active=%s queued=%s thread=%s",
            self._key,
            mode,
            active,
            queued,
            thread_name,
        )
        return lease

    def _finish(
        self,
        lease: _RequestLease,
        response: object,
        attempt: int,
        *,
        will_retry: bool,
    ) -> None:
        status_code = int(getattr(response, "status_code", 0) or 0)
        rate_limited = status_code == 429
        successful = 0 < status_code < 400
        transitioned = False
        recovered = False
        released_queued = 0
        delay = 0.0
        retry_after = None

        with self._condition:
            self._active_requests = max(0, self._active_requests - 1)
            if rate_limited:
                delay = rate_limit_delay(response, attempt)
                retry_after = retry_after_value(response)
                transitioned = not self._serial_mode
                self._serial_mode = True
                now = time.monotonic()
                self._current_interval = min(
                    self._max_interval,
                    max(self._current_interval * 1.6, delay),
                )
                self._retry_not_before = max(self._retry_not_before, now + delay)
                self._next_allowed_at = max(self._next_allowed_at, self._retry_not_before)
            elif successful:
                if self._current_interval > self._min_interval:
                    self._current_interval = max(
                        self._min_interval,
                        self._current_interval * 0.85,
                    )
                if lease.serial:
                    now = time.monotonic()
                    if self._active_requests == 0 and now >= self._retry_not_before:
                        released_queued = len(self._wait_queue)
                        self._serial_mode = False
                        self._current_interval = self._min_interval
                        self._retry_not_before = 0.0
                        self._next_allowed_at = now
                        recovered = True

            active = self._active_requests
            queued = len(self._wait_queue) + (1 if rate_limited and will_retry else 0)
            mode = "serial" if self._serial_mode else "parallel"
            self._condition.notify_all()

        if rate_limited:
            logger.warning(
                "[GIGACHAT_RATE_LIMIT] status=429 limiter=%s attempt=%s "
                "retry_after=%s delay_seconds=%.3f action=global_serial_queue",
                self._key,
                attempt + 1,
                retry_after or "unknown",
                delay,
            )
            logger.warning(
                "[GIGACHAT_QUEUE] parallel dispatch paused; requests moved to "
                "the global FIFO queue; concurrency=1 until_successful_probe "
                "queued=%s active=%s",
                queued,
                active,
            )
            if transitioned:
                logger.info(
                    "[GIGACHAT_POOL] event=mode_changed limiter=%s "
                    "from=parallel to=serial concurrency=1",
                    self._key,
                )
        else:
            logger.info(
                "[GIGACHAT_POOL] event=request_completed limiter=%s status=%s "
                "mode=%s active=%s queued=%s",
                self._key,
                status_code or "exception",
                mode,
                active,
                queued,
            )
        if recovered:
            logger.info(
                "[GIGACHAT_POOL] event=serial_probe_succeeded limiter=%s "
                "released_queued=%s next_mode=parallel",
                self._key,
                released_queued,
            )

    def _finish_exception(self, lease: _RequestLease) -> None:
        with self._condition:
            self._active_requests = max(0, self._active_requests - 1)
            active = self._active_requests
            queued = len(self._wait_queue)
            mode = "serial" if self._serial_mode else "parallel"
            self._condition.notify_all()
        logger.info(
            "[GIGACHAT_POOL] event=request_failed limiter=%s mode=%s "
            "active=%s queued=%s",
            self._key,
            mode,
            active,
            queued,
        )

    def execute(self, request: Callable[[], ResponseT], *, max_attempts: int = 6) -> ResponseT:
        response: ResponseT | None = None
        for attempt in range(max(1, max_attempts)):
            lease = self._acquire()
            try:
                response = request()
            except BaseException:
                self._finish_exception(lease)
                raise
            status_code = int(getattr(response, "status_code", 0) or 0)
            will_retry = status_code == 429 and attempt + 1 < max_attempts
            self._finish(lease, response, attempt, will_retry=will_retry)
            if not will_retry:
                return response
        if response is None:
            raise RuntimeError("Rate-limited request was not executed")
        return response


_LIMITERS: dict[str, AdaptiveRateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_rate_limiter(key: str) -> AdaptiveRateLimiter:
    min_interval = _env_float("GIGACHAT_MIN_REQUEST_INTERVAL_SECONDS", 0.0)
    max_interval = _env_float("GIGACHAT_MAX_REQUEST_INTERVAL_SECONDS", 20.0)
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = AdaptiveRateLimiter(
                key=key,
                min_interval=min_interval,
                max_interval=max_interval,
            )
            _LIMITERS[key] = limiter
        return limiter
