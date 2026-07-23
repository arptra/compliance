from __future__ import annotations

import concurrent.futures
import logging
import threading
import time

from complaints_trends.gigachat_api.rate_limit import AdaptiveRateLimiter


class _Response:
    def __init__(self, status_code: int, *, retry_after: str | None = None) -> None:
        self.status_code = status_code
        self.headers = {"Retry-After": retry_after} if retry_after is not None else {}


def test_limiter_allows_configured_callers_to_overlap_before_429() -> None:
    limiter = AdaptiveRateLimiter(key="test:parallel", min_interval=0, max_interval=0)
    barrier = threading.Barrier(8)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def request() -> _Response:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=2)
            time.sleep(0.01)
            return _Response(200)
        finally:
            with lock:
                active -= 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(lambda _: limiter.execute(request), range(8)))

    assert [response.status_code for response in responses] == [200] * 8
    assert max_active == 8


def test_429_moves_waiting_requests_to_one_serial_fifo_queue(monkeypatch, caplog) -> None:
    import complaints_trends.gigachat_api.rate_limit as rate_limit

    monkeypatch.setattr(rate_limit, "rate_limit_delay", lambda response, attempt: 0)
    caplog.set_level(logging.INFO, logger="uvicorn.error.gigachat")
    limiter = AdaptiveRateLimiter(key="test:429", min_interval=0, max_interval=0)
    first_response_started = threading.Event()
    release_serial_request = threading.Event()
    call_lock = threading.Lock()
    calls = 0
    serial_lock = threading.Lock()
    serial_active = 0
    max_serial_active = 0

    def serial_success() -> _Response:
        nonlocal serial_active, max_serial_active
        with serial_lock:
            serial_active += 1
            max_serial_active = max(max_serial_active, serial_active)
        try:
            release_serial_request.wait(timeout=2)
            time.sleep(0.01)
            return _Response(200)
        finally:
            with serial_lock:
                serial_active -= 1

    def rate_limited_then_success() -> _Response:
        nonlocal calls
        with call_lock:
            calls += 1
            current_call = calls
        if current_call == 1:
            first_response_started.set()
            return _Response(429, retry_after="0")
        return serial_success()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(limiter.execute, rate_limited_then_success)
        assert first_response_started.wait(timeout=1)

        deadline = time.monotonic() + 1
        while limiter.snapshot()["mode"] != "GLOBAL_SERIAL_QUEUE":
            assert time.monotonic() < deadline
            time.sleep(0.001)

        others = [executor.submit(limiter.execute, serial_success) for _ in range(3)]
        release_serial_request.set()
        responses = [first.result(timeout=2), *(future.result(timeout=2) for future in others)]

    assert [response.status_code for response in responses] == [200] * 4
    assert max_serial_active == 1
    assert limiter.snapshot()["mode"] == "PARALLEL"
    assert "[GIGACHAT_RATE_LIMIT] status=429" in caplog.text
    assert "action=global_serial_queue" in caplog.text
    assert "global FIFO queue; concurrency=1" in caplog.text
