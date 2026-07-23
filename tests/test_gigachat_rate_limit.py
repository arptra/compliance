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


def test_429_uses_one_serial_probe_then_releases_queue_in_parallel(monkeypatch, caplog) -> None:
    import complaints_trends.gigachat_api.rate_limit as rate_limit

    monkeypatch.setattr(rate_limit, "rate_limit_delay", lambda response, attempt: 0)
    caplog.set_level(logging.INFO, logger="uvicorn.error.gigachat")
    limiter = AdaptiveRateLimiter(key="test:429", min_interval=0, max_interval=0)
    first_response_started = threading.Event()
    serial_probe_started = threading.Event()
    release_serial_probe = threading.Event()
    call_lock = threading.Lock()
    calls = 0
    parallel_barrier = threading.Barrier(3)
    parallel_lock = threading.Lock()
    parallel_active = 0
    max_parallel_active = 0

    def parallel_success() -> _Response:
        nonlocal parallel_active, max_parallel_active
        with parallel_lock:
            parallel_active += 1
            max_parallel_active = max(max_parallel_active, parallel_active)
        try:
            parallel_barrier.wait(timeout=2)
            time.sleep(0.01)
            return _Response(200)
        finally:
            with parallel_lock:
                parallel_active -= 1

    def rate_limited_then_probe() -> _Response:
        nonlocal calls
        with call_lock:
            calls += 1
            current_call = calls
        if current_call == 1:
            first_response_started.set()
            return _Response(429, retry_after="0")
        serial_probe_started.set()
        assert release_serial_probe.wait(timeout=2)
        return _Response(200)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(limiter.execute, rate_limited_then_probe)
        assert first_response_started.wait(timeout=1)

        deadline = time.monotonic() + 1
        while limiter.snapshot()["mode"] != "GLOBAL_SERIAL_QUEUE":
            assert time.monotonic() < deadline
            time.sleep(0.001)

        assert serial_probe_started.wait(timeout=1)
        others = [executor.submit(limiter.execute, parallel_success) for _ in range(3)]
        deadline = time.monotonic() + 1
        while limiter.snapshot()["queued"] != 3:
            assert time.monotonic() < deadline
            time.sleep(0.001)

        release_serial_probe.set()
        responses = [first.result(timeout=2), *(future.result(timeout=2) for future in others)]

    assert [response.status_code for response in responses] == [200] * 4
    assert max_parallel_active == 3
    assert limiter.snapshot()["mode"] == "PARALLEL"
    limiter.execute(lambda: _Response(429), max_attempts=1)
    assert limiter.snapshot()["mode"] == "GLOBAL_SERIAL_QUEUE"
    assert "[GIGACHAT_RATE_LIMIT] status=429" in caplog.text
    assert "action=global_serial_queue" in caplog.text
    assert "global FIFO queue; concurrency=1 until_successful_probe" in caplog.text
    assert "event=serial_probe_succeeded" in caplog.text
    assert "released_queued=3 next_mode=parallel" in caplog.text
