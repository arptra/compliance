from __future__ import annotations

import importlib

import pandas as pd
import pytest

pytest.importorskip("fastapi")

PatternMonitorService = importlib.import_module("complaints_trends.api.services.pattern_monitor_service").PatternMonitorService


class _FakeLoader:
    def __init__(self, scored: pd.DataFrame) -> None:
        self._scored = scored

    def resolve_tag(self, family: str, tag: str) -> str:
        return tag

    def load_pattern_monitor_scored(self, tag: str) -> pd.DataFrame:
        return self._scored.copy()


def test_alerts_returns_scored_rows_when_no_alert_flags_present():
    scored = pd.DataFrame(
        [
            {"row_id": "r1", "date": "2025-12-01", "category": "login", "pattern_like_score": 0.9, "is_pattern_alert": False, "row_dialog": "a"},
            {"row_id": "r2", "date": "2025-12-01", "category": "login", "pattern_like_score": 0.4, "is_pattern_alert": False, "row_dialog": "b"},
        ]
    )
    svc = PatternMonitorService(loader=_FakeLoader(scored))

    resp = svc.alerts("latest", {"top_n": 50})

    assert len(resp.rows) == 2
    assert all(bool(r.get("no_alerts_in_selection")) for r in resp.rows)
