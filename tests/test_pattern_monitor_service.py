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

    def load_pattern_monitor_pressure(self, tag: str) -> pd.DataFrame:
        return pd.DataFrame()

    def load_pattern_monitor_state(self, tag: str) -> pd.DataFrame:
        return pd.DataFrame()


def test_alerts_returns_empty_when_no_alert_flags_present():
    scored = pd.DataFrame(
        [
            {"row_id": "r1", "date": "2025-12-01", "category": "login", "pattern_like_score": 0.9, "is_pattern_alert": False, "row_dialog": "a"},
            {"row_id": "r2", "date": "2025-12-01", "category": "login", "pattern_like_score": 0.4, "is_pattern_alert": False, "row_dialog": "b"},
        ]
    )
    svc = PatternMonitorService(loader=_FakeLoader(scored))

    resp = svc.alerts("latest", {"top_n": 50})

    assert len(resp.rows) == 0


def test_alerts_does_not_drop_rows_when_upload_filter_column_missing():
    scored = pd.DataFrame(
        [
            {"row_id": "r1", "date": "2025-12-01", "category": "login", "pattern_like_score": 0.9, "is_pattern_alert": True, "row_dialog": "a"},
        ]
    )
    svc = PatternMonitorService(loader=_FakeLoader(scored))

    resp = svc.alerts("latest", {"upload_id": "upload-123", "top_n": 50})

    assert len(resp.rows) == 1


def test_run_output_rows_returns_rows_without_alert_filtering():
    scored = pd.DataFrame(
        [
            {"row_id": "r2", "category": "login", "pattern_like_score": 0.4, "is_pattern_alert": False, "row_dialog": "b"},
            {"row_id": "r1", "category": "login", "pattern_like_score": 0.9, "is_pattern_alert": True, "row_dialog": "a"},
        ]
    )
    svc = PatternMonitorService(loader=_FakeLoader(scored))
    resp = svc.run_output_rows("latest", top_n=10)
    assert [r["row_id"] for r in resp.rows] == ["r1"]
