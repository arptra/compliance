from __future__ import annotations

from pathlib import Path

import pandas as pd

from complaints_trends.pattern_fit import run_pattern_fit
from complaints_trends.pattern_monitor import run_pattern_monitor
from tests.test_pattern_fit import _cfg


def test_pattern_monitor_scores_event_like_rows(tmp_path: Path):
    cfg = _cfg(tmp_path)
    rows = []
    for m in ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08"]:
        for i in range(80):
            rows.append({
                "row_id": f"n-{m}-{i}",
                "month": m,
                "event_time": f"{m}-10 10:00:00",
                "client_first_message": "не работает кнопка входа",
                "is_complaint_llm": True,
                "complaint_category_llm": "login",
            })
    for m in ["2025-11", "2025-12"]:
        for i in range(50):
            txt = "после ввода кода подтверждения кнопка входа не активна" if i < 30 else "не работает кнопка входа"
            rows.append({
                "row_id": f"e-{m}-{i}",
                "month": m,
                "event_time": f"{m}-11 10:00:00",
                "client_first_message": txt,
                "is_complaint_llm": True,
                "complaint_category_llm": "login",
            })

    target_day = "2025-12-15"
    for i in range(20):
        rows.append({
            "row_id": f"t-normal-{i}",
            "month": "2025-12",
            "event_time": f"{target_day} 11:00:00",
            "client_first_message": "не работает кнопка входа",
            "is_complaint_llm": True,
            "complaint_category_llm": "login",
        })
    alert_ids = []
    for i in range(2):
        rid = f"t-alert-{i}"
        alert_ids.append(rid)
        rows.append({
            "row_id": rid,
            "month": "2025-12",
            "event_time": f"{target_day} 12:00:00",
            "client_first_message": "на шаге подтверждения входа кнопка не нажимается",
            "is_complaint_llm": True,
            "complaint_category_llm": "login",
        })

    pd.DataFrame(rows).to_parquet(cfg.prepare.output_parquet, index=False)
    run_pattern_fit(cfg, tag="t2", normal_period="2025-01..2025-08", event_period="2025-11..2025-12", label_source="llm")
    scored_path, state_path, _ = run_pattern_monitor(cfg, tag="t2", label_source="llm", date_from=target_day, date_to=target_day)

    scored = pd.read_parquet(scored_path)
    assert "pattern_like_score" in scored.columns
    median_score = scored["pattern_like_score"].median()
    alert_rows = scored[scored["row_id"].isin(alert_ids)]
    assert (alert_rows["pattern_like_score"] > median_score).all()

    state = pd.read_parquet(state_path)
    assert not state.empty
    cat_pressure = pd.read_parquet(Path(cfg.analysis.pattern_monitoring.interim_dir) / "pattern_monitor_t2" / "category_daily_pressure.parquet")
    assert (cat_pressure["category_pressure"] > 0).any()
