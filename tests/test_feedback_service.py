from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _setup(tmp_path: Path):
    data = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    data["prepare"]["output_parquet"] = str(tmp_path / "all_prepared.parquet")
    data.setdefault("analysis", {}).setdefault("pattern_monitoring", {})["interim_dir"] = str(tmp_path)
    data.setdefault("training", {})["model_dir"] = str(tmp_path / "models")
    pd.DataFrame({"event_time": ["2025-01-01"], "category": ["A"]}).to_parquet(tmp_path / "all_prepared.parquet", index=False)
    d = tmp_path / "pattern_monitor_latest"
    d.mkdir()
    pd.DataFrame({"date": ["2025-01-01"], "category": ["A"], "subcategory": ["SUB_A"], "pattern_like_score": [0.9], "is_pattern_alert": [True], "row_dialog": ["dialog"]}).to_parquet(d / "scored_rows.parquet", index=False)
    pd.DataFrame({"date": ["2025-01-01"], "category": ["A"], "pressure": [0.7]}).to_parquet(d / "category_daily_pressure.parquet", index=False)
    pd.DataFrame({"date": ["2025-01-01"], "state": [0.5]}).to_parquet(d / "overall_daily_state.parquet", index=False)
    cfg = tmp_path / "project.yaml"
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return cfg


def test_feedback_single_create_and_summary(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    alerts = client.get('/api/pattern-monitor/alerts', params={'pattern_tag': 'latest'}).json()['rows']
    row_id = alerts[0]['row_id']
    r = client.post('/api/feedback', json={'row_id': row_id, 'pattern_tag': 'latest', 'verdict': 'true', 'category': 'A'})
    assert r.status_code == 200
    s = client.get('/api/feedback/summary', params={'pattern_tag': 'latest'})
    assert s.status_code == 200
    assert s.json()['reviewed_rows'] == 1
    assert s.json()['precision_reviewed'] == 1.0
