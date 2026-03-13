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
    pd.DataFrame({"event_time": ["2025-01-01"], "category": ["A"]}).to_parquet(tmp_path / "all_prepared.parquet", index=False)
    d = tmp_path / "pattern_monitor_latest"
    d.mkdir()
    pd.DataFrame({"date": ["2025-01-01"], "category": ["A"], "row_score": [0.9], "is_alert": [True]}).to_parquet(d / "scored_rows.parquet", index=False)
    pd.DataFrame({"date": ["2025-01-01"], "category": ["A"], "pressure": [0.7]}).to_parquet(d / "category_daily_pressure.parquet", index=False)
    pd.DataFrame({"date": ["2025-01-01"], "state": [0.5]}).to_parquet(d / "overall_daily_state.parquet", index=False)
    cfg = tmp_path / "project.yaml"
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return cfg


def test_pattern_monitor_endpoints(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    r = client.get("/api/pattern-monitor/summary", params={"pattern_tag": "latest"})
    assert r.status_code == 200
    assert r.json()["summary"]["alert_rows"] == 1
    assert client.get("/api/pattern-monitor/alerts", params={"pattern_tag": "latest"}).status_code == 200



def test_pattern_monitor_latest_tag_resolves(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    r = client.get("/api/pattern-monitor/summary", params={"pattern_tag": "latest"})
    assert r.status_code == 200
    assert r.json()["tag"] == "latest"
