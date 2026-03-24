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


def test_feedback_dataset_filters_and_export(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    client.post("/api/feedback", json={"row_id": "r1", "pattern_tag": "latest", "verdict": "true", "category": "BILLING", "subcategory": "duplicate_charge", "reviewer": "ann", "comment": "good"})
    client.post("/api/feedback", json={"row_id": "r2", "pattern_tag": "latest", "verdict": "false", "category": "B", "reviewer": "bob", "comment": "bad"})

    resp = client.get("/api/feedback/dataset", params={"reviewer": "ann", "q": "good"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["row_id"] == "r1"
    assert body["items"][0]["category_label_ru"] == "Платежи и списания"
    assert body["items"][0]["subcategory_label_ru"] == "Двойное списание"

    export_resp = client.get("/api/feedback/export", params={"output_format": "json", "reviewer": "ann"})
    assert export_resp.status_code == 200
    assert "r1" in export_resp.text
