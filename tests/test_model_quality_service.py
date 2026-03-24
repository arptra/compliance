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


def test_model_quality_compare_modes(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    client.post("/api/feedback", json={"row_id": "a", "pattern_tag": "latest", "verdict": "true", "category": "A", "base_score": 0.9, "rerank_score": 0.95})
    client.post("/api/feedback", json={"row_id": "b", "pattern_tag": "latest", "verdict": "false", "category": "A", "base_score": 0.8, "rerank_score": 0.2})

    resp = client.get("/api/model-quality", params={"pattern_tag": "latest"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["compare_modes"]) == 3
    assert body["compare_modes"][0]["mode"] == "base"
    assert body["compare_modes"][2]["mode"] == "reranked"
    assert body["compare_modes"][0]["precision_at"][2]["k"] == 50
    assert body["compare_modes"][0]["by_category"]
    assert body["compare_modes"][0]["by_score_bucket"]
