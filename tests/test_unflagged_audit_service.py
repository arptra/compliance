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
    pd.DataFrame(
        {
            "event_time": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "category": ["A", "A", "B"],
            "subcategory": ["X", "X", "Y"],
            "row_dialog": ["one", "two", "three"],
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        }
    ).to_parquet(tmp_path / "all_prepared.parquet", index=False)
    d = tmp_path / "pattern_monitor_latest"
    d.mkdir()
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "category": ["A", "A"],
            "subcategory": ["X", "X"],
            "pattern_like_score": [0.9, 0.8],
            "is_pattern_alert": [True, False],
            "row_dialog": ["one", "two"],
        }
    ).to_parquet(d / "scored_rows.parquet", index=False)
    pd.DataFrame({"date": ["2025-01-01"], "category": ["A"], "pressure": [0.7]}).to_parquet(d / "category_daily_pressure.parquet", index=False)
    pd.DataFrame({"date": ["2025-01-01"], "state": [0.5]}).to_parquet(d / "overall_daily_state.parquet", index=False)
    cfg = tmp_path / "project.yaml"
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return cfg


def test_unflagged_audit_create_review_estimate(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    created = client.post(
        "/api/audit/unflagged/create",
        json={"pattern_tag": "latest", "date_from": "2025-01-01", "date_to": "2025-01-03", "sample_size": 2, "random_seed": 7},
    )
    assert created.status_code == 200
    sample_id = created.json()["sample_id"]
    rows = created.json()["rows"]
    assert len(rows) >= 1

    review_payload = {"rows": [{"row_id": rows[0]["row_id"], "review_verdict": "true", "reviewer": "qa"}]}
    reviewed = client.post(f"/api/audit/unflagged/{sample_id}/review", json=review_payload)
    assert reviewed.status_code == 200
    estimate = reviewed.json()["estimate"]
    assert estimate["reviewed_in_sample"] >= 1
    assert estimate["true_in_sample"] >= 1
    assert estimate["estimated_hidden_positive_rate"] is not None
