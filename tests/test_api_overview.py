from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _config(tmp_path: Path) -> Path:
    data = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    data["prepare"]["output_parquet"] = str(tmp_path / "all_prepared.parquet")
    data.setdefault("analysis", {}).setdefault("pattern_monitoring", {})["interim_dir"] = str(tmp_path)
    p = tmp_path / "project.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


def test_overview_endpoint_and_baseline_modes(tmp_path: Path):
    cfg = _config(tmp_path)
    df = pd.DataFrame({"event_time": pd.date_range("2025-01-01", periods=20, freq="D"), "category": ["A"] * 10 + ["B"] * 10, "subcategory": ["x"] * 20})
    df.to_parquet(tmp_path / "all_prepared.parquet", index=False)

    client = TestClient(create_app(str(cfg)))
    for mode in ["previous_period", "same_weekday", "seasonal", "custom_range"]:
        r = client.get("/api/overview", params={"date_from": "2025-01-10", "date_to": "2025-01-20", "baseline_mode": mode, "baseline_date_from": "2025-01-01", "baseline_date_to": "2025-01-09"})
        assert r.status_code == 200
        payload = r.json()
        assert "kpis" in payload
        assert "actual_vs_expected" in payload
