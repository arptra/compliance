from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _setup(tmp_path: Path) -> Path:
    data = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    data["prepare"]["output_parquet"] = str(tmp_path / "all_prepared.parquet")
    data.setdefault("analysis", {}).setdefault("pattern_monitoring", {})["interim_dir"] = str(tmp_path)

    df = pd.DataFrame(
        {
            "event_time": pd.date_range("2025-01-01", periods=20, freq="D"),
            "category": ["A"] * 12 + ["B"] * 8,
            "subcategory": ["A1"] * 6 + ["A2"] * 6 + ["B1"] * 8,
        }
    )
    df.to_parquet(tmp_path / "all_prepared.parquet", index=False)

    cfg = tmp_path / "project.yaml"
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return cfg


def test_categories_with_baseline_and_subcategories(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))

    table = client.get(
        "/api/categories",
        params={
            "date_from": "2025-01-10",
            "date_to": "2025-01-20",
            "baseline_mode": "previous_period",
        },
    )
    assert table.status_code == 200
    rows = table.json()["rows"]
    assert len(rows) > 0
    assert "delta_abs" in rows[0]
    assert "baseline_count" in rows[0]

    subs = client.get("/api/categories/A/subcategories", params={"date_from": "2025-01-01", "date_to": "2025-01-20"})
    assert subs.status_code == 200
    assert len(subs.json()["subcategories"]) > 0


def test_categories_mapped_from_prepare_llm_columns(tmp_path: Path):
    data = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    data["prepare"]["output_parquet"] = str(tmp_path / "all_prepared.parquet")
    data.setdefault("analysis", {}).setdefault("pattern_monitoring", {})["interim_dir"] = str(tmp_path)

    df = pd.DataFrame({
        "event_time": pd.date_range("2025-01-01", periods=4, freq="D"),
        "complaint_category_llm": ["PAYMENTS", "PAYMENTS", "LOGIN", "LOGIN"],
        "complaint_subcategory_llm": ["CARD", "CARD", "OTP", "PASSWORD"],
        "is_complaint_llm": [True, True, True, True],
    })
    df.to_parquet(tmp_path / "all_prepared.parquet", index=False)

    cfg = tmp_path / "project.yaml"
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    client = TestClient(create_app(str(cfg)))
    r = client.get("/api/categories", params={"date_from": "2025-01-01", "date_to": "2025-01-04"})
    assert r.status_code == 200
    cats = [x["category"] for x in r.json()["rows"]]
    assert "PAYMENTS" in cats
    assert "LOGIN" in cats
