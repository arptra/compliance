from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app
from complaints_trends.config import load_config
from complaints_trends.pattern_fit import run_pattern_fit


def _setup(tmp_path: Path) -> Path:
    data = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    data["prepare"]["output_parquet"] = str(tmp_path / "prepared.parquet")
    data.setdefault("analysis", {}).setdefault("pattern_monitoring", {})["interim_dir"] = str(tmp_path / "interim")
    data["analysis"]["pattern_monitoring"]["exports_dir"] = str(tmp_path / "exports")
    data["analysis"]["pattern_monitoring"]["reports_dir"] = str(tmp_path / "reports")
    data.setdefault("training", {})["model_dir"] = str(tmp_path / "models")

    rows = []
    for m in ["2025-01", "2025-02"]:
        for i in range(30):
            rows.append({
                "row_id": f"n-{m}-{i}",
                "month": m,
                "event_time": f"{m}-10 10:00:00",
                "client_first_message": "не работает кнопка входа",
                "is_complaint_llm": True,
                "complaint_category_llm": "login",
            })
    for m in ["2025-03"]:
        for i in range(20):
            rows.append({
                "row_id": f"e-{m}-{i}",
                "month": m,
                "event_time": f"{m}-10 10:00:00",
                "client_first_message": "после смс нельзя завершить вход",
                "is_complaint_llm": True,
                "complaint_category_llm": "login",
            })
    pd.DataFrame(rows).to_parquet(data["prepare"]["output_parquet"], index=False)

    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    cfg = load_config(cfg_path)
    run_pattern_fit(cfg, tag="fitA", normal_period="2025-01..2025-02", event_period="2025-03..2025-03", label_source="llm")
    return cfg_path


def test_run_pattern_monitor_accepts_separate_fit_tag(tmp_path: Path):
    cfg_path = _setup(tmp_path)
    client = TestClient(create_app(str(cfg_path)))
    r = client.post(
        "/api/runs/pattern-monitor",
        json={"params": {"tag": "monitorA", "fit_tag": "fitA", "label_source": "llm", "date_from": "2025-03-01", "date_to": "2025-03-31"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert "scored" in body["outputs"]
