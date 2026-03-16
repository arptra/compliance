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
    d = tmp_path / "pattern_fit_latest"
    d.mkdir()
    pd.DataFrame({"category": ["A"], "delta": [12]}).to_parquet(d / "category_growth_summary.parquet", index=False)
    pd.DataFrame({"category": ["A"], "seed": ["s"]}).to_parquet(d / "seed_pool.parquet", index=False)
    pd.DataFrame({"category": ["A"], "cluster_id": [1]}).to_parquet(d / "cluster_members.parquet", index=False)
    (d / "cluster_profiles.json").write_text('{"A": [{"cluster_id": 1, "terms": ["error"]}]}', encoding="utf-8")
    cfg = tmp_path / "project.yaml"
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return cfg


def test_pattern_fit_endpoints(tmp_path: Path):
    cfg = _setup(tmp_path)
    client = TestClient(create_app(str(cfg)))
    assert client.get("/api/pattern-fit/summary", params={"tag": "latest"}).status_code == 200
    assert client.get("/api/pattern-fit/category/A/clusters", params={"tag": "latest"}).status_code == 200
