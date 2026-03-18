from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from complaints_trends.api.services.data_loader import DataLoader
from complaints_trends.config import load_config


def test_read_parquet_with_missing_projection_columns_returns_available_subset(tmp_path: Path):
    data = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    prepared = tmp_path / "all_prepared.parquet"
    data["prepare"]["output_parquet"] = str(prepared)
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    pd.DataFrame({"event_time": ["2025-01-01"], "category": ["A"]}).to_parquet(prepared, index=False)

    loader = DataLoader(load_config(str(cfg_path)))
    out = loader.read_parquet(prepared, columns=["event_time", "label_source"])

    assert list(out.columns) == ["event_time", "label_source"]
    assert out.shape[0] == 1
    assert out["event_time"].iloc[0] == "2025-01-01"
    assert out["label_source"].isna().all()
