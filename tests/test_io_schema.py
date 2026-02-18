from pathlib import Path

import pandas as pd
import pytest

from app.io import SchemaError, load_and_split, merge_message_columns, split_baseline_december


def test_missing_column_raises():
    cfg = {
        "input": {
            "message_col": "message",
            "date_col": "date",
            "month_col": None,
            "date_format": None,
            "baseline": {"mode": "last_n_months", "n_months": 1},
            "december": {"mode": "month_value", "month_value": "2025-12"},
        }
    }
    with pytest.raises(SchemaError):
        split_baseline_december(pd.DataFrame({"text": ["a"]}), cfg)


def test_month_split_works():
    df = pd.DataFrame(
        {
            "message": ["a", "b", "c"],
            "month": ["2025-11", "2025-12", "2025-12"],
        }
    )
    cfg = {
        "input": {
            "message_col": "message",
            "date_col": None,
            "month_col": "month",
            "date_format": None,
            "baseline": {"mode": "last_n_months", "n_months": 1},
            "december": {"mode": "month_value", "month_value": "2025-12"},
        }
    }
    out = split_baseline_december(df, cfg)
    assert len(out.baseline) == 2
    assert len(out.december) == 2


def test_merge_multiple_message_columns():
    df = pd.DataFrame({"m1": ["привет", None], "m2": ["мир", "канал"]})
    merged = merge_message_columns(df, ["m1", "m2"]).tolist()
    assert merged[0] == "привет мир"
    assert merged[1] == "канал"


def test_multifile_mode_baseline_and_december(tmp_path: Path):
    baseline_file = tmp_path / "baseline_2025_11.xlsx"
    december_file = tmp_path / "december_2025_12.xlsx"

    pd.DataFrame({"message": ["обычное сообщение"]}).to_excel(baseline_file, index=False)
    pd.DataFrame({"message": ["жалоба за декабрь"]}).to_excel(december_file, index=False)

    cfg = {
        "input": {
            "path": None,
            "baseline_paths": [str(baseline_file)],
            "december_path": str(december_file),
            "december_paths": [],
            "message_col": "message",
            "message_cols": None,
            "date_col": None,
            "month_col": None,
            "date_format": None,
            "baseline": {"mode": "last_n_months", "n_months": 1},
            "december": {"mode": "month_value", "month_value": "2025-12"},
        }
    }

    splits = load_and_split(cfg)
    assert len(splits.baseline) == 1
    assert len(splits.december) == 1
    assert "message_joined" in splits.baseline.columns
