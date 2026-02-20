from pathlib import Path

import pandas as pd
import pytest

from app.io import SchemaError, load_and_split, split_baseline_december


def test_missing_column_raises():
    cfg = {
        "input": {
            "message_col": "message",
            "message_cols": None,
            "date_col": "date",
            "month_col": None,
            "date_format": None,
            "role_parsing": {"enabled": False},
            "baseline": {"mode": "last_n_months", "n_months": 1},
            "december": {"mode": "month_value", "month_value": "2025-12"},
        },
        "output": {"raw_text_col_name": "message_raw", "analyzed_text_col_name": "message_client_first"},
    }
    with pytest.raises(SchemaError):
        split_baseline_december(pd.DataFrame({"text": ["a"]}), cfg)


def test_month_split_works():
    df = pd.DataFrame({"message": ["a", "b", "c"], "month": ["2025-11", "2025-12", "2025-12"]})
    cfg = {
        "input": {
            "message_col": "message",
            "message_cols": None,
            "date_col": None,
            "month_col": "month",
            "date_format": None,
            "role_parsing": {"enabled": False},
            "baseline": {"mode": "last_n_months", "n_months": 1},
            "december": {"mode": "month_value", "month_value": "2025-12"},
        },
        "output": {"raw_text_col_name": "message_raw", "analyzed_text_col_name": "message_client_first"},
    }
    out = split_baseline_december(df, cfg)
    assert len(out.baseline) == 2
    assert "message_raw" in out.baseline.columns


def test_multifile_mode_baseline_and_december(tmp_path: Path):
    baseline_file = tmp_path / "baseline.xlsx"
    december_file = tmp_path / "december.xlsx"

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
            "role_parsing": {"enabled": False},
            "baseline": {"mode": "last_n_months", "n_months": 1},
            "december": {"mode": "month_value", "month_value": "2025-12"},
        },
        "output": {"raw_text_col_name": "message_raw", "analyzed_text_col_name": "message_client_first"},
    }

    splits = load_and_split(cfg)
    assert len(splits.baseline) == 1
    assert len(splits.december) == 1
