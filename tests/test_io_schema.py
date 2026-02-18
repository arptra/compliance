import pandas as pd
import pytest

from app.io import SchemaError, split_baseline_december


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
