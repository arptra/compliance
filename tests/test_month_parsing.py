from pathlib import Path

import pandas as pd

from complaints_trends.config import InputConfig
from complaints_trends.io_excel import load_excel_with_month, parse_event_time


def test_parse_event_time_default_and_format():
    s = pd.Series(["2025-01-09 12:55:29", "2025-02-01 00:00:00"])
    out = parse_event_time(s)
    assert out.dt.strftime("%Y-%m").tolist() == ["2025-01", "2025-02"]

    out2 = parse_event_time(s, "%Y-%m-%d %H:%M:%S")
    assert out2.dt.strftime("%Y-%m").tolist() == ["2025-01", "2025-02"]


def test_load_excel_with_month_uses_datetime_column(tmp_path):
    p = tmp_path / "sample.xlsx"
    pd.DataFrame({"created_at": ["2025-03-09 12:55:29"], "dialog_text": ["x"]}).to_excel(p, index=False)

    cfg = InputConfig.model_validate(
        {
            "input_dir": str(tmp_path),
            "file_glob": "*.xlsx",
            "datetime_column": "created_at",
            "datetime_format": "%Y-%m-%d %H:%M:%S",
            "id_column": None,
            "signal_columns": ["dialog_text"],
            "dialog_column": "dialog_text",
            "encoding": "utf-8",
        }
    )
    df = load_excel_with_month(p, cfg)
    assert df.loc[0, "month"] == "2025-03"


def test_load_excel_with_month_raises_clear_error_when_missing_datetime_column(tmp_path):
    p = tmp_path / "sample.xlsx"
    pd.DataFrame({"other_col": ["2025-03-09 12:55:29"], "dialog_text": ["x"]}).to_excel(p, index=False)

    cfg = InputConfig.model_validate(
        {
            "input_dir": str(tmp_path),
            "file_glob": "*.xlsx",
            "datetime_column": "created_at",
            "datetime_format": "%Y-%m-%d %H:%M:%S",
            "id_column": None,
            "signal_columns": ["dialog_text"],
            "dialog_column": "dialog_text",
            "encoding": "utf-8",
        }
    )

    try:
        load_excel_with_month(p, cfg)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "No datetime column" in str(e)
