from pathlib import Path

import pandas as pd

from complaints_trends.io_excel import extract_month_from_column, extract_month_from_filename


def test_extract_month_from_filename_multiple_patterns():
    p1 = Path("2025-01_report.xlsx")
    p2 = Path("calls_2025_02_part.xlsx")
    patterns = [r"(\\d{4})-(\\d{2})", r"(\\d{4})_(\\d{2})"]
    assert extract_month_from_filename(p1, patterns=patterns) == "2025-01"
    assert extract_month_from_filename(p2, patterns=patterns) == "2025-02"


def test_extract_month_from_column_datetime_string():
    s = pd.Series(["2025-01-09 12:55:29", "2025-02-01 00:00:00"])
    out = extract_month_from_column(s)
    assert out.tolist() == ["2025-01", "2025-02"]


def test_extract_month_from_column_with_explicit_format():
    s = pd.Series(["2025-03-09 12:55:29"])
    out = extract_month_from_column(s, "%Y-%m-%d %H:%M:%S")
    assert out.iloc[0] == "2025-03"
