from __future__ import annotations

from datetime import date, datetime

import orjson
import pandas as pd

from complaints_trends.api.services.parquet_lake_service import ParquetLakeService


def test_frame_to_rows_returns_json_serializable_values() -> None:
    df = pd.DataFrame(
        {
            "nan_value": [float("nan")],
            "inf_value": [float("inf")],
            "nat_value": [pd.NaT],
            "timestamp_value": [pd.Timestamp("2026-06-05 13:45:00")],
            "date_value": [date(2026, 6, 5)],
            "bytes_value": [b"text"],
            "nested_value": [{"items": [pd.NA, float("-inf"), datetime(2026, 6, 5, 13, 45)]}],
        }
    )

    rows = ParquetLakeService._frame_to_rows(df)

    assert rows == [
        {
            "nan_value": None,
            "inf_value": None,
            "nat_value": None,
            "timestamp_value": "2026-06-05T13:45:00",
            "date_value": "2026-06-05",
            "bytes_value": "text",
            "nested_value": {"items": [None, None, "2026-06-05T13:45:00"]},
        }
    ]
    orjson.dumps(rows)


def test_exists_any_filter_matches_any_non_empty_field() -> None:
    df = pd.DataFrame(
        {
            "first_tag": ["alpha", "", None, float("inf")],
            "second_tag": [None, "beta", "", None],
            "other": ["x", "y", "z", "w"],
        }
    )

    fields = ParquetLakeService._filter_field_names({"column": "ignored", "op": "exists_any", "value": ["first_tag", "second_tag"]})
    mask = ParquetLakeService._field_exists_mask(df, fields)

    assert fields == ["first_tag", "second_tag"]
    assert mask.tolist() == [True, True, False, False]


def test_exists_any_filter_parses_comma_separated_columns() -> None:
    fields = ParquetLakeService._filter_field_names({"column": "first_tag, second_tag", "op": "exists_any"})

    assert fields == ["first_tag", "second_tag"]
