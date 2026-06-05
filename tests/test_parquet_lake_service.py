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
