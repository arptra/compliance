from __future__ import annotations

from datetime import timedelta
from typing import Literal

import pandas as pd


BaselineMode = Literal["previous_period", "same_weekday", "seasonal", "custom_range"]


def _to_dt(v: str | None) -> pd.Timestamp | None:
    if not v:
        return None
    return pd.to_datetime(v, errors="coerce")


def _ensure_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "category" not in out.columns:
        out["category"] = "UNKNOWN"
    out["category"] = out["category"].fillna("UNKNOWN").astype(str)
    return out


def filter_by_date(df: pd.DataFrame, date_col: str, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()]
    if date_from:
        out = out[out[date_col] >= _to_dt(date_from)]
    if date_to:
        out = out[out[date_col] <= _to_dt(date_to)]
    return out


def resolve_compare_window(
    date_from: str | None,
    date_to: str | None,
    baseline_mode: BaselineMode,
    baseline_date_from: str | None = None,
    baseline_date_to: str | None = None,
    seasonal_weeks: int = 8,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start, end = _to_dt(date_from), _to_dt(date_to)
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return None, None
    days = (end - start).days + 1
    if baseline_mode == "custom_range":
        return _to_dt(baseline_date_from), _to_dt(baseline_date_to)
    if baseline_mode == "previous_period":
        return start - timedelta(days=days), start - timedelta(days=1)
    if baseline_mode == "seasonal":
        shift = timedelta(days=7 * seasonal_weeks)
        return start - shift, end - shift
    if baseline_mode == "same_weekday":
        return start - timedelta(days=7 * seasonal_weeks), end - timedelta(days=7)
    return None, None


def compute_expected_series(
    history: pd.DataFrame,
    actual_series: pd.DataFrame,
    baseline_start: pd.Timestamp | None,
    baseline_end: pd.Timestamp | None,
    mode: BaselineMode,
) -> pd.Series:
    if history.empty or baseline_start is None or baseline_end is None:
        return pd.Series([None] * len(actual_series), index=actual_series.index)
    baseline = history[(history["date"] >= baseline_start) & (history["date"] <= baseline_end)].copy()
    if baseline.empty:
        return pd.Series([None] * len(actual_series), index=actual_series.index)
    baseline["date"] = pd.to_datetime(baseline["date"], errors="coerce")
    baseline = baseline[baseline["date"].notna()]
    if baseline.empty:
        return pd.Series([None] * len(actual_series), index=actual_series.index)

    if mode == "same_weekday":
        medians = baseline.groupby(baseline["date"].dt.dayofweek)["actual"].median()
        return actual_series["date"].dt.dayofweek.map(medians)

    expected = float(baseline["actual"].mean())
    return pd.Series([expected] * len(actual_series), index=actual_series.index)


def compute_category_compare_metrics(actual_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    actual_df = _ensure_category(actual_df)
    baseline_df = _ensure_category(baseline_df)

    a = actual_df.groupby("category").size().rename("count")
    b = baseline_df.groupby("category").size().rename("baseline_count")
    out = pd.concat([a, b], axis=1).fillna(0.0).reset_index()

    total = max(float(out["count"].sum()), 1.0)
    out["share"] = out["count"] / total
    out["delta_abs"] = out["count"] - out["baseline_count"]
    out["delta_pct"] = out.apply(
        lambda x: (x["delta_abs"] / x["baseline_count"]) if float(x["baseline_count"]) > 0 else None,
        axis=1,
    )
    out["anomaly_score"] = out["delta_pct"].fillna(0.0).clip(lower=0.0)
    out["pattern_score"] = 0.0
    return out.sort_values("delta_abs", ascending=False).reset_index(drop=True)
