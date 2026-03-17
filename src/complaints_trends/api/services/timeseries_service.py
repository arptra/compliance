from __future__ import annotations

from datetime import timedelta
from typing import Literal

import pandas as pd

BaselineMode = Literal["previous_period", "same_weekday", "seasonal", "custom_range"]
CategoryMode = Literal["top", "custom", "all"]


def _to_dt(v: str | None) -> pd.Timestamp | None:
    if not v:
        return None
    ts = pd.to_datetime(v, errors="coerce")
    return None if pd.isna(ts) else ts


def _value_col(df: pd.DataFrame) -> str | None:
    for c in ("metric_count", "count"):
        if c in df.columns:
            return c
    return None


def _ensure_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "category" not in out.columns:
        out["category"] = "UNKNOWN"
    out["category"] = out["category"].fillna("UNKNOWN").astype(str)
    return out


def filter_by_date(df: pd.DataFrame, date_col: str, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    dt = pd.to_datetime(df[date_col], errors="coerce")
    mask = dt.notna()
    start = _to_dt(date_from)
    end = _to_dt(date_to)
    if start is not None:
        mask &= dt >= start
    if end is not None:
        mask &= dt <= end
    out = df.loc[mask].copy(deep=False)
    out[date_col] = dt.loc[mask]
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
    if start is None or end is None:
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


def _freq_rule(granularity: str) -> str:
    return {"D": "D", "W": "W-MON", "M": "MS"}.get(granularity, "D")


def _prepare(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = _ensure_category(df)
    out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out["date"].notna()].copy()
    vc = _value_col(out)
    out["value"] = out[vc].astype(float) if vc else 1.0
    return out


def resolve_category_scope(
    df: pd.DataFrame,
    category_mode: CategoryMode,
    top_n: int,
    selected_categories: list[str] | None,
    include_other: bool,
    category_column: str = "category",
) -> tuple[pd.DataFrame, list[str], bool]:
    out = _ensure_category(df)
    selected = [c for c in (selected_categories or []) if c]
    if category_mode == "all":
        resolved = sorted(out[category_column].unique().tolist())
        return out, resolved, False
    if category_mode == "custom" and selected:
        f = out[out[category_column].isin(selected)].copy()
        return f, selected, False

    vc = _value_col(out)
    by_cat = out.groupby(category_column)[vc].sum() if vc else out.groupby(category_column).size()
    top = by_cat.sort_values(ascending=False).head(max(int(top_n or 10), 1)).index.tolist()
    f = out[out[category_column].isin(top)].copy()
    used_other = False
    if include_other:
        tail = out[~out[category_column].isin(top)].copy()
        if not tail.empty:
            tail[category_column] = "OTHER"
            f = pd.concat([f, tail], ignore_index=True)
            used_other = True
    return f, top, used_other


def aggregate_timeseries_overall(
    actual_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    date_col: str,
    baseline_col: str,
    granularity: str,
) -> dict:
    if actual_df.empty:
        return {"actual": [], "expected": [], "delta": [], "cumulative": [], "summary": {}}

    a = _prepare(actual_df, date_col)
    a["bucket"] = a["date"].dt.to_period(_freq_rule(granularity)).dt.to_timestamp()
    actual = a.groupby("bucket")["value"].sum().reset_index(name="actual")

    expected = pd.DataFrame(columns=["bucket", "expected"])
    if not baseline_df.empty and baseline_col in baseline_df.columns:
        b = _prepare(baseline_df, baseline_col)
        b["bucket"] = b["date"].dt.to_period(_freq_rule(granularity)).dt.to_timestamp()
        expected = b.groupby("bucket")["value"].sum().reset_index(name="expected").sort_values("bucket")

    merged = actual.merge(expected, on="bucket", how="left")

    # Baseline window usually has different dates than actual window; if exact date join gives no overlap,
    # align expected by bucket order (relative position in period) instead of calendar date.
    if not expected.empty and merged["expected"].isna().all():
        exp_vals = expected["expected"].tolist()
        if len(exp_vals) == len(merged):
            merged["expected"] = exp_vals
        elif len(exp_vals) > 0:
            # fallback: repeat last known expected or trim to the actual horizon
            padded = (exp_vals + [exp_vals[-1]] * len(merged))[: len(merged)]
            merged["expected"] = padded

    merged["expected"] = merged["expected"].fillna(0.0)
    merged["delta"] = merged["actual"] - merged["expected"]
    merged["delta_pct"] = merged.apply(lambda r: (r["delta"] / r["expected"]) if r["expected"] else None, axis=1)
    merged["actual_cumulative"] = merged["actual"].cumsum()
    merged["expected_cumulative"] = merged["expected"].cumsum()

    return {
        "actual": [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(merged["bucket"], merged["actual"])],
        "expected": [{"date": d.date().isoformat(), "value": float(v)} for d, v in zip(merged["bucket"], merged["expected"])],
        "delta": [
            {
                "date": d.date().isoformat(),
                "actual": float(a),
                "expected": float(e),
                "delta_abs": float(x),
                "delta_pct": (None if pd.isna(p) else float(p)),
            }
            for d, a, e, x, p in zip(merged["bucket"], merged["actual"], merged["expected"], merged["delta"], merged["delta_pct"])
        ],
        "cumulative": [
            {"date": d.date().isoformat(), "actual": float(a), "expected": float(e)}
            for d, a, e in zip(merged["bucket"], merged["actual_cumulative"], merged["expected_cumulative"])
        ],
        "summary": {
            "actual_total": float(merged["actual"].sum()),
            "expected_total": float(merged["expected"].sum()),
            "delta_abs": float(merged["delta"].sum()),
            "delta_pct": (float(merged["delta"].sum() / merged["expected"].sum()) if float(merged["expected"].sum()) else None),
        },
    }


def aggregate_timeseries_by_category(df: pd.DataFrame, date_col: str, granularity: str) -> dict:
    if df.empty:
        return {"rows": []}
    out = _prepare(df, date_col)
    out["bucket"] = out["date"].dt.to_period(_freq_rule(granularity)).dt.to_timestamp()
    grp = out.groupby(["bucket", "category"])["value"].sum().reset_index(name="count")
    totals = grp.groupby("bucket")["count"].sum().rename("total")
    grp = grp.merge(totals, on="bucket", how="left")
    grp["share"] = (grp["count"] / grp["total"].where(grp["total"] != 0, 1.0)).fillna(0.0)
    return {
        "rows": [
            {"date": r.bucket.date().isoformat(), "category": str(r.category), "count": float(r.count), "share": float(r.share)}
            for r in grp.itertuples(index=False)
        ]
    }


def aggregate_heatmap(df: pd.DataFrame, date_col: str) -> dict:
    if df.empty or date_col not in df.columns:
        return {"weekday_hour": [], "calendar": []}
    dt = pd.to_datetime(df[date_col], errors="coerce")
    tmp = pd.DataFrame({"date": dt}).dropna()
    if tmp.empty:
        return {"weekday_hour": [], "calendar": []}
    tmp["dow"] = tmp["date"].dt.dayofweek
    tmp["hour"] = tmp["date"].dt.hour
    wh = tmp.groupby(["dow", "hour"]).size().reset_index(name="value")
    cal = tmp.groupby(tmp["date"].dt.date).size().reset_index(name="value")
    return {
        "weekday_hour": [{"dow": int(r.dow), "hour": int(r.hour), "value": float(r.value)} for r in wh.itertuples(index=False)],
        "calendar": [{"date": str(r.date), "value": float(r.value)} for r in cal.itertuples(index=False)],
    }


def compute_compare_summary(actual_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict:
    a = _prepare(actual_df, "date") if (not actual_df.empty and "date" in actual_df.columns) else pd.DataFrame()
    b = _prepare(baseline_df, "date") if (not baseline_df.empty and "date" in baseline_df.columns) else pd.DataFrame()
    actual_total = float(a["value"].sum()) if not a.empty else 0.0
    expected_total = float(b["value"].sum()) if not b.empty else 0.0
    delta = actual_total - expected_total
    return {
        "actual_total": actual_total,
        "baseline_total": expected_total,
        "delta_abs": delta,
        "delta_pct": (delta / expected_total if expected_total else None),
    }


def compute_category_contribution(actual_df: pd.DataFrame, baseline_df: pd.DataFrame) -> list[dict]:
    a = _ensure_category(actual_df)
    b = _ensure_category(baseline_df)
    a["value"] = a[_value_col(a)].astype(float) if _value_col(a) else 1.0
    b["value"] = b[_value_col(b)].astype(float) if _value_col(b) else 1.0
    ag = a.groupby("category")["value"].sum().rename("actual_count")
    bg = b.groupby("category")["value"].sum().rename("expected_count")
    out = pd.concat([ag, bg], axis=1).fillna(0.0).reset_index()
    out["delta_abs"] = out["actual_count"] - out["expected_count"]
    out["delta_pct"] = out["delta_abs"] / out["expected_count"].replace({0.0: pd.NA})
    total_actual = max(float(out["actual_count"].sum()), 1.0)
    total_delta = float(out["delta_abs"].sum())
    out["share"] = out["actual_count"] / total_actual
    out["contribution_to_growth"] = out["delta_abs"].apply(lambda x: (x / total_delta) if total_delta else 0.0)
    out["anomaly_score"] = out["delta_pct"].fillna(0.0).clip(lower=0.0)
    out = out.sort_values("delta_abs", ascending=False)
    return [
        {
            "category": str(r.category),
            "actual_count": float(r.actual_count),
            "expected_count": float(r.expected_count),
            "delta_abs": float(r.delta_abs),
            "delta_pct": (None if pd.isna(r.delta_pct) else float(r.delta_pct)),
            "share": float(r.share),
            "contribution_to_growth": float(r.contribution_to_growth),
            "anomaly_score": float(r.anomaly_score),
        }
        for r in out.itertuples(index=False)
    ]


def compute_category_compare_metrics(actual_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    rows = compute_category_contribution(actual_df, baseline_df)
    if not rows:
        return pd.DataFrame(columns=["category","count","share","baseline_count","delta_abs","delta_pct","anomaly_score","pattern_score"])
    out = pd.DataFrame(rows)
    out = out.rename(columns={"actual_count":"count","expected_count":"baseline_count"})
    out["pattern_score"] = 0.0
    return out[["category","count","share","baseline_count","delta_abs","delta_pct","anomaly_score","pattern_score"]]


def compute_expected_series(history: pd.DataFrame, actual_series: pd.DataFrame, baseline_start: pd.Timestamp | None, baseline_end: pd.Timestamp | None, mode: BaselineMode):
    if history.empty or baseline_start is None or baseline_end is None:
        return pd.Series([None] * len(actual_series), index=actual_series.index)
    baseline = history[(history["date"] >= baseline_start) & (history["date"] <= baseline_end)].copy()
    if baseline.empty:
        return pd.Series([None] * len(actual_series), index=actual_series.index)
    if mode == "same_weekday":
        med = baseline.groupby(pd.to_datetime(baseline["date"]).dt.dayofweek)["actual"].median()
        return pd.to_datetime(actual_series["date"]).dt.dayofweek.map(med)
    expected = float(baseline["actual"].mean())
    return pd.Series([expected] * len(actual_series), index=actual_series.index)
