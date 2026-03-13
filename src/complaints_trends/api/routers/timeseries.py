from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from ..deps import get_service_container
from ..schemas import CompareResponse, HeatmapCell, TimeSeriesPoint
from ..services.timeseries_service import compute_category_compare_metrics, filter_by_date

router = APIRouter(prefix="/api/timeseries", tags=["timeseries"])


def _df(services, viz_tag: str | None):
    loader = services["loader"]
    return loader.load_viz_state(viz_tag) if viz_tag else loader.load_prepare()


@router.get("/overall", response_model=list[TimeSeriesPoint])
def overall(date_from: str | None = None, date_to: str | None = None, viz_tag: str | None = None, services=Depends(get_service_container)):
    df = _df(services, viz_tag)
    if df.empty:
        return []

    col = "date" if "date" in df.columns else "event_time"
    df = filter_by_date(df, col, date_from, date_to)
    if df.empty:
        return []

    df["date"] = pd.to_datetime(df[col], errors="coerce").dt.date
    df = df[df["date"].notna()]
    if "metric_count" in df.columns:
        out = df.groupby("date")["metric_count"].sum().reset_index(name="actual")
    elif "count" in df.columns:
        out = df.groupby("date")["count"].sum().reset_index(name="actual")
    else:
        out = df.groupby("date").size().reset_index(name="actual")
    return [TimeSeriesPoint(date=r.date, actual=float(r.actual), expected=None, delta=None) for r in out.itertuples(index=False)]


@router.get("/by-category")
def by_category(date_from: str | None = None, date_to: str | None = None, viz_tag: str | None = None, services=Depends(get_service_container)):
    df = _df(services, viz_tag)
    if df.empty:
        return {"rows": []}

    col = "date" if "date" in df.columns else "event_time"
    df = filter_by_date(df, col, date_from, date_to)
    if "category" not in df.columns:
        df["category"] = "UNKNOWN"

    df["date"] = pd.to_datetime(df[col], errors="coerce").dt.date
    df = df[df["date"].notna()]
    value_col = "metric_count" if "metric_count" in df.columns else ("count" if "count" in df.columns else None)
    if value_col:
        grp = df.groupby(["date", "category"])[value_col].sum().reset_index(name="count")
    else:
        grp = df.groupby(["date", "category"]).size().reset_index(name="count")
    return {"rows": grp.to_dict(orient="records")}


@router.get("/heatmap", response_model=list[HeatmapCell])
def heatmap(viz_tag: str | None = None, services=Depends(get_service_container)):
    df = _df(services, viz_tag)
    col = "date" if "date" in df.columns else "event_time"
    if df.empty or col not in df.columns:
        return []

    dt = pd.to_datetime(df[col], errors="coerce")
    s = pd.DataFrame({"dow": dt.dt.dayofweek, "hour": dt.dt.hour.fillna(0).astype(int)})
    s = s[s["dow"].notna()]
    grp = s.value_counts().reset_index(name="value")
    return [HeatmapCell(dow=int(r.dow), hour=int(r.hour), value=float(r.value)) for r in grp.itertuples(index=False)]


@router.get("/compare", response_model=CompareResponse)
def compare(date_from: str | None = None, date_to: str | None = None, viz_tag: str | None = None, services=Depends(get_service_container)):
    actual = _df(services, viz_tag)
    base = services["loader"].load_prepare()
    if actual.empty:
        return CompareResponse(actual_total=0, baseline_total=0, delta_abs=0, delta_pct=None, contributions=[])

    col = "date" if "date" in actual.columns else "event_time"
    actual = filter_by_date(actual, col, date_from, date_to)
    base = filter_by_date(base, "event_time" if "event_time" in base.columns else "date", date_from, date_to)

    a = float(actual["metric_count"].sum()) if "metric_count" in actual.columns else float(len(actual))
    b = float(base["count"].sum()) if "count" in base.columns else float(len(base))

    contributions_df = compute_category_compare_metrics(actual, base).head(15)
    contributions = [
        {
            "category": row["category"],
            "count": float(row["count"]),
            "share": float(row["share"]),
            "baseline_count": float(row["baseline_count"]),
            "delta_abs": float(row["delta_abs"]),
            "delta_pct": (None if pd.isna(row["delta_pct"]) else float(row["delta_pct"])),
            "anomaly_score": float(row["anomaly_score"]),
            "pattern_score": float(row["pattern_score"]),
        }
        for _, row in contributions_df.iterrows()
    ]

    return CompareResponse(
        actual_total=a,
        baseline_total=b,
        delta_abs=a - b,
        delta_pct=((a - b) / b if b else None),
        contributions=contributions,
    )
