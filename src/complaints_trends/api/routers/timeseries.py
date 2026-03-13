from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from ..deps import get_service_container
from ..schemas import CompareResponse, HeatmapCell, TimeSeriesPoint
from ..services.timeseries_service import filter_by_date

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
    d = pd.to_datetime(df[col], errors="coerce").dt.date
    out = d.value_counts().sort_index().reset_index()
    out.columns = ["date", "actual"]
    return [TimeSeriesPoint(date=r.date, actual=float(r.actual), expected=None, delta=None) for r in out.itertuples(index=False)]


@router.get("/by-category")
def by_category(date_from: str | None = None, date_to: str | None = None, viz_tag: str | None = None, services=Depends(get_service_container)):
    df = _df(services, viz_tag)
    if df.empty or "category" not in df.columns:
        return {"rows": []}
    col = "date" if "date" in df.columns else "event_time"
    df = filter_by_date(df, col, date_from, date_to)
    grp = df.groupby([pd.to_datetime(df[col], errors="coerce").dt.date.rename("date"), "category"]).size().reset_index(name="count")
    return {"rows": grp.to_dict(orient="records")}


@router.get("/heatmap", response_model=list[HeatmapCell])
def heatmap(viz_tag: str | None = None, services=Depends(get_service_container)):
    df = _df(services, viz_tag)
    col = "date" if "date" in df.columns else "event_time"
    if df.empty or col not in df.columns:
        return []
    dt = pd.to_datetime(df[col], errors="coerce")
    grp = pd.DataFrame({"dow": dt.dt.dayofweek, "hour": dt.dt.hour.fillna(0).astype(int)}).value_counts().reset_index(name="value")
    return [HeatmapCell(dow=int(r.dow), hour=int(r.hour), value=float(r.value)) for r in grp.itertuples(index=False)]


@router.get("/compare", response_model=CompareResponse)
def compare(date_from: str | None = None, date_to: str | None = None, viz_tag: str | None = None, services=Depends(get_service_container)):
    actual = _df(services, viz_tag)
    base = services["loader"].load_prepare()
    if actual.empty:
        return CompareResponse(actual_total=0, baseline_total=0, delta_abs=0, delta_pct=None, contributions=[])
    col = "date" if "date" in actual.columns else "event_time"
    actual = filter_by_date(actual, col, date_from, date_to)
    a, b = float(len(actual)), float(len(base))
    return CompareResponse(actual_total=a, baseline_total=b, delta_abs=a - b, delta_pct=((a - b) / b if b else None), contributions=[])
