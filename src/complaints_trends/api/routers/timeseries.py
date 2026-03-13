from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..deps import get_service_container
from ..services.timeseries_service import (
    aggregate_heatmap,
    aggregate_timeseries_by_category,
    aggregate_timeseries_overall,
    compute_category_contribution,
    compute_compare_summary,
    filter_by_date,
    resolve_category_scope,
    resolve_compare_window,
)

router = APIRouter(prefix="/api/timeseries", tags=["timeseries"])


def _df(services, viz_tag: str | None):
    loader = services["loader"]
    return loader.load_viz_state(viz_tag) if viz_tag else loader.load_prepare()


@router.get("/overall")
def overall(
    date_from: str | None = None,
    date_to: str | None = None,
    granularity: str = "D",
    baseline_mode: str = "previous_period",
    baseline_date_from: str | None = None,
    baseline_date_to: str | None = None,
    viz_tag: str | None = None,
    services=Depends(get_service_container),
):
    df = _df(services, viz_tag)
    if df.empty:
        return {"actual": [], "expected": [], "delta": [], "cumulative": [], "summary": {}}
    col = "date" if "date" in df.columns else "event_time"
    actual = filter_by_date(df, col, date_from, date_to)
    b_from, b_to = resolve_compare_window(date_from, date_to, baseline_mode, baseline_date_from, baseline_date_to)
    baseline = filter_by_date(df, col, b_from.isoformat() if b_from is not None else None, b_to.isoformat() if b_to is not None else None)
    return aggregate_timeseries_overall(actual, baseline, col, col, granularity)


@router.get("/by-category")
def by_category(
    date_from: str | None = None,
    date_to: str | None = None,
    granularity: str = "D",
    category_mode: str = "top",
    top_n: int = 10,
    categories: list[str] = Query(default_factory=list),
    include_other: bool = True,
    viz_tag: str | None = None,
    services=Depends(get_service_container),
):
    df = _df(services, viz_tag)
    if df.empty:
        return {"rows": [], "resolved_categories": [], "used_other": False}
    col = "date" if "date" in df.columns else "event_time"
    actual = filter_by_date(df, col, date_from, date_to)
    filtered, resolved, used_other = resolve_category_scope(actual, category_mode, top_n, categories, include_other)
    data = aggregate_timeseries_by_category(filtered, col, granularity)
    data["resolved_categories"] = resolved
    data["used_other"] = used_other
    return data


@router.get("/heatmap")
def heatmap(
    date_from: str | None = None,
    date_to: str | None = None,
    viz_tag: str | None = None,
    services=Depends(get_service_container),
):
    df = _df(services, viz_tag)
    col = "date" if "date" in df.columns else "event_time"
    if df.empty or col not in df.columns:
        return {"weekday_hour": [], "calendar": []}
    actual = filter_by_date(df, col, date_from, date_to)
    return aggregate_heatmap(actual, col)


@router.get("/compare")
def compare(
    date_from: str | None = None,
    date_to: str | None = None,
    baseline_mode: str = "previous_period",
    baseline_date_from: str | None = None,
    baseline_date_to: str | None = None,
    category_mode: str = "top",
    top_n: int = 10,
    categories: list[str] = Query(default_factory=list),
    include_other: bool = True,
    viz_tag: str | None = None,
    services=Depends(get_service_container),
):
    df = _df(services, viz_tag)
    if df.empty:
        return {"summary": {"actual_total": 0, "baseline_total": 0, "delta_abs": 0, "delta_pct": None}, "contributions": []}

    col = "date" if "date" in df.columns else "event_time"
    actual = filter_by_date(df, col, date_from, date_to)
    b_from, b_to = resolve_compare_window(date_from, date_to, baseline_mode, baseline_date_from, baseline_date_to)
    baseline = filter_by_date(df, col, b_from.isoformat() if b_from is not None else None, b_to.isoformat() if b_to is not None else None)

    actual_f, _, _ = resolve_category_scope(actual, category_mode, top_n, categories, include_other)
    base_f, _, _ = resolve_category_scope(baseline, category_mode, top_n, categories, include_other)

    actual_f = actual_f.copy()
    baseline_f = base_f.copy()
    actual_f["date"] = actual_f[col]
    baseline_f["date"] = baseline_f[col]
    return {
        "summary": compute_compare_summary(actual_f, baseline_f),
        "contributions": compute_category_contribution(actual_f, baseline_f),
    }
