from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..deps import get_service_container
from ..schemas import OverviewResponse

router = APIRouter(prefix="/api/overview", tags=["overview"])


@router.get("", response_model=OverviewResponse)
def get_overview(
    date_from: str | None = None,
    date_to: str | None = None,
    viz_tag: str | None = None,
    pattern_tag: str | None = None,
    label_source: str | None = None,
    category: list[str] = Query(default_factory=list),
    subcategory: list[str] = Query(default_factory=list),
    baseline_mode: str = "previous_period",
    baseline_date_from: str | None = None,
    baseline_date_to: str | None = None,
    metric: str = "count",
    services=Depends(get_service_container),
) -> OverviewResponse:
    return services["overview"].get_overview(locals())
