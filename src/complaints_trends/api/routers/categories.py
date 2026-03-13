from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_service_container
from ..schemas import CategoryDetailResponse, CategoryTableResponse, ExamplesResponse, SubcategoryResponse

router = APIRouter(prefix="/api/categories", tags=["categories"])


@router.get("", response_model=CategoryTableResponse)
def categories(
    date_from: str | None = None,
    date_to: str | None = None,
    viz_tag: str | None = None,
    baseline_mode: str = "previous_period",
    baseline_date_from: str | None = None,
    baseline_date_to: str | None = None,
    services=Depends(get_service_container),
):
    return services["categories"].table(locals())


@router.get("/{category}/timeseries", response_model=CategoryDetailResponse)
def category_timeseries(category: str, date_from: str | None = None, date_to: str | None = None, viz_tag: str | None = None, services=Depends(get_service_container)):
    return services["categories"].category_timeseries(category, locals())


@router.get("/{category}/subcategories", response_model=SubcategoryResponse)
def category_subcategories(category: str, date_from: str | None = None, date_to: str | None = None, viz_tag: str | None = None, services=Depends(get_service_container)):
    return services["categories"].subcategories(category, locals())


@router.get("/{category}/examples", response_model=ExamplesResponse)
def category_examples(category: str, limit: int = 20, services=Depends(get_service_container)):
    return services["categories"].examples(category, limit)
