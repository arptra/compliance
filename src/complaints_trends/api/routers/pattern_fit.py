from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_service_container
from ..schemas import ClusterProfileResponse, PatternFitSummaryResponse

router = APIRouter(prefix="/api/pattern-fit", tags=["pattern-fit"])


@router.get("/summary", response_model=PatternFitSummaryResponse)
def summary(tag: str = "latest", services=Depends(get_service_container)):
    return services["pattern_fit"].summary(tag)


@router.get("/categories")
def categories(tag: str = "latest", services=Depends(get_service_container)):
    return {"categories": services["pattern_fit"].categories(tag)}


@router.get("/category/{category}")
def category(category: str, tag: str = "latest", services=Depends(get_service_container)):
    return services["pattern_fit"].category(tag, category)


@router.get("/category/{category}/clusters", response_model=ClusterProfileResponse)
def clusters(category: str, tag: str = "latest", services=Depends(get_service_container)):
    return services["pattern_fit"].clusters(tag, category)


@router.get("/category/{category}/seeds")
def seeds(category: str, tag: str = "latest", services=Depends(get_service_container)):
    return services["pattern_fit"].seeds(tag, category)
