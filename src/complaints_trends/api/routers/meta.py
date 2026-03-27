from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends

from ..deps import get_service_container
from ..schemas import AvailableTagsResponse, DatasetMeta, MetaConfigResponse, MetaDatasetsResponse

router = APIRouter(prefix="/api/meta", tags=["meta"])


@router.get("/config", response_model=MetaConfigResponse)
def config_meta(services=Depends(get_service_container)):
    cfg = services["cfg"]
    return MetaConfigResponse(
        prepare_output_parquet=cfg.prepare.output_parquet,
        reports_dir=cfg.analysis.reports_dir,
        interim_dir=cfg.analysis.pattern_monitoring.interim_dir,
    )


@router.get("/datasets", response_model=MetaDatasetsResponse)
def datasets_meta(services=Depends(get_service_container)):
    cfg = services["cfg"]
    prepare = Path(cfg.prepare.output_parquet)
    datasets = [DatasetMeta(name="prepare", path=str(prepare), exists=prepare.exists())]
    for p in sorted(Path("data/interim").glob("month_*.parquet")):
        datasets.append(DatasetMeta(name=p.stem, path=str(p), exists=True))

    min_date = max_date = None
    label_sources: set[str] = set()
    if prepare.exists():
        loader = services["loader"]
        df = loader.read_parquet(prepare, columns=["event_time", "label_source"])
        col = "event_time" if "event_time" in df.columns else None
        if col:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                min_date = dt.min().date()
                max_date = dt.max().date()
        if "label_source" in df.columns:
            label_sources = set(df["label_source"].dropna().astype(str).unique().tolist())
    return MetaDatasetsResponse(datasets=datasets, min_date=min_date, max_date=max_date, label_sources=sorted(label_sources))


@router.get("/tags", response_model=AvailableTagsResponse)
def tags_meta(services=Depends(get_service_container)):
    loader = services["loader"]
    return AvailableTagsResponse(
        viz_tags=loader.find_viz_tags(),
        pattern_fit_tags=loader.find_pattern_fit_tags(),
        pattern_monitor_tags=loader.find_pattern_monitor_tags(),
    )


@router.get("/categories")
def categories_meta(date_from: str | None = None, date_to: str | None = None, services=Depends(get_service_container)):
    loader = services["loader"]
    df = loader.load_prepare_timeseries()
    if df.empty:
        return {"categories": []}
    if "event_time" in df.columns:
        dt = pd.to_datetime(df["event_time"], errors="coerce")
        if date_from:
            df = df[dt >= pd.to_datetime(date_from)]
            dt = pd.to_datetime(df.get("event_time"), errors="coerce")
        if date_to:
            df = df[dt <= pd.to_datetime(date_to)]
    if "category" not in df.columns and "complaint_category_llm" in df.columns:
        df["category"] = df["complaint_category_llm"]
    cats = sorted(df.get("category", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    return {"categories": cats}


@router.get("/taxonomy-labels")
def taxonomy_labels_meta(services=Depends(get_service_container)):
    labels = services.get("labels")
    if labels is None:
        return {"category_labels": {}, "subcategory_labels": {}}
    return {
        "category_labels": labels.category_labels,
        "subcategory_labels": labels.subcategory_labels,
    }


@router.get("/prepare-preview")
def prepare_preview_meta(
    page: int = 1,
    page_size: int = 50,
    q: str | None = None,
    services=Depends(get_service_container),
):
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 200))

    loader = services["loader"]
    df = loader.load_prepare()
    if df.empty:
        return {"items": [], "columns": [], "total": 0, "page": page, "page_size": page_size, "path": str(services["cfg"].prepare.output_parquet)}

    if q:
        qv = q.strip().lower()
        if qv:
            text_cols = [c for c in ["row_id", "client_first_message", "dialog_text", "category", "subcategory", "complaint_category_llm", "complaint_subcategory_llm"] if c in df.columns]
            if text_cols:
                mask = pd.Series(False, index=df.index)
                for col in text_cols:
                    mask = mask | df[col].fillna("").astype(str).str.lower().str.contains(qv, regex=False)
                df = df[mask]

    total = int(len(df))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end].copy()

    for col in page_df.columns:
        if pd.api.types.is_datetime64_any_dtype(page_df[col]):
            page_df[col] = page_df[col].astype(str)
    page_df = page_df.where(pd.notna(page_df), None)

    return {
        "items": page_df.to_dict(orient="records"),
        "columns": [str(c) for c in page_df.columns.tolist()],
        "total": total,
        "page": page,
        "page_size": page_size,
        "path": str(services["cfg"].prepare.output_parquet),
    }
