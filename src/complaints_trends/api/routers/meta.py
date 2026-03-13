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
        df = pd.read_parquet(prepare)
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
