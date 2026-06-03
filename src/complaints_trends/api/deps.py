from __future__ import annotations

from pathlib import Path

from fastapi import Request

from ..config import load_config
from .services.gigachat_connection_service import GigaChatConnectionService
from .services.gigachat_lab_service import GigaChatLabService
from .services.catalog_service import CatalogService
from .services.parquet_lake_service import ParquetLakeService


def get_services(config_path: str) -> dict:
    cfg = load_config(config_path)
    catalog = CatalogService(Path("data/app.sqlite"))
    record_lake = ParquetLakeService(Path("data/lake"), catalog)
    gigachat_lab = GigaChatLabService(cfg, catalog_service=catalog, lake_service=record_lake)
    gigachat_connection = GigaChatConnectionService(cfg, lab_service=gigachat_lab)
    return {
        "cfg": cfg,
        "catalog": catalog,
        "record_lake": record_lake,
        "gigachat_lab": gigachat_lab,
        "gigachat_connection": gigachat_connection,
    }


def get_service_container(request: Request):
    return get_services(request.app.state.config_path)
