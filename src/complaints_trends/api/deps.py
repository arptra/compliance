from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from ..config import load_config
from .services.categories_service import CategoriesService
from .services.data_loader import DataLoader
from .services.overview_service import OverviewService
from .services.pattern_fit_service import PatternFitService
from .services.pattern_monitor_service import PatternMonitorService
from .services.report_service import ReportService
from .services.preparation_service import PreparationService
from .services.run_service import RunService


@lru_cache(maxsize=2)
def get_loader(config_path: str) -> DataLoader:
    cfg = load_config(config_path)
    return DataLoader(cfg)


def get_services(config_path: str) -> dict:
    cfg = load_config(config_path)
    loader = get_loader(config_path)
    overview = OverviewService(loader)
    monitor = PatternMonitorService(loader)
    return {
        "cfg": cfg,
        "loader": loader,
        "overview": overview,
        "categories": CategoriesService(loader),
        "pattern_fit": PatternFitService(loader),
        "pattern_monitor": monitor,
        "report": ReportService(overview, monitor, loader),
        "preparation": PreparationService(cfg),
        "run": RunService(cfg),
    }


def get_service_container(request: Request):
    return get_services(request.app.state.config_path)
