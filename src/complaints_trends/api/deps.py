from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Request

from ..config import load_config
from .services.categories_service import CategoriesService
from .services.data_loader import DataLoader
from .services.overview_service import OverviewService
from .services.pattern_fit_service import PatternFitService
from .services.pattern_monitor_service import PatternMonitorService
from .services.feedback_db import FeedbackDB
from .services.feedback_service import FeedbackService
from .services.feedback_dataset_service import FeedbackDatasetService
from .services.feature_build_service import FeatureBuildService
from .services.model_registry_service import ModelRegistryService
from .services.calibrator_service import CalibratorService
from .services.quality_service import QualityService
from .services.audit_service import AuditService
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
    feedback_db = FeedbackDB(Path(cfg.analysis.pattern_monitoring.interim_dir) / "feedback.db")
    feedback = FeedbackService(feedback_db)
    feedback_dataset = FeedbackDatasetService(feedback_db)
    model_registry = ModelRegistryService(feedback_db)
    feature_builder = FeatureBuildService()
    calibrator = CalibratorService(Path(cfg.training.model_dir) / "rerankers", feedback, model_registry, feature_builder)
    monitor = PatternMonitorService(loader, feedback_service=feedback, calibrator_service=calibrator, registry_service=model_registry)
    quality = QualityService(feedback, model_registry)
    audit = AuditService(feedback_db, loader, feedback)
    return {
        "cfg": cfg,
        "loader": loader,
        "overview": overview,
        "categories": CategoriesService(loader),
        "pattern_fit": PatternFitService(loader),
        "pattern_monitor": monitor,
        "feedback": feedback,
        "feedback_dataset": feedback_dataset,
        "quality": quality,
        "audit": audit,
        "model_registry": model_registry,
        "calibrator": calibrator,
        "report": ReportService(overview, monitor, loader),
        "preparation": PreparationService(cfg),
        "run": RunService(cfg),
    }


def get_service_container(request: Request):
    return get_services(request.app.state.config_path)
