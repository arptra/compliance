from __future__ import annotations

from typing import Any

from .feedback_service import FeedbackService
from .model_quality_service import compute_mode_metrics
from .model_registry_service import ModelRegistryService
from .taxonomy_label_service import TaxonomyLabelService


class QualityService:
    def __init__(self, feedback: FeedbackService, registry: ModelRegistryService, labels: TaxonomyLabelService | None = None) -> None:
        self.feedback = feedback
        self.registry = registry
        self.labels = labels

    def compute_model_quality(self, params: dict[str, Any]) -> dict[str, Any]:
        feedback_rows = self.feedback.list_feedback({**params, "limit": 100000})
        include_uncertain_as = str(params.get("include_uncertain_as") or "ignore")

        base = compute_mode_metrics(feedback_rows, mode="base", include_uncertain_as=include_uncertain_as)
        reranked = compute_mode_metrics(feedback_rows, mode="reranked", include_uncertain_as=include_uncertain_as)
        calibrated = compute_mode_metrics(feedback_rows, mode="calibrated", include_uncertain_as=include_uncertain_as)

        if self.labels:
            for mode in (base, calibrated, reranked):
                for item in mode.get("by_category", []):
                    item["label_ru"] = self.labels.category_label_ru(item.get("name"))

        base_p50 = next((x["precision"] for x in base["precision_at"] if x["k"] == 50), None)
        rerank_p50 = next((x["precision"] for x in reranked["precision_at"] if x["k"] == 50), None)
        lift = None
        if base_p50 not in (None, 0) and rerank_p50 is not None:
            lift = (rerank_p50 - base_p50) / base_p50

        versions = []
        for v in self.registry.list_versions():
            metrics = v.get("metrics_json") or {}
            versions.append(
                {
                    "version_id": v["version_id"],
                    "train_rows": v.get("train_rows"),
                    "precision_reviewed": metrics.get("precision_reviewed"),
                    "precision_at_50": metrics.get("precision_at_50"),
                    "precision_at_100": metrics.get("precision_at_100"),
                    "active": bool(v.get("active")),
                }
            )

        return {
            "reviewed_rows": base.get("reviewed_rows") or 0,
            "compare_modes": [base, calibrated, reranked],
            "lift_vs_base": lift,
            "versions": versions,
        }
