from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .feature_build_service import FeatureBuildService
from .feedback_service import FeedbackService
from .model_registry_service import ModelRegistryService


ScoringMode = Literal["base", "calibrated", "reranked"]


class ScoringStrategy:
    mode: ScoringMode = "base"

    def score_rows(self, rows: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
        return rows

    def get_mode_name(self) -> str:
        return self.mode


class BasePatternStrategy(ScoringStrategy):
    mode: ScoringMode = "base"


class ModelBasedStrategy(ScoringStrategy):
    mode: ScoringMode = "calibrated"

    def __init__(self, model_bundle: dict[str, Any], feature_builder: FeatureBuildService, sort_only: bool = False) -> None:
        self.model_bundle = model_bundle
        self.feature_builder = feature_builder
        self.sort_only = sort_only

    def score_rows(self, rows: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
        if rows.empty:
            return rows
        X, _ = self.feature_builder.build_reranker_features(rows)
        columns = self.model_bundle.get("feature_columns", [])
        X = X.reindex(columns=columns, fill_value=0.0)
        probs = self.model_bundle["model"].predict_proba(X)[:, 1]
        out = rows.copy()
        out["calibrated_score"] = probs
        out["rerank_score"] = probs
        score_col = "rerank_score" if self.sort_only else "calibrated_score"
        out = out.sort_values(score_col, ascending=False)
        return out


class RerankedPatternStrategy(ModelBasedStrategy):
    mode: ScoringMode = "reranked"

    def __init__(self, model_bundle: dict[str, Any], feature_builder: FeatureBuildService) -> None:
        super().__init__(model_bundle=model_bundle, feature_builder=feature_builder, sort_only=True)


@dataclass
class PatternMonitorScoringPipeline:
    base: BasePatternStrategy
    calibrated: ScoringStrategy
    reranked: ScoringStrategy

    def score_rows(self, rows: pd.DataFrame, requested_mode: ScoringMode) -> tuple[pd.DataFrame, ScoringMode, bool]:
        strategy = {"base": self.base, "calibrated": self.calibrated, "reranked": self.reranked}.get(requested_mode, self.base)
        available = not isinstance(strategy, BasePatternStrategy) or requested_mode == "base"
        if requested_mode != "base" and isinstance(strategy, BasePatternStrategy):
            scored = self.base.score_rows(rows, {})
            return scored, "base", False
        scored = strategy.score_rows(rows, {})
        return scored, strategy.get_mode_name(), available


class CalibratorService:
    def __init__(self, models_dir: Path, feedback_service: FeedbackService, registry: ModelRegistryService, feature_builder: FeatureBuildService) -> None:
        self.models_dir = models_dir
        self.feedback_service = feedback_service
        self.registry = registry
        self.feature_builder = feature_builder
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train(self, request: dict[str, Any], scored_rows: pd.DataFrame) -> dict[str, Any]:
        feedback_rows = self.feedback_service.list_feedback(request)
        if not feedback_rows:
            raise ValueError("No feedback rows available for training")
        fb = pd.DataFrame(feedback_rows)
        fb = fb[fb["verdict"].isin(["true", "false"])].copy()
        if fb.empty:
            raise ValueError("No trainable feedback rows: only uncertain labels")
        merged = scored_rows.copy()
        if merged.empty:
            raise ValueError("No scored rows for selected filter")
        if "row_id" not in merged.columns:
            merged["row_id"] = [self.feedback_service.build_row_id(r) for r in merged.to_dict(orient="records")]
        merged = merged.merge(fb[["row_id", "verdict"]], on="row_id", how="inner")
        if merged.empty:
            raise ValueError("No overlap between scored rows and feedback labels")

        X, meta = self.feature_builder.build_reranker_features(merged)
        y = (merged["verdict"] == "true").astype(int)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        pred = (probs >= 0.5).astype(int)
        precision = float(((pred == 1) & (y == 1)).sum() / max((pred == 1).sum(), 1))

        version_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + f"_{uuid4().hex[:8]}"
        artifact_path = self.models_dir / f"reranker_{version_id}.joblib"
        bundle = {"model": model, "feature_columns": meta["columns"], "algorithm": "logistic_regression"}
        joblib.dump(bundle, artifact_path)

        record = {
            "version_id": version_id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "ready",
            "algorithm": request.get("algorithm") or "logistic_regression",
            "metrics_json": {"precision_reviewed": precision},
            "artifact_path": str(artifact_path),
            "train_rows": int(len(merged)),
            "active": 0,
            "notes": None,
        }
        self.registry.create_version(record)
        if request.get("activate_if_better"):
            self.registry.set_active(version_id, active=True)
        return record

    def _load_active_bundle(self) -> dict[str, Any] | None:
        active = self.registry.get_active()
        if not active or not active.get("artifact_path"):
            return None
        path = Path(active["artifact_path"])
        if not path.exists():
            return None
        bundle = joblib.load(path)
        bundle["version_id"] = active["version_id"]
        return bundle

    def scoring_pipeline(self) -> PatternMonitorScoringPipeline:
        bundle = self._load_active_bundle()
        if bundle is None:
            return PatternMonitorScoringPipeline(base=BasePatternStrategy(), calibrated=BasePatternStrategy(), reranked=BasePatternStrategy())
        return PatternMonitorScoringPipeline(
            base=BasePatternStrategy(),
            calibrated=ModelBasedStrategy(bundle, self.feature_builder),
            reranked=RerankedPatternStrategy(bundle, self.feature_builder),
        )
