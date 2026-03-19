from __future__ import annotations

from typing import Any

import pandas as pd


class FeatureBuildService:
    NUMERIC_FEATURES = [
        "pattern_like_score",
        "event_similarity",
        "normal_similarity",
        "normal_distance",
        "row_novelty_to_normal",
        "day_category_anomaly",
        "overall_pressure",
        "smoothed_state",
        "score",
        "row_score",
    ]
    CATEGORICAL_FEATURES = ["category", "subcategory", "cluster_id"]

    def build_reranker_features(self, scored_rows_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame = scored_rows_df.copy()
        for c in self.NUMERIC_FEATURES:
            frame[c] = pd.to_numeric(frame[c], errors="coerce") if c in frame.columns else 0.0
        for c in self.CATEGORICAL_FEATURES:
            frame[c] = frame[c].fillna("UNKNOWN").astype(str) if c in frame.columns else "UNKNOWN"

        features = pd.DataFrame(index=frame.index)
        for c in self.NUMERIC_FEATURES:
            features[c] = frame[c] if c in frame.columns else 0.0
        for c in self.CATEGORICAL_FEATURES:
            if c in frame.columns:
                dummies = pd.get_dummies(frame[c], prefix=c)
                features = pd.concat([features, dummies], axis=1)

        features = features.fillna(0.0)
        return features, {"columns": list(features.columns)}
