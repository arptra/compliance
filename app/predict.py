from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from app.cluster import assign_clusters
from app.io import get_message_columns, load_and_split, save_labeled_outputs
from app.novelty import (
    cluster_novel_complaints,
    compute_novelty_flags,
    emerging_terms,
    save_novelty_reports,
)
from app.preprocess import TextPreprocessor
from app.report import build_report, save_report
from app.utils import load_yaml, setup_logging, timed_step


def run_predict(config_path: str) -> None:
    cfg = load_yaml(config_path)
    setup_logging(cfg.get("logging_level", "INFO"))

    model_dir = Path(cfg["output"]["model_dir"])
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    preprocessor = TextPreprocessor(cfg)
    cluster_model = joblib.load(model_dir / "cluster_model.joblib")
    complaint_model = joblib.load(model_dir / "complaint_model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    with timed_step("load-and-split"):
        splits = load_and_split(cfg)
        baseline = splits.baseline.copy()
        december = splits.december.copy()

    get_message_columns(cfg["input"])  # validate config structure
    msg_col = metadata.get("message_col_runtime", "message_joined")

    with timed_step("preprocess+vectorize"):
        baseline["processed_text"] = preprocessor.preprocess_series(baseline[msg_col])
        december["processed_text"] = preprocessor.preprocess_series(december[msg_col])
        x_baseline = vectorizer.transform(baseline["processed_text"])
        x_december = vectorizer.transform(december["processed_text"])

    with timed_step("cluster+complaint"):
        baseline["cluster_id"], baseline["cluster_sim_to_centroid"] = assign_clusters(x_baseline, cluster_model)
        december["cluster_id"], december["cluster_sim_to_centroid"] = assign_clusters(x_december, cluster_model)

        threshold = float(metadata["complaint_threshold"])
        baseline["complaint_score"] = complaint_model.score(x_baseline)
        december["complaint_score"] = complaint_model.score(x_december)
        baseline["is_complaint"] = (baseline["complaint_score"] >= threshold).astype(int)
        december["is_complaint"] = (december["complaint_score"] >= threshold).astype(int)

    with timed_step("novelty"):
        centroids = cluster_model.cluster_centers_
        baseline["max_sim_to_baseline_centroid"], baseline["is_novel"] = compute_novelty_flags(
            x_baseline,
            centroids,
            float(metadata["novelty_threshold"]),
        )
        december["max_sim_to_baseline_centroid"], december["is_novel"] = compute_novelty_flags(
            x_december,
            centroids,
            float(metadata["novelty_threshold"]),
        )

        terms_df = emerging_terms(x_baseline, x_december, vectorizer.get_feature_names_out(), top_n=30)
        novel_mask = (december["is_complaint"] == 1) & (december["is_novel"] == 1)
        x_novel = x_december[novel_mask.values]
        novel_texts = december.loc[novel_mask, msg_col].astype(str).tolist()
        novel_clusters_df = cluster_novel_complaints(x_novel, novel_texts, vectorizer.get_feature_names_out(), cfg)
        save_novelty_reports(terms_df, novel_clusters_df, "reports")

    with timed_step("export-and-report"):
        combined = pd.concat([baseline.assign(split="baseline"), december.assign(split="december")], ignore_index=True)
        save_labeled_outputs(baseline, december, combined, cfg["output"])
        cluster_summary_df = pd.read_csv("reports/cluster_summaries.csv")
        metrics = json.loads(Path("reports/complaint_seed_metrics.json").read_text(encoding="utf-8"))
        report = build_report(
            baseline_df=baseline,
            december_df=december,
            cluster_summary_df=cluster_summary_df,
            metrics=metrics,
            novelty_threshold=float(metadata["novelty_threshold"]),
            novelty_percentile=float(metadata["novelty_percentile"]),
            emerging_df=terms_df,
            novel_clusters_df=novel_clusters_df,
            cfg=cfg,
        )
        save_report(report, cfg["output"]["report_md"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and generate novelty report")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_predict(args.config)
