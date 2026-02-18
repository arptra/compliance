from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.cluster import assign_clusters, cluster_summaries, save_cluster_artifacts, train_cluster_model
from app.complaint import train_complaint_classifier
from app.io import get_message_columns, load_and_split
from app.novelty import max_similarity_to_centroids, novelty_threshold_from_baseline
from app.preprocess import TextPreprocessor
from app.utils import dump_json, ensure_dir, load_yaml, setup_logging, timed_step
from app.vectorize import fit_transform, save_vectorizer


def run_train(config_path: str, sample: int | None = None) -> None:
    cfg = load_yaml(config_path)
    setup_logging(cfg.get("logging_level", "INFO"))

    model_dir = ensure_dir(cfg["output"]["model_dir"])
    reports_dir = ensure_dir("reports")
    ensure_dir("outputs")

    with timed_step("load-and-split"):
        splits = load_and_split(cfg)
        baseline = splits.baseline.copy()
        if sample:
            baseline = baseline.sample(min(sample, len(baseline)), random_state=cfg.get("random_state", 42))

    preprocessor = TextPreprocessor(cfg)
    message_cols = get_message_columns(cfg["input"])
    msg_col = "message_joined"

    with timed_step("preprocess"):
        baseline["processed_text"] = preprocessor.preprocess_series(baseline[msg_col])
        december_texts_for_vocab = preprocessor.preprocess_series(splits.december[msg_col])

    with timed_step("vectorize"):
        vectorizer, x_baseline, x_december = fit_transform(
            baseline["processed_text"].tolist(),
            december_texts_for_vocab,
            cfg,
        )
        save_vectorizer(vectorizer, model_dir)

    with timed_step("cluster-train"):
        cluster_model = train_cluster_model(x_baseline, cfg)
        baseline_cluster_ids, baseline_cluster_sims = assign_clusters(x_baseline, cluster_model)
        baseline["cluster_id"] = baseline_cluster_ids
        baseline["cluster_sim_to_centroid"] = baseline_cluster_sims
        csum = cluster_summaries(
            x_baseline,
            baseline[msg_col].astype(str).tolist(),
            baseline_cluster_ids,
            cluster_model,
            vectorizer.get_feature_names_out(),
            cfg,
        )
        save_cluster_artifacts(cluster_model, csum, model_dir, reports_dir)

    with timed_step("complaint-train"):
        complaint_model, seed_labels, seed_reasons, metrics = train_complaint_classifier(
            x_baseline,
            baseline[msg_col].astype(str).tolist(),
            cfg,
            "configs/complaint_lexicon.yaml",
            model_dir,
            reports_dir,
        )
        baseline["seed_complaint"] = seed_labels
        baseline["seed_reasons"] = seed_reasons
        baseline["complaint_score"] = complaint_model.score(x_baseline)
        threshold = float(cfg["complaint"].get("threshold", 0.5))
        baseline["is_complaint"] = (baseline["complaint_score"] >= threshold).astype(int)

    with timed_step("novelty-threshold"):
        centroid_norm = cluster_model.cluster_centers_
        baseline_max_sim = max_similarity_to_centroids(x_baseline, centroid_norm)
        nov_percentile = float(cfg["novelty"].get("percentile", 2))
        nov_threshold = novelty_threshold_from_baseline(baseline_max_sim, nov_percentile)
        baseline["max_sim_to_baseline_centroid"] = baseline_max_sim
        baseline["is_novel"] = (baseline_max_sim < nov_threshold).astype(int)

    with timed_step("save-train-artifacts"):
        meta = {
            "config_path": config_path,
            "message_cols": message_cols,
            "message_col_runtime": msg_col,
            "complaint_threshold": threshold,
            "novelty_threshold": nov_threshold,
            "novelty_percentile": nov_percentile,
        }
        dump_json(meta, model_dir / "metadata.json")
        dump_json(metrics, reports_dir / "complaint_seed_metrics.json")
        baseline.to_excel(cfg["output"]["labeled_baseline_xlsx"], index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train clustering + weak complaint model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run_train(args.config, args.sample)
