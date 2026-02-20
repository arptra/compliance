from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from app.cluster import assign_clusters
from app.io import load_and_split, save_labeled_outputs
from app.novelty import cluster_december_complaints, cluster_novel_complaints, compute_novelty_flags, emerging_terms, save_novelty_reports
from app.preprocess import TextPreprocessor
from app.report import build_december_report, save_report
from app.utils import load_yaml, setup_logging


def run_compare_december(config_path: str) -> None:
    cfg = load_yaml(config_path)
    setup_logging(cfg.get("logging_level", "INFO"))

    model_dir = Path(cfg["output"]["model_dir"])
    required = [model_dir / "vectorizer.joblib", model_dir / "cluster_model.joblib", model_dir / "complaint_model.joblib", model_dir / "metadata.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing models from stage-2 train_full: {missing}")

    vec = joblib.load(model_dir / "vectorizer.joblib")
    km = joblib.load(model_dir / "cluster_model.joblib")
    cm = joblib.load(model_dir / "complaint_model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    pre = TextPreprocessor(cfg)

    splits = load_and_split(cfg)
    base = splits.baseline.copy()
    dec = splits.december.copy()

    msg_col = meta.get("message_col_runtime", "message_client_first")
    x_base = vec.transform(base[msg_col].astype(str).tolist())
    x_dec = vec.transform(dec[msg_col].astype(str).tolist())

    base["cluster_id"], base["cluster_sim_to_centroid"] = assign_clusters(x_base, km)
    dec["cluster_id"], dec["cluster_sim_to_centroid"] = assign_clusters(x_dec, km)

    thr = float(meta["complaint_threshold"])
    base["complaint_score"] = cm.score(x_base)
    dec["complaint_score"] = cm.score(x_dec)
    base["is_complaint"] = (base["complaint_score"] >= thr).astype(int)
    dec["is_complaint"] = (dec["complaint_score"] >= thr).astype(int)

    n_th = float(meta["novelty_threshold"])
    base["max_sim_to_baseline_centroid"], base["is_novel"] = compute_novelty_flags(x_base, km.cluster_centers_, n_th)
    dec["max_sim_to_baseline_centroid"], dec["is_novel"] = compute_novelty_flags(x_dec, km.cluster_centers_, n_th)

    terms = emerging_terms(x_base, x_dec, vec.get_feature_names_out(), pre, cfg)
    complaint_mask = (dec["is_complaint"] == 1).values
    novel_mask = ((dec["is_complaint"] == 1) & (dec["is_novel"] == 1)).values
    novel_clusters = cluster_novel_complaints(x_dec[novel_mask], dec.loc[novel_mask, msg_col].astype(str).tolist(), vec.get_feature_names_out(), cfg, pre)
    complaint_clusters = cluster_december_complaints(x_dec, complaint_mask, dec[msg_col].astype(str).tolist(), vec.get_feature_names_out(), cfg, pre)
    save_novelty_reports(terms, novel_clusters, "reports")
    complaint_clusters.to_csv("reports/december_complaint_clusters.csv", index=False)

    combined = pd.concat([base.assign(split="baseline"), dec.assign(split="december")], ignore_index=True)
    save_labeled_outputs(base, dec, combined, cfg["output"])

    cl_sum = pd.read_csv("reports/cluster_summaries.csv")
    metrics = json.loads(Path("reports/complaint_seed_metrics.json").read_text(encoding="utf-8")) if Path("reports/complaint_seed_metrics.json").exists() else {}
    report = build_december_report(base, dec, cl_sum, metrics, n_th, float(meta["novelty_percentile"]), terms, novel_clusters, complaint_clusters)
    save_report(report, "reports/december_report.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    run_compare_december(a.config)
