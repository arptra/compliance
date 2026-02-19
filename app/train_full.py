from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.cluster import assign_clusters, cluster_summaries, save_cluster_artifacts, train_cluster_model
from app.complaint import train_complaint_classifier
from app.io import load_and_split
from app.novelty import max_similarity_to_centroids, novelty_threshold_from_baseline
from app.preprocess import TextPreprocessor
from app.report import build_train_report, save_report
from app.utils import dump_json, ensure_dir, load_yaml, setup_logging
from app.vectorize import fit_transform, save_vectorizer


def _check_approval(cfg: dict) -> None:
    st = cfg.get("stages", {})
    if not st.get("require_pilot_approval", False):
        return
    p = Path(st.get("approval_file", "reports/pilot_approval.txt"))
    if not p.exists() or "APPROVED" not in p.read_text(encoding="utf-8").upper():
        raise SystemExit(f"Pilot approval missing. Put APPROVED in {p}")


def run_train_full(config_path: str) -> None:
    cfg = load_yaml(config_path)
    setup_logging(cfg.get("logging_level", "INFO"))
    _check_approval(cfg)

    splits = load_and_split(cfg)
    baseline = splits.baseline.copy()
    pre = TextPreprocessor(cfg)

    ensure_dir(cfg["output"]["model_dir"])
    ensure_dir("reports")

    vec, x_base, _ = fit_transform(baseline["message_client_first"].tolist(), splits.december["message_client_first"].tolist(), cfg, analyzer=pre.analyzer)
    save_vectorizer(vec, cfg["output"]["model_dir"])

    km = train_cluster_model(x_base, cfg)
    cid, csim = assign_clusters(x_base, km)
    baseline["cluster_id"] = cid
    baseline["cluster_sim_to_centroid"] = csim
    csum = cluster_summaries(x_base, baseline["message_client_first"].tolist(), cid, km, vec.get_feature_names_out(), cfg, pre)
    save_cluster_artifacts(km, csum, cfg["output"]["model_dir"], "reports")

    model, seed, reasons, metrics = train_complaint_classifier(x_base, baseline["message_client_first"].tolist(), cfg, "configs/complaint_lexicon.yaml", cfg["output"]["model_dir"], "reports")
    baseline["seed_complaint"] = seed
    baseline["seed_reasons"] = reasons
    baseline["complaint_score"] = model.score(x_base)
    thr = float(cfg.get("complaint", {}).get("threshold", 0.5))
    baseline["is_complaint"] = (baseline["complaint_score"] >= thr).astype(int)

    bsim = max_similarity_to_centroids(x_base, km.cluster_centers_)
    nperc = float(cfg.get("novelty", {}).get("percentile", 2))
    nth = novelty_threshold_from_baseline(bsim, nperc)
    baseline["max_sim_to_baseline_centroid"] = bsim
    baseline["is_novel"] = (bsim < nth).astype(int)

    baseline.to_excel(cfg["output"]["labeled_baseline_xlsx"], index=False)
    meta = {"complaint_threshold": thr, "novelty_threshold": nth, "novelty_percentile": nperc, "message_col_runtime": "message_client_first"}
    dump_json(meta, Path(cfg["output"]["model_dir"]) / "metadata.json")

    save_report(build_train_report(baseline, csum, metrics), "reports/train_report.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    run_train_full(a.config)
