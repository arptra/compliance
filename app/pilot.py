from __future__ import annotations

import argparse

from pathlib import Path

import pandas as pd

from app.cluster import assign_clusters, cluster_summaries, train_cluster_model
from app.complaint import build_seed_labels, train_complaint_classifier
from app.io import load_input_df, split_baseline_december
from app.report import build_pilot_report, save_report
from app.utils import ensure_dir, load_yaml, setup_logging
from app.vectorize import fit_transform


def run_pilot(config_path: str, month: str) -> None:
    cfg = load_yaml(config_path)
    setup_logging(cfg.get("logging_level", "INFO"))

    df = load_input_df(cfg)
    i = cfg["input"]
    # temporary month split for pilot
    i_pilot = {**i, "path": i.get("path"), "baseline": {"mode": "month_value", "month_value": month}, "december": {"mode": "month_value", "month_value": month}}
    splits = split_baseline_december(df, {**cfg, "input": i_pilot})
    pilot_df = splits.baseline.copy()

    pre = __import__("app.preprocess", fromlist=["TextPreprocessor"]).TextPreprocessor(cfg)
    vec, x, _ = fit_transform(pilot_df["message_client_first"].tolist(), pilot_df["message_client_first"].tolist(), cfg, analyzer=pre.analyzer)
    k = int(cfg.get("pilot", {}).get("n_clusters", 20))
    c_cfg = {**cfg, "clustering": {**cfg.get("clustering", {}), "n_clusters": k}}
    km = train_cluster_model(x, c_cfg)
    cid, csim = assign_clusters(x, km)
    pilot_df["cluster_id"] = cid
    pilot_df["cluster_sim_to_centroid"] = csim
    cs = cluster_summaries(x, pilot_df["message_client_first"].tolist(), cid, km, vec.get_feature_names_out(), cfg, pre)

    mode = cfg.get("pilot", {}).get("complaint_mode", "rules_only")
    if mode == "weak_train":
        model, _, _, _ = train_complaint_classifier(x, pilot_df["message_client_first"].tolist(), cfg, "configs/complaint_lexicon.yaml", "models", "reports")
        pilot_df["complaint_score"] = model.score(x)
    else:
        labels, reasons = build_seed_labels(pilot_df["message_client_first"].tolist(), "configs/complaint_lexicon.yaml")
        pilot_df["complaint_score"] = pd.Series(labels).fillna(0).astype(float)
        pilot_df["seed_reasons"] = reasons

    pilot_df["is_complaint"] = (pilot_df["complaint_score"] >= cfg.get("complaint", {}).get("threshold", 0.5)).astype(int)

    ensure_dir("outputs")
    ensure_dir("reports")
    pilot_df.to_excel("outputs/pilot_labeled.xlsx", index=False)
    cs.to_csv("reports/pilot_cluster_summaries.csv", index=False)

    pos = pilot_df.sort_values("complaint_score", ascending=False).head(30)
    mid = pilot_df.loc[(pilot_df["complaint_score"] - cfg.get("complaint", {}).get("threshold", 0.5)).abs().sort_values().index].head(30)
    neg = pilot_df.sort_values("complaint_score", ascending=True).head(30)
    examples_md = "\n\n".join(
        [
            "### Top complaints\n" + pos[["message_client_first", "message_raw", "complaint_score"]].to_markdown(index=False),
            "### Borderline\n" + mid[["message_client_first", "message_raw", "complaint_score"]].to_markdown(index=False),
            "### Non-complaints\n" + neg[["message_client_first", "message_raw", "complaint_score"]].to_markdown(index=False),
        ]
    )
    save_report(examples_md, "reports/pilot_complaint_examples.md")
    save_report(build_pilot_report(month, pilot_df, cs, examples_md), "reports/pilot_report.md")

    approval = cfg.get("stages", {}).get("approval_file", "reports/pilot_approval.txt")
    Path(approval).write_text("Write APPROVED here after review to unlock train_full.\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--month", required=True)
    a = p.parse_args()
    run_pilot(a.config, a.month)
