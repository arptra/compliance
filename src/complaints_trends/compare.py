from __future__ import annotations

import pandas as pd
import joblib
from pathlib import Path

from .config import ProjectConfig
from .novelty import cluster_novel_texts, compute_novelty_scores
from .reports.render import render_template


def compare_month(cfg: ProjectConfig, new_month: str, baseline_range: str) -> pd.DataFrame:
    start, end = baseline_range.split("..")
    all_df = pd.read_parquet(cfg.prepare.output_parquet)
    new_df = pd.read_parquet(f"data/interim/month_{new_month}.parquet")
    base = all_df[(all_df["month"] >= start) & (all_df["month"] <= end) & (all_df["is_complaint_llm"] == True)].copy()
    new_c = new_df[new_df["is_complaint_pred"] == True].copy()

    vec = joblib.load(Path(cfg.training.model_dir) / "vectorizers.joblib")
    xb = vec.transform(base["client_first_message"].astype(str))
    xn = vec.transform(new_c["client_first_message"].astype(str))

    scores, thr, znew = compute_novelty_scores(
        xb, xn,
        method=cfg.analysis.novelty.method,
        svd_components=cfg.analysis.novelty.svd_components,
        kmeans_k=cfg.analysis.novelty.kmeans_k,
        threshold_percentile=cfg.analysis.novelty.threshold_percentile,
    )
    new_c["novelty_score"] = scores
    new_c["is_novel"] = new_c["novelty_score"] > thr
    new_c["cluster_id"] = cluster_novel_texts(znew, new_c["is_novel"].values)

    out = f"exports/new_topics_{new_month}.xlsx"
    new_c.to_excel(out, index=False)

    cat_base = base["complaint_category_llm"].value_counts(normalize=True)
    cat_new = new_c["category_pred"].value_counts(normalize=True)
    merged = pd.DataFrame({"baseline_share": cat_base, "new_share": cat_new}).fillna(0)
    merged["delta_pp"] = (merged["new_share"] - merged["baseline_share"]) * 100

    render_template("compare_report.html.j2", f"reports/compare_{new_month}_vs_baseline.html", {
        "monthly": merged.reset_index().rename(columns={"index": "category"}).to_dict(orient="records"),
        "by_cat": new_c[new_c["is_novel"]].head(100).to_dict(orient="records"),
    })
    return new_c
