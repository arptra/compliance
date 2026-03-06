from __future__ import annotations

import pandas as pd
import joblib
from pathlib import Path

from .config import ProjectConfig
from .novelty import cluster_novel_texts, compute_novelty_scores
from .reports.render import render_template


def _pick_first_existing(df: pd.DataFrame, candidates: list[str], default: str) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=object)


def _share_table(base_series: pd.Series, new_series: pd.Series) -> pd.DataFrame:
    b = base_series.value_counts(normalize=True) if len(base_series) else pd.Series(dtype=float)
    n = new_series.value_counts(normalize=True) if len(new_series) else pd.Series(dtype=float)
    out = pd.DataFrame({"baseline_share": b, "new_share": n}).fillna(0)
    out["delta_pp"] = (out["new_share"] - out["baseline_share"]) * 100
    return out.sort_values("delta_pp", ascending=False)


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

    base_cat = _pick_first_existing(base, ["complaint_category_llm", "category_pred"], default="OTHER")
    new_cat = _pick_first_existing(new_c, ["category_pred", "complaint_category_llm"], default="OTHER")
    base_sub = _pick_first_existing(base, ["complaint_subcategory_llm", "subcategory_pred"], default="UNKNOWN")
    new_sub = _pick_first_existing(new_c, ["subcategory_pred", "complaint_subcategory_llm"], default="UNKNOWN")

    merged_cat = _share_table(base_cat, new_cat)
    merged_sub = _share_table(base_sub, new_sub)

    render_template("compare_report.html.j2", f"reports/compare_{new_month}_vs_baseline.html", {
        "monthly": merged_cat.reset_index().rename(columns={"index": "category"}).to_dict(orient="records"),
        "monthly_subcategories": merged_sub.reset_index().rename(columns={"index": "subcategory"}).head(50).to_dict(orient="records"),
        "by_cat": new_c[new_c["is_novel"]].head(100).to_dict(orient="records"),
    })
    return new_c
