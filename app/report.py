from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def build_report(
    baseline_df: pd.DataFrame,
    december_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    metrics: Dict,
    novelty_threshold: float,
    novelty_percentile: float,
    emerging_df: pd.DataFrame,
    novel_clusters_df: pd.DataFrame,
    cfg: Dict,
) -> str:
    baseline_complaint_rate = baseline_df["is_complaint"].mean() if len(baseline_df) else 0.0
    december_complaint_rate = december_df["is_complaint"].mean() if len(december_df) else 0.0
    novel_complaints = december_df[(december_df["is_complaint"] == 1) & (december_df["is_novel"] == 1)]
    novel_share = (len(novel_complaints) / len(december_df)) if len(december_df) else 0.0

    top_clusters = cluster_summary_df.head(10)
    top_clusters_md = top_clusters[["cluster_id", "size", "top_terms"]].to_markdown(index=False)
    emerging_md = emerging_df.head(15).to_markdown(index=False)
    novel_themes_md = novel_clusters_df.head(10).to_markdown(index=False) if len(novel_clusters_df) else "No novel complaint clusters."

    artifacts = cfg["output"]
    return f"""# Complaint & Novel Context Report

## 1) Dataset sizes
- Baseline rows: **{len(baseline_df)}**
- December rows: **{len(december_df)}**

## 2) Topic clustering overview
- Number of baseline clusters: **{cfg['clustering']['n_clusters']}**
- Top 10 clusters by volume:

{top_clusters_md}

## 3) Complaint detection overview (weak supervision)
- Seeded validation precision: **{metrics['precision']:.3f}**
- Seeded validation recall: **{metrics['recall']:.3f}**
- Seeded validation F1: **{metrics['f1']:.3f}**
- Baseline complaint rate: **{baseline_complaint_rate:.2%}**
- December complaint rate: **{december_complaint_rate:.2%}**

## 4) Novelty detection
- Novelty threshold percentile: **p={novelty_percentile}**
- Novelty similarity threshold value: **{novelty_threshold:.4f}**
- Novel December complaints: **{len(novel_complaints)}** ({novel_share:.2%} of December)

### Top emerging terms (December vs baseline)
{emerging_md}

### Novel complaint subclusters
{novel_themes_md}

## 5) Artifact paths
- Baseline labeled: `{artifacts['labeled_baseline_xlsx']}`
- December labeled: `{artifacts['labeled_december_xlsx']}`
- Combined labeled: `{artifacts['combined_xlsx']}`
- Cluster summaries CSV: `reports/cluster_summaries.csv`
- Novel complaint clusters CSV: `reports/december_novel_complaints_clusters.csv`
- Emerging terms CSV: `reports/emerging_terms.csv`
- Metrics JSON: `reports/complaint_seed_metrics.json`
- Model directory: `{artifacts['model_dir']}`
"""


def save_report(report_text: str, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report_text, encoding="utf-8")
