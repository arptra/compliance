from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def _md(df: pd.DataFrame, cols: list[str], n: int = 10) -> str:
    if df.empty:
        return "(нет данных)"
    return df[cols].head(n).to_markdown(index=False)


def build_pilot_report(month: str, labeled_df: pd.DataFrame, cluster_df: pd.DataFrame, complaint_examples_md: str) -> str:
    return f"""# Pilot report ({month})

## How to review
1. Проверьте, что `message_client_first` — это действительно первое сообщение клиента, без хвоста оператора/бота.
2. Проверьте кластеры по **примерам** (а не только по словам).
3. Проверьте жалобы: top, borderline, non-complaints.

## Dataset size
- Rows: **{len(labeled_df)}**

## Pilot clusters
{_md(cluster_df, ['cluster_id','size','top_terms','example_messages'], n=20)}

## Complaint examples
{complaint_examples_md}
"""


def build_train_report(baseline_df: pd.DataFrame, cluster_df: pd.DataFrame, metrics: Dict) -> str:
    return f"""# Full Train report

- Baseline rows: **{len(baseline_df)}**
- Seed precision: **{metrics.get('precision',0):.3f}**
- Seed recall: **{metrics.get('recall',0):.3f}**
- Seed f1: **{metrics.get('f1',0):.3f}**

## Baseline clusters
{_md(cluster_df, ['cluster_id','size','top_terms','example_messages'], n=20)}
"""


def build_december_report(
    baseline_df: pd.DataFrame,
    december_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    metrics: Dict,
    novelty_threshold: float,
    novelty_percentile: float,
    emerging_df: pd.DataFrame,
    novel_clusters_df: pd.DataFrame,
    complaint_clusters_df: pd.DataFrame,
) -> str:
    novel = december_df[(december_df["is_complaint"] == 1) & (december_df["is_novel"] == 1)]
    return f"""# December/Target comparison report

## How to read this report
- Category = cluster похожих сообщений
- Complaint = бинарный флаг жалобы
- Novel = низкое сходство с baseline-кластерами

- Baseline rows: **{len(baseline_df)}**
- Target rows: **{len(december_df)}**
- Novelty threshold percentile: **{novelty_percentile}**
- Novelty threshold value: **{novelty_threshold:.4f}**
- Target complaints (all): **{int((december_df["is_complaint"]==1).sum())}**
- Novel complaints: **{len(novel)}**
- Non-novel complaints: **{int(((december_df["is_complaint"]==1)&(december_df["is_novel"]==0)).sum())}**

> Важно: число baseline-кластеров фиксировано моделью Stage 2. Для декабря дополнительно смотрите отдельные complaint-кластеры ниже.

## Cluster overview
{_md(cluster_summary_df, ['cluster_id','size','top_terms','example_messages'], n=10)}

## Emerging terms
{_md(emerging_df, [c for c in ['term','december_df','lift'] if c in emerging_df.columns], n=20)}

## December complaint groups (all complaints)
{_md(complaint_clusters_df, ['cluster_id','size','top_terms','example_messages'], n=10)}

## Novel complaint groups
{_md(novel_clusters_df, ['cluster_id','size','top_terms','example_messages'], n=10)}
"""


def save_report(text: str, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
