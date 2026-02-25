from __future__ import annotations

import pandas as pd

from .config import ProjectConfig
from .reports.render import render_template


def build_trends(cfg: ProjectConfig) -> pd.DataFrame:
    df = pd.read_parquet(cfg.prepare.output_parquet)
    monthly = (
        df.groupby("month")
        .agg(complaint_rate=("is_complaint_llm", "mean"), n=("row_id", "count"))
        .reset_index()
        .sort_values("month")
    )
    by_cat = (
        df[df["is_complaint_llm"] == True]
        .groupby(["month", "complaint_category_llm"])
        .size()
        .reset_index(name="count")
    )
    render_template("compare_report.html.j2", "reports/trends_report.html", {"monthly": monthly.to_dict(orient="records"), "by_cat": by_cat.head(100).to_dict(orient="records")})
    return monthly
