from __future__ import annotations

import pandas as pd

from ..schemas import CategoryDetailResponse, CategoryMetric, CategoryTableResponse, ExamplesResponse, SubcategoryResponse, TimeSeriesPoint
from .data_loader import DataLoader
from .timeseries_service import filter_by_date


class CategoriesService:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def _df(self, viz_tag: str | None, date_from: str | None, date_to: str | None) -> pd.DataFrame:
        df = self.loader.load_viz_state(viz_tag) if viz_tag else self.loader.load_prepare()
        col = "date" if "date" in df.columns else "event_time"
        return filter_by_date(df, col, date_from, date_to)

    def table(self, params: dict) -> CategoryTableResponse:
        df = self._df(params.get("viz_tag"), params.get("date_from"), params.get("date_to"))
        if df.empty:
            return CategoryTableResponse(rows=[])
        if "category" not in df.columns:
            df["category"] = "UNKNOWN"
        g = df.groupby("category").size().rename("count").reset_index()
        total = max(float(g["count"].sum()), 1.0)
        g["share"] = g["count"] / total
        g["baseline_count"] = None
        g["delta_abs"] = None
        g["delta_pct"] = None
        g["anomaly_score"] = None
        g["pattern_score"] = None
        return CategoryTableResponse(rows=[CategoryMetric(**r) for r in g.sort_values("count", ascending=False).to_dict(orient="records")])

    def category_timeseries(self, category: str, params: dict) -> CategoryDetailResponse:
        df = self._df(params.get("viz_tag"), params.get("date_from"), params.get("date_to"))
        date_col = "date" if "date" in df.columns else "event_time"
        if "category" in df.columns:
            df = df[df["category"] == category]
        df["d"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        ts = df.groupby("d").size().rename("actual").reset_index()
        rows = [TimeSeriesPoint(date=r["d"], actual=float(r["actual"]), expected=None, delta=None) for _, r in ts.iterrows()]
        return CategoryDetailResponse(category=category, timeseries=rows)

    def subcategories(self, category: str, params: dict) -> SubcategoryResponse:
        df = self._df(params.get("viz_tag"), params.get("date_from"), params.get("date_to"))
        if "category" in df.columns:
            df = df[df["category"] == category]
        if "subcategory" not in df.columns:
            return SubcategoryResponse(category=category, subcategories=[])
        out = df.groupby("subcategory").size().rename("count").reset_index().sort_values("count", ascending=False)
        return SubcategoryResponse(category=category, subcategories=out.to_dict(orient="records"))

    def examples(self, category: str, limit: int = 20) -> ExamplesResponse:
        df = self.loader.load_prepare()
        if "category" in df.columns:
            df = df[df["category"] == category]
        cols = [c for c in ["event_time", "category", "subcategory", "client_first_message", "dialog_text"] if c in df.columns]
        ex = df[cols].head(limit).to_dict(orient="records") if not df.empty else []
        return ExamplesResponse(category=category, examples=ex)
