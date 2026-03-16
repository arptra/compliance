from __future__ import annotations

import pandas as pd

from ..schemas import CategoryDetailResponse, CategoryMetric, CategoryTableResponse, ExamplesResponse, SubcategoryResponse, TimeSeriesPoint
from .data_loader import DataLoader
from .timeseries_service import compute_category_compare_metrics, filter_by_date, resolve_compare_window


class CategoriesService:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def _df(self, viz_tag: str | None, date_from: str | None, date_to: str | None) -> pd.DataFrame:
        df = self.loader.load_viz_state(viz_tag) if viz_tag else self.loader.load_prepare_timeseries()
        col = "date" if "date" in df.columns else "event_time"
        df = filter_by_date(df, col, date_from, date_to)
        if "category" not in df.columns:
            df["category"] = "UNKNOWN"
        if "subcategory" not in df.columns:
            df["subcategory"] = "UNKNOWN"
        return df

    def table(self, params: dict) -> CategoryTableResponse:
        df = self._df(params.get("viz_tag"), params.get("date_from"), params.get("date_to"))
        if df.empty:
            return CategoryTableResponse(rows=[])

        baseline_start, baseline_end = resolve_compare_window(
            params.get("date_from"),
            params.get("date_to"),
            params.get("baseline_mode", "previous_period"),
            params.get("baseline_date_from"),
            params.get("baseline_date_to"),
        )

        base = self.loader.load_prepare_timeseries()
        base_col = "event_time" if "event_time" in base.columns else ("date" if "date" in base.columns else None)
        if base_col:
            base = filter_by_date(
                base,
                base_col,
                str(baseline_start.date()) if baseline_start is not None else None,
                str(baseline_end.date()) if baseline_end is not None else None,
            )

        metrics = compute_category_compare_metrics(df, base)
        rows = [CategoryMetric(**r) for r in metrics.to_dict(orient="records")]
        return CategoryTableResponse(rows=rows)

    def category_timeseries(self, category: str, params: dict) -> CategoryDetailResponse:
        df = self._df(params.get("viz_tag"), params.get("date_from"), params.get("date_to"))
        date_col = "date" if "date" in df.columns else "event_time"
        df = df[df["category"].astype(str) == category]
        if df.empty:
            return CategoryDetailResponse(category=category, timeseries=[])

        df["d"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        val_col = "metric_count" if "metric_count" in df.columns else ("count" if "count" in df.columns else None)
        if val_col:
            ts = df.groupby("d")[val_col].sum().rename("actual").reset_index()
        else:
            ts = df.groupby("d").size().rename("actual").reset_index()
        rows = [TimeSeriesPoint(date=r["d"], actual=float(r["actual"]), expected=None, delta=None) for _, r in ts.iterrows()]
        return CategoryDetailResponse(category=category, timeseries=rows)

    def subcategories(self, category: str, params: dict) -> SubcategoryResponse:
        df = self._df(params.get("viz_tag"), params.get("date_from"), params.get("date_to"))
        df = df[df["category"].astype(str) == category]
        if df.empty:
            return SubcategoryResponse(category=category, subcategories=[])

        val_col = "metric_count" if "metric_count" in df.columns else ("count" if "count" in df.columns else None)
        if val_col:
            out = df.groupby("subcategory")[val_col].sum().rename("count").reset_index()
        else:
            out = df.groupby("subcategory").size().rename("count").reset_index()
        total = max(float(out["count"].sum()), 1.0)
        out["share"] = out["count"] / total
        out = out.sort_values("count", ascending=False)
        return SubcategoryResponse(category=category, subcategories=out.to_dict(orient="records"))

    def examples(self, category: str, limit: int = 20) -> ExamplesResponse:
        df = self.loader.load_prepare()
        if "category" in df.columns:
            df = df[df["category"].astype(str) == category]
        cols = [c for c in ["event_time", "category", "subcategory", "client_first_message", "dialog_text"] if c in df.columns]
        ex = df[cols].head(limit).to_dict(orient="records") if (not df.empty and cols) else []
        return ExamplesResponse(category=category, examples=ex)
