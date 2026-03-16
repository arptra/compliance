from __future__ import annotations

import pandas as pd

from ..schemas import CategoryMetric, KPIItem, OverviewResponse, TimeSeriesPoint
from .data_loader import DataLoader
from .timeseries_service import (
    compute_category_compare_metrics,
    compute_expected_series,
    filter_by_date,
    resolve_compare_window,
)


class OverviewService:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def _source_df(self, viz_tag: str | None) -> pd.DataFrame:
        if viz_tag:
            df = self.loader.load_viz_state(viz_tag)
            if not df.empty:
                if "metric_count" in df.columns:
                    df = df.rename(columns={"metric_count": "count"})
                return df
        return self.loader.load_prepare_timeseries()

    def _timeseries(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        ds = df.copy()
        ds["date"] = pd.to_datetime(ds[date_col], errors="coerce").dt.normalize()
        ds = ds[ds["date"].notna()]
        if ds.empty:
            return pd.DataFrame(columns=["date", "actual"])

        if "count" in ds.columns:
            ts = ds.groupby("date")["count"].sum().rename("actual").reset_index()
        else:
            ts = ds.groupby("date").size().rename("actual").reset_index()
        ts["actual"] = ts["actual"].astype(float)
        return ts

    def get_overview(self, params: dict) -> OverviewResponse:
        df = self._source_df(params.get("viz_tag"))
        if df.empty:
            return OverviewResponse(kpis=[], actual_vs_expected=[], top_growth_categories=[], top_alert_categories=[], executive_summary="Нет данных", cards=[], sparklines={})

        date_col = "date" if "date" in df.columns else "event_time"
        df = filter_by_date(df, date_col, params.get("date_from"), params.get("date_to"))

        if "category" not in df.columns:
            df["category"] = "UNKNOWN"
        if "subcategory" not in df.columns:
            df["subcategory"] = "UNKNOWN"

        if params.get("category"):
            df = df[df["category"].isin(params["category"])]
        if params.get("subcategory"):
            df = df[df["subcategory"].isin(params["subcategory"])]

        ts = self._timeseries(df, date_col)

        baseline_start, baseline_end = resolve_compare_window(
            params.get("date_from"),
            params.get("date_to"),
            params.get("baseline_mode", "previous_period"),
            params.get("baseline_date_from"),
            params.get("baseline_date_to"),
        )

        hist = self.loader.load_prepare_timeseries()
        if not hist.empty:
            hist_col = "event_time" if "event_time" in hist.columns else ("date" if "date" in hist.columns else None)
            if hist_col:
                hist = self._timeseries(hist, hist_col)

        ts["expected"] = compute_expected_series(
            hist if not hist.empty else pd.DataFrame(),
            ts,
            baseline_start,
            baseline_end,
            params.get("baseline_mode", "previous_period"),
        )
        ts["delta"] = ts["actual"] - ts["expected"].fillna(0.0)

        baseline_df = self.loader.load_prepare_timeseries()
        baseline_col = "event_time" if "event_time" in baseline_df.columns else ("date" if "date" in baseline_df.columns else None)
        if baseline_col is not None:
            baseline_df = filter_by_date(
                baseline_df,
                baseline_col,
                str(baseline_start.date()) if baseline_start is not None else None,
                str(baseline_end.date()) if baseline_end is not None else None,
            )

        metrics = compute_category_compare_metrics(df, baseline_df)

        total = float(df["count"].sum()) if "count" in df.columns else float(len(df))
        expected_total = float(ts["expected"].sum()) if (not ts.empty and ts["expected"].notna().any()) else 0.0

        kpis = [
            KPIItem(key="actual_total", value=total),
            KPIItem(key="expected_total", value=expected_total),
            KPIItem(key="delta_total", value=total - expected_total),
        ]

        growth = [CategoryMetric(**row) for row in metrics.head(10).to_dict(orient="records")]
        alerts = [CategoryMetric(**row) for row in metrics.sort_values("anomaly_score", ascending=False).head(10).to_dict(orient="records")]
        tvs = [
            TimeSeriesPoint(
                date=r["date"].date(),
                actual=float(r["actual"]),
                expected=(None if pd.isna(r["expected"]) else float(r["expected"])),
                delta=float(r["delta"]),
            )
            for _, r in ts.iterrows()
        ]

        summary = f"Всего обращений: {int(total)}. Топ растущая категория: {(growth[0].category if growth else 'N/A')}."
        return OverviewResponse(
            kpis=kpis,
            actual_vs_expected=tvs,
            top_growth_categories=growth,
            top_alert_categories=alerts,
            executive_summary=summary,
            cards=[{"title": "Actual", "value": total}, {"title": "Expected", "value": expected_total}],
            sparklines={"overall": ts["actual"].astype(float).tolist() if not ts.empty else []},
        )
