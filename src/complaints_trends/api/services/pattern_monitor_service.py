from __future__ import annotations

import pandas as pd

from ..schemas import AlertRowResponse, DailyPressureResponse, OverallStateResponse, PatternMonitorSummaryResponse
from .data_loader import DataLoader


class PatternMonitorService:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def _filter(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        out = df.copy()
        if out.empty:
            return out
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            if params.get("date_from"):
                out = out[out["date"] >= pd.to_datetime(params["date_from"])]
            if params.get("date_to"):
                out = out[out["date"] <= pd.to_datetime(params["date_to"])]
        if params.get("category") and "category" in out.columns:
            out = out[out["category"].isin(params["category"])]
        if params.get("min_score") is not None:
            score_col = "row_score" if "row_score" in out.columns else ("score" if "score" in out.columns else None)
            if score_col:
                out = out[out[score_col] >= float(params["min_score"])]
        return out

    def summary(self, tag: str, params: dict) -> PatternMonitorSummaryResponse:
        scored = self._filter(self.loader.load_pattern_monitor_scored(tag), params)
        pressure = self._filter(self.loader.load_pattern_monitor_pressure(tag), params)
        return PatternMonitorSummaryResponse(
            tag=tag,
            summary={
                "scored_rows": int(len(scored)),
                "alert_rows": int((scored.get("is_alert", False) == True).sum()) if not scored.empty and "is_alert" in scored.columns else 0,
                "pressure_days": int(len(pressure)),
            },
        )

    def alerts(self, tag: str, params: dict) -> AlertRowResponse:
        scored = self._filter(self.loader.load_pattern_monitor_scored(tag), params)
        if "is_alert" in scored.columns:
            scored = scored[scored["is_alert"] == True]
        return AlertRowResponse(rows=scored.head(int(params.get("top_n", 200))).to_dict(orient="records"))

    def pressure(self, tag: str, params: dict) -> DailyPressureResponse:
        data = self._filter(self.loader.load_pattern_monitor_pressure(tag), params)
        return DailyPressureResponse(rows=data.to_dict(orient="records"))

    def state(self, tag: str, params: dict) -> OverallStateResponse:
        data = self._filter(self.loader.load_pattern_monitor_state(tag), params)
        return OverallStateResponse(rows=data.to_dict(orient="records"))

    def examples(self, tag: str, params: dict) -> dict:
        scored = self._filter(self.loader.load_pattern_monitor_scored(tag), params)
        cols = [c for c in ["date", "category", "subcategory", "row_score", "client_first_message", "dialog_text"] if c in scored.columns]
        return {"rows": scored[cols].head(int(params.get("top_n", 50))).to_dict(orient="records") if not scored.empty else []}
