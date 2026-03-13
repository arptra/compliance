from __future__ import annotations

import pandas as pd

from ..schemas import AlertRowResponse, DailyPressureResponse, OverallStateResponse, PatternMonitorSummaryResponse
from .data_loader import DataLoader


class PatternMonitorService:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def _resolved_tag(self, tag: str) -> str:
        return self.loader.resolve_tag("pattern_monitor", tag)

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


    @staticmethod
    def _series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
        if col in df.columns:
            return df[col]
        return pd.Series([default] * len(df), index=df.index)

    @staticmethod
    def _alert_mask(df: pd.DataFrame) -> pd.Series:
        if "is_pattern_alert" in df.columns:
            return df["is_pattern_alert"] == True
        if "is_alert" in df.columns:
            return df["is_alert"] == True
        return pd.Series([False] * len(df), index=df.index)

    def summary(self, tag: str, params: dict) -> PatternMonitorSummaryResponse:
        resolved = self._resolved_tag(tag)
        scored = self._filter(self.loader.load_pattern_monitor_scored(resolved), params)
        pressure = self._filter(self.loader.load_pattern_monitor_pressure(resolved), params)
        alert_rows = int(self._alert_mask(scored).sum()) if not scored.empty else 0
        return PatternMonitorSummaryResponse(
            tag=resolved,
            summary={
                "scored_rows": int(len(scored)),
                "alert_rows": alert_rows,
                "pressure_days": int(len(pressure)),
            },
        )

    def alerts(self, tag: str, params: dict) -> AlertRowResponse:
        resolved = self._resolved_tag(tag)
        scored = self._filter(self.loader.load_pattern_monitor_scored(resolved), params)
        if not scored.empty:
            scored = scored[self._alert_mask(scored)]
        return AlertRowResponse(rows=scored.head(int(params.get("top_n", 200))).to_dict(orient="records"))

    def pressure(self, tag: str, params: dict) -> DailyPressureResponse:
        resolved = self._resolved_tag(tag)
        data = self._filter(self.loader.load_pattern_monitor_pressure(resolved), params)
        return DailyPressureResponse(rows=data.to_dict(orient="records"))

    def state(self, tag: str, params: dict) -> OverallStateResponse:
        resolved = self._resolved_tag(tag)
        data = self._filter(self.loader.load_pattern_monitor_state(resolved), params)
        return OverallStateResponse(rows=data.to_dict(orient="records"))

    def examples(self, tag: str, params: dict) -> dict:
        resolved = self._resolved_tag(tag)
        scored = self._filter(self.loader.load_pattern_monitor_scored(resolved), params)
        if not scored.empty:
            scored = scored[self._alert_mask(scored)]

        response = pd.DataFrame()
        if not scored.empty:
            response["date"] = self._series(scored, "date", "")
            response["category"] = self._series(scored, "category", "UNKNOWN").fillna("UNKNOWN").astype(str)
            response["subcategory"] = self._series(scored, "subcategory", "UNKNOWN").fillna("UNKNOWN").astype(str)
            if "pattern_like_score" in scored.columns:
                response["score"] = scored["pattern_like_score"]
            elif "row_score" in scored.columns:
                response["score"] = scored["row_score"]
            else:
                response["score"] = self._series(scored, "score", "")
            for field in ("row_dialog", "raw_dialog", "dialog_text", "client_first_message"):
                if field in scored.columns:
                    response["row_dialog"] = scored[field].fillna("").astype(str)
                    break
            if "row_dialog" not in response.columns:
                response["row_dialog"] = self._series(scored, "row_dialog", "").astype(str)

        return {
            "tag": resolved,
            "rows": response.head(int(params.get("top_n", 200))).to_dict(orient="records") if not response.empty else [],
        }
