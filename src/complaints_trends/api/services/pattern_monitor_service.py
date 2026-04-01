from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from ..schemas import AlertRowResponse, DailyPressureResponse, OverallStateResponse, PatternMonitorSummaryPayload, PatternMonitorSummaryResponse
from .data_loader import DataLoader
from .taxonomy_label_service import TaxonomyLabelService


class PatternMonitorService:
    def __init__(self, loader: DataLoader, feedback_service=None, calibrator_service=None, registry_service=None, labels: TaxonomyLabelService | None = None) -> None:
        self.loader = loader
        self.feedback_service = feedback_service
        self.calibrator_service = calibrator_service
        self.registry_service = registry_service
        self.labels = labels

    def _resolved_tag(self, tag: str) -> str:
        return self.loader.resolve_tag("pattern_monitor", tag)

    def _filter(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        out = df.copy()
        if out.empty:
            return out
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            if params.get("date_from"):
                out = out[out["date"] >= pd.to_datetime(params["date_from"], errors="coerce")]
            if params.get("date_to"):
                date_to = pd.to_datetime(params["date_to"], errors="coerce")
                # If UI sends plain date (YYYY-MM-DD), make the upper bound inclusive for the full day.
                if pd.notna(date_to) and "T" not in str(params["date_to"]) and " " not in str(params["date_to"]):
                    date_to = date_to + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                out = out[out["date"] <= date_to]
        if params.get("category") and "category" in out.columns:
            out = out[out["category"].isin(params["category"])]
        if params.get("upload_id"):
            if "source_upload_id" in out.columns:
                out = out[out["source_upload_id"].astype(str) == str(params["upload_id"])]
            else:
                # Some historical/scored artifacts do not carry source_upload_id.
                # In that case do not hard-drop all rows, otherwise dashboard shows
                # empty results while underlying pattern_monitor outputs contain data.
                out = out
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
    def _json_records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
        out = df.head(limit) if limit is not None else df
        records = out.to_dict(orient="records")
        return [PatternMonitorService._to_json_safe(r) for r in records]

    @staticmethod
    def _to_json_safe(value):
        if value is None:
            return None
        if value is pd.NaT:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, np.generic):
            return PatternMonitorService._to_json_safe(value.item())
        if isinstance(value, (pd.Timestamp, datetime, date, time)):
            return value.isoformat()
        if isinstance(value, pd.Timedelta):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, dict):
            return {str(k): PatternMonitorService._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [PatternMonitorService._to_json_safe(v) for v in value]
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _with_row_dialog(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        fallback_sources = ("raw_dialog", "dialog_text", "client_first_message", "text_original")
        fallback_col = next((c for c in fallback_sources if c in out.columns), None)
        if "row_dialog" not in out.columns and fallback_col is None:
            out["row_dialog"] = ""
            return out
        if "row_dialog" not in out.columns:
            out["row_dialog"] = out[fallback_col] if fallback_col else ""
        else:
            row_dialog = out["row_dialog"].fillna("").astype(str).str.strip()
            fallback = out[fallback_col].fillna("").astype(str) if fallback_col else ""
            out["row_dialog"] = row_dialog.where(row_dialog != "", fallback)
        return out

    def _with_ru_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.labels or df.empty:
            return df
        out = df.copy()
        if "category" in out.columns:
            out["category_label_ru"] = out["category"].map(lambda v: self.labels.category_label_ru(None if pd.isna(v) else str(v)))
        else:
            out["category_label_ru"] = ""
        if "subcategory" in out.columns:
            out["subcategory_label_ru"] = out.apply(
                lambda r: self.labels.subcategory_label_ru(
                    None if pd.isna(r.get("category")) else str(r.get("category")),
                    None if pd.isna(r.get("subcategory")) else str(r.get("subcategory")),
                ),
                axis=1,
            )
        else:
            out["subcategory_label_ru"] = ""
        return out

    @staticmethod
    def _alert_mask(df: pd.DataFrame) -> pd.Series:
        if "is_pattern_alert" in df.columns:
            return df["is_pattern_alert"] == True
        if "is_alert" in df.columns:
            return df["is_alert"] == True
        return pd.Series([False] * len(df), index=df.index)

    @staticmethod
    def _clamp01(value: float | None) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(max(0.0, min(1.0, float(value))))

    @staticmethod
    def _safe_last_state(state: pd.DataFrame) -> tuple[float | None, float | None]:
        if state.empty:
            return None, None
        out = state.copy()
        if "date" in out.columns:
            out = out.sort_values("date")
        last = out.iloc[-1]
        smoothed = None
        overall = None
        for c in ("smoothed_state", "smoothed", "state"):
            if c in out.columns and not pd.isna(last.get(c)):
                smoothed = float(last.get(c))
                break
        for c in ("overall_pressure", "pressure", "state"):
            if c in out.columns and not pd.isna(last.get(c)):
                overall = float(last.get(c))
                break
        return PatternMonitorService._clamp01(smoothed), PatternMonitorService._clamp01(overall)

    def _compute_alert_component(self, scored: pd.DataFrame) -> tuple[float | None, int, int]:
        if scored.empty:
            return None, 0, 0
        total_rows = int(len(scored))
        alert_rows = int(self._alert_mask(scored).sum())
        return self._clamp01(alert_rows / max(total_rows, 1)), alert_rows, total_rows

    def _compute_pressure_component(self, pressure: pd.DataFrame, period_days: int) -> tuple[float | None, int]:
        if pressure.empty:
            return None, 0
        out = pressure.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out[out["date"].notna()]
        if out.empty:
            return None, 0

        metric_col = next((c for c in ("pressure_score", "pressure", "overall_pressure", "severity") if c in out.columns), None)
        if metric_col is not None:
            pressure_days = int(out.loc[pd.to_numeric(out[metric_col], errors="coerce").fillna(0.0) > 0, "date"].nunique()) if "date" in out.columns else int((pd.to_numeric(out[metric_col], errors="coerce").fillna(0.0) > 0).sum())
        else:
            pressure_days = int(out["date"].nunique()) if "date" in out.columns else int(len(out))
        component = self._clamp01(pressure_days / max(period_days, 1))
        return component, pressure_days

    @staticmethod
    def _risk_label(score: float | None) -> tuple[str, str, str]:
        if score is None:
            return "unavailable", "Недоступно", "neutral"
        if score < 0.30:
            return "low", "Низкий", "success"
        if score < 0.60:
            return "medium", "Средний", "warning"
        return "high", "Высокий", "danger"

    def _compute_pattern_risk(self, state_component: float | None, alert_component: float | None, pressure_component: float | None) -> tuple[float | None, str]:
        if state_component is not None and alert_component is not None and pressure_component is not None:
            score = 0.60 * state_component + 0.25 * alert_component + 0.15 * pressure_component
            return self._clamp01(score), "full"
        if state_component is not None:
            return self._clamp01(state_component), "state_only"

        components = []
        weights = []
        if alert_component is not None:
            components.append(alert_component)
            weights.append(0.25)
        if pressure_component is not None:
            components.append(pressure_component)
            weights.append(0.15)
        if components and weights:
            total_weight = sum(weights)
            weighted = sum(c * w for c, w in zip(components, weights)) / total_weight
            return self._clamp01(weighted), "alerts_only"
        return None, "unavailable"

    @staticmethod
    def _period_days(params: dict, scored: pd.DataFrame, pressure: pd.DataFrame, state: pd.DataFrame) -> int:
        d_from = pd.to_datetime(params.get("date_from"), errors="coerce") if params.get("date_from") else None
        d_to = pd.to_datetime(params.get("date_to"), errors="coerce") if params.get("date_to") else None
        if d_from is not None and d_to is not None and not pd.isna(d_from) and not pd.isna(d_to):
            return max((d_to.normalize() - d_from.normalize()).days + 1, 1)

        dates = set()
        for df in (scored, pressure, state):
            if not df.empty and "date" in df.columns:
                vals = pd.to_datetime(df["date"], errors="coerce").dropna()
                dates.update(v.date().isoformat() for v in vals)
        return max(len(dates), 1)

    def summary(self, tag: str, params: dict) -> PatternMonitorSummaryResponse:
        resolved = self._resolved_tag(tag)

        with ThreadPoolExecutor(max_workers=3) as ex:
            f_scored = ex.submit(self.loader.load_pattern_monitor_scored, resolved)
            f_pressure = ex.submit(self.loader.load_pattern_monitor_pressure, resolved)
            f_state = ex.submit(self.loader.load_pattern_monitor_state, resolved)
            try:
                scored = self._filter(f_scored.result(), params)
            except Exception:
                scored = pd.DataFrame()
            try:
                pressure = self._filter(f_pressure.result(), params)
            except Exception:
                pressure = pd.DataFrame()
            try:
                state = self._filter(f_state.result(), params)
            except Exception:
                state = pd.DataFrame()

        state_smoothed, state_overall = self._safe_last_state(state)
        state_component = state_smoothed if state_smoothed is not None else state_overall
        alert_component, alert_rows, scored_rows = self._compute_alert_component(scored)
        period_days = self._period_days(params, scored, pressure, state)
        pressure_component, pressure_days = self._compute_pressure_component(pressure, period_days)

        risk_score, calc_mode = self._compute_pattern_risk(state_component, alert_component, pressure_component)
        risk_label, display_label, risk_status = self._risk_label(risk_score)

        return PatternMonitorSummaryResponse(
            tag=resolved,
            summary=PatternMonitorSummaryPayload(
                scored_rows=int(scored_rows),
                alert_rows=int(alert_rows),
                pressure_days=int(pressure_days),
                latest_overall_pressure=state_overall,
                latest_smoothed_state=state_smoothed,
                pattern_risk_score=risk_score,
                pattern_risk_label=risk_label,
                pattern_risk_display_label=display_label,
                pattern_risk_status=risk_status,
                pattern_risk_calc_mode=calc_mode,
            ),
        )

    def alerts(self, tag: str, params: dict) -> AlertRowResponse:
        resolved = self._resolved_tag(tag)
        scored = self._filter(self.loader.load_pattern_monitor_scored(resolved), params)
        had_alert_rows = False
        if not scored.empty:
            alert_mask = self._alert_mask(scored)
            had_alert_rows = bool(alert_mask.any())
            if had_alert_rows:
                scored = scored[alert_mask]
            else:
                scored = scored.copy()
                if "is_pattern_alert" not in scored.columns:
                    scored["is_pattern_alert"] = False

        requested_mode = params.get("scoring_mode") or "base"
        effective_mode = "base"
        reranker_available = False
        if self.calibrator_service is not None and not scored.empty:
            pipeline = self.calibrator_service.scoring_pipeline()
            scored, effective_mode, reranker_available = pipeline.score_rows(scored, requested_mode=requested_mode)

        if not scored.empty:
            if self.feedback_service is not None:
                scored = scored.copy()
                if "row_id" not in scored.columns:
                    scored["row_id"] = [self.feedback_service.build_row_id(r) for r in scored.to_dict(orient="records")]
                fb_rows = self.feedback_service.list_feedback({"pattern_tag": resolved, "limit": 100000}) if self.feedback_service else []
                verdict_by_row = {r["row_id"]: r["verdict"] for r in fb_rows}
                scored["feedback_verdict"] = scored["row_id"].map(verdict_by_row)
            sort_col = "rerank_score" if effective_mode == "reranked" and "rerank_score" in scored.columns else ("calibrated_score" if effective_mode == "calibrated" and "calibrated_score" in scored.columns else ("pattern_like_score" if "pattern_like_score" in scored.columns else "row_score"))
            if sort_col in scored.columns:
                scored = scored.sort_values(sort_col, ascending=False)
            if not had_alert_rows:
                scored["no_alerts_in_selection"] = True
            scored = self._with_row_dialog(scored)
            scored = self._with_ru_labels(scored)

        active_version = self.registry_service.get_active() if self.registry_service else None
        rows = self._json_records(scored, int(params.get("top_n", 200)))
        return AlertRowResponse(
            rows=rows,
            scoring_mode_requested=requested_mode,
            scoring_mode_effective=effective_mode,
            reranker_available=reranker_available,
            active_calibrator_version=(active_version.get("version_id") if active_version else None),
        )

    def pressure(self, tag: str, params: dict) -> DailyPressureResponse:
        resolved = self._resolved_tag(tag)
        data = self._filter(self.loader.load_pattern_monitor_pressure(resolved), params)
        return DailyPressureResponse(rows=self._json_records(data))

    def state(self, tag: str, params: dict) -> OverallStateResponse:
        resolved = self._resolved_tag(tag)
        data = self._filter(self.loader.load_pattern_monitor_state(resolved), params)
        return OverallStateResponse(rows=self._json_records(data))

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
            response = self._with_ru_labels(response)

        return {
            "tag": resolved,
            "rows": self._json_records(response, int(params.get("top_n", 200))) if not response.empty else [],
        }

    def run_output_rows(self, tag: str, top_n: int = 300) -> AlertRowResponse:
        resolved = self._resolved_tag(tag)
        scored = self.loader.load_pattern_monitor_scored(resolved)
        if not scored.empty:
            mask = self._alert_mask(scored)
            if bool(mask.any()):
                scored = scored[mask]
        if not scored.empty:
            sort_col = "pattern_like_score" if "pattern_like_score" in scored.columns else ("row_score" if "row_score" in scored.columns else None)
            if sort_col:
                scored = scored.sort_values(sort_col, ascending=False)
            scored = self._with_row_dialog(scored)
            scored = self._with_ru_labels(scored)
        return AlertRowResponse(rows=self._json_records(scored, top_n))

    def run_output_rows_by_path(self, path: str, top_n: int = 300) -> AlertRowResponse:
        p = Path(path)
        if not p.exists() or p.suffix != ".parquet":
            return AlertRowResponse(rows=[])
        scored = self.loader.read_parquet(p)
        if not scored.empty:
            sort_col = "pattern_like_score" if "pattern_like_score" in scored.columns else ("row_score" if "row_score" in scored.columns else None)
            if sort_col:
                scored = scored.sort_values(sort_col, ascending=False)
            scored = self._with_row_dialog(scored)
            scored = self._with_ru_labels(scored)
        return AlertRowResponse(rows=self._json_records(scored, top_n))
