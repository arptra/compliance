from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ...config import ProjectConfig
from ...pattern_monitor import run_pattern_monitor
from ...prepare_dataset import prepare_dataset
from ..schemas import (
    PatternMonitorPresetPayload,
    PatternMonitorPresetResponse,
    PreparationJobSummary,
    PreparationPreviewResponse,
    PreparationRunResponse,
    PreparationUploadResponse,
)


class PreparationService:
    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self.base_dir = Path(cfg.analysis.pattern_monitoring.interim_dir) / "preparation_jobs"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.base_dir / "jobs_registry.json"

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _load_registry(self) -> list[dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        return pd.read_json(self.registry_path).to_dict(orient="records")

    def _save_registry(self, rows: list[dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_json(self.registry_path, orient="records", force_ascii=False, date_format="iso")

    def _upsert_job(self, job: dict[str, Any]) -> None:
        rows = self._load_registry()
        rows = [r for r in rows if str(r.get("upload_id")) != str(job.get("upload_id"))]
        rows.append(job)
        rows = sorted(rows, key=lambda r: str(r.get("uploaded_at", "")), reverse=True)
        self._save_registry(rows)

    def create_upload_job(self, filename: str, content: bytes) -> PreparationUploadResponse:
        upload_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        d = self.base_dir / upload_id
        d.mkdir(parents=True, exist_ok=True)
        ext = Path(filename).suffix or ".xlsx"
        stored = d / f"source{ext}"
        stored.write_bytes(content)
        uploaded_at = self._now()

        preview = self._detect_preview(stored)
        job = {
            "upload_id": upload_id,
            "original_filename": filename,
            "stored_path": str(stored),
            "uploaded_at": uploaded_at.isoformat(),
            "status": "uploaded",
            "started_at": None,
            "finished_at": None,
            "error_message": None,
            "rows_total": preview.get("rows_total", 0),
            "prepared_rows": 0,
            "complaints_rows": 0,
            "date_min": preview.get("date_min"),
            "date_max": preview.get("date_max"),
            "output_prepared_parquet": None,
            "merged_into_main": False,
            "available_for_pattern_monitor": False,
            "available_columns": preview.get("available_columns", []),
        }
        self._upsert_job(job)
        return PreparationUploadResponse(upload_id=upload_id, filename=filename, uploaded_at=uploaded_at, status="uploaded")

    def list_preparation_jobs(self, status: str | None = None, limit: int = 30) -> list[PreparationJobSummary]:
        rows = self._load_registry()
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return [PreparationJobSummary(**self._normalize_job(r)) for r in rows[: max(limit, 1)]]

    def get_preparation_job(self, upload_id: str) -> PreparationJobSummary | None:
        for r in self._load_registry():
            if str(r.get("upload_id")) == upload_id:
                return PreparationJobSummary(**self._normalize_job(r))
        return None

    def run_preparation_job(self, upload_id: str) -> PreparationRunResponse:
        job = self.get_preparation_job(upload_id)
        if job is None:
            return PreparationRunResponse(upload_id=upload_id, status="failed", error_message="upload_not_found")

        d = self.base_dir / upload_id
        logs_path = d / "logs.txt"
        source_path = Path(job.stored_path)
        prepared_path = d / "prepared.parquet"

        current = job.model_dump()
        current.update({"status": "running", "started_at": self._now().isoformat(), "error_message": None})
        self._upsert_job(current)

        try:
            cfg2 = self.cfg.model_copy(deep=True)
            cfg2.input.input_dir = str(source_path.parent)
            cfg2.input.file_names = [source_path.name]
            cfg2.input.file_glob = source_path.name
            suffix = source_path.suffix.lower()
            if suffix == ".csv":
                cfg2.input.file_format = "csv"
            elif suffix in {".xlsx", ".xls"}:
                cfg2.input.file_format = "excel"
            else:
                cfg2.input.file_format = "auto"
            cfg2.prepare.output_parquet = str(prepared_path)

            df = prepare_dataset(cfg2, pilot=False, llm_mock=not cfg2.llm.enabled)
            if df.empty:
                raise ValueError("prepared dataframe is empty")

            date_series = pd.to_datetime(df.get("event_time"), errors="coerce") if "event_time" in df.columns else pd.Series(dtype="datetime64[ns]")
            date_min = None if date_series.dropna().empty else date_series.min().date().isoformat()
            date_max = None if date_series.dropna().empty else date_series.max().date().isoformat()

            df = df.copy()
            df["source_upload_id"] = upload_id
            df["source_filename"] = job.original_filename
            df["source_uploaded_at"] = job.uploaded_at.isoformat()
            df["source_date_min"] = date_min
            df["source_date_max"] = date_max
            df["source_type"] = "uploaded_preparation"
            df.to_parquet(prepared_path, index=False)

            merged_rows = self.merge_prepared_upload_into_main(df, upload_id)
            complaints_rows = int(df["is_complaint_llm"].fillna(False).sum()) if "is_complaint_llm" in df.columns else 0
            pattern_tag = upload_id

            monitor_error = None
            try:
                run_pattern_monitor(
                    self.cfg,
                    tag=pattern_tag,
                    label_source="llm",
                    fit_tag="latest",
                    date_from=date_min,
                    date_to=date_max,
                )
            except Exception as e:
                monitor_error = f"pattern_monitor_autorun_failed: {e}"

            current = self.get_preparation_job(upload_id).model_dump() if self.get_preparation_job(upload_id) else current
            current.update(
                {
                    "status": "succeeded",
                    "finished_at": self._now().isoformat(),
                    "rows_total": int(len(df)),
                    "prepared_rows": int(len(df)),
                    "complaints_rows": complaints_rows,
                    "date_min": date_min,
                    "date_max": date_max,
                    "output_prepared_parquet": str(prepared_path),
                    "merged_into_main": True,
                    "available_for_pattern_monitor": True,
                    "pattern_monitor_tag": pattern_tag,
                    "error_message": None,
                }
            )
            self._upsert_job(current)
            logs_text = f"prepared_rows={len(df)}\nmerged_main_rows={merged_rows}\npattern_monitor_tag={pattern_tag}\n"
            if monitor_error:
                logs_text += f"{monitor_error}\n"
            logs_path.write_text(logs_text, encoding="utf-8")
            return PreparationRunResponse(upload_id=upload_id, status="succeeded")
        except Exception as e:
            current = self.get_preparation_job(upload_id).model_dump() if self.get_preparation_job(upload_id) else current
            current.update({"status": "failed", "finished_at": self._now().isoformat(), "error_message": str(e), "available_for_pattern_monitor": False})
            self._upsert_job(current)
            logs_path.write_text(str(e), encoding="utf-8")
            return PreparationRunResponse(upload_id=upload_id, status="failed", error_message=str(e))

    def merge_prepared_upload_into_main(self, prepared_df: pd.DataFrame, upload_id: str) -> int:
        main_path = Path(self.cfg.prepare.output_parquet)
        existing = pd.DataFrame()
        if main_path.exists():
            try:
                existing = pd.read_parquet(main_path)
            except Exception:
                existing = pd.DataFrame()
        if not existing.empty and "source_upload_id" in existing.columns:
            existing = existing[existing["source_upload_id"] != upload_id].copy()
        merged = pd.concat([existing, prepared_df], ignore_index=True)
        main_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(main_path, index=False)
        return int(len(merged))

    def get_preview(self, upload_id: str) -> PreparationPreviewResponse | None:
        job = self.get_preparation_job(upload_id)
        if not job:
            return None
        payload = self._normalize_job(job.model_dump())
        return PreparationPreviewResponse(
            upload_id=upload_id,
            filename=payload["original_filename"],
            status=payload["status"],
            rows_total=payload.get("rows_total", 0),
            date_min=payload.get("date_min"),
            date_max=payload.get("date_max"),
            available_columns=list(payload.get("available_columns", [])),
        )

    def build_pattern_monitor_preset(self, upload_id: str) -> PatternMonitorPresetResponse:
        job = self.get_preparation_job(upload_id)
        if job is None:
            return PatternMonitorPresetResponse(allowed=False, reason="upload_not_found", pattern_monitor_preset=None)
        if job.status != "succeeded" or not job.available_for_pattern_monitor:
            return PatternMonitorPresetResponse(allowed=False, reason="preparation_not_finished", pattern_monitor_preset=None)
        target_month = (job.date_max or job.date_min or "")[:7] or None
        if target_month is None:
            return PatternMonitorPresetResponse(allowed=False, reason="upload_month_not_detected", pattern_monitor_preset=None)
        monitor_tag = "mig_2025_q4"
        try:
            run_pattern_monitor(
                self.cfg,
                tag=monitor_tag,
                label_source="llm",
                fit_tag="latest",
                month=target_month,
                force_materialize=True,
            )
        except Exception as e:
            return PatternMonitorPresetResponse(allowed=False, reason=f"pattern_monitor_run_failed: {e}", pattern_monitor_preset=None)
        return PatternMonitorPresetResponse(
            allowed=True,
            reason=None,
            pattern_monitor_preset=PatternMonitorPresetPayload(
                date_from=job.date_min,
                date_to=job.date_max,
                month=target_month,
                upload_id=upload_id,
                pattern_tag=monitor_tag,
                label_source="llm",
                source_filename=job.original_filename,
            ),
        )

    def _detect_preview(self, source_path: Path) -> dict[str, Any]:
        try:
            if source_path.suffix.lower() == ".csv":
                df = pd.read_csv(source_path)
            else:
                df = pd.read_excel(source_path)
            if df.empty:
                return {"rows_total": 0, "date_min": None, "date_max": None, "available_columns": list(df.columns)}
            dt_col = self.cfg.input.datetime_column if self.cfg.input.datetime_column in df.columns else None
            if dt_col is None:
                for c in ("created_at", "event_time", "date"):
                    if c in df.columns:
                        dt_col = c
                        break
            dmin = dmax = None
            if dt_col:
                ds = pd.to_datetime(df[dt_col], errors="coerce").dropna()
                if not ds.empty:
                    dmin = ds.min().date().isoformat()
                    dmax = ds.max().date().isoformat()
            return {"rows_total": int(len(df)), "date_min": dmin, "date_max": dmax, "available_columns": list(df.columns)}
        except Exception:
            return {"rows_total": 0, "date_min": None, "date_max": None, "available_columns": []}

    @staticmethod
    def _normalize_job(payload: dict[str, Any]) -> dict[str, Any]:
        out = dict(payload)
        for k, v in list(out.items()):
            if isinstance(v, float) and pd.isna(v):
                out[k] = None
        for dt_key in ("uploaded_at", "started_at", "finished_at"):
            if out.get(dt_key) is not None and not isinstance(out.get(dt_key), str):
                parsed = pd.to_datetime(out[dt_key], errors="coerce")
                out[dt_key] = parsed.isoformat() if not pd.isna(parsed) else None
        return out
