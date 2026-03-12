from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from .config import ProjectConfig
from .pattern_common import (
    aggregate_daily_pressure,
    build_text_clean,
    get_label_columns,
    render_pattern_monitor_report,
    score_target_rows,
)
from .pattern_paths import PatternFitPaths, PatternMonitorPaths
from .viz.report import materialize_predictions

try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE as _OPENPYXL_ILLEGAL_RE
except Exception:  # pragma: no cover
    _OPENPYXL_ILLEGAL_RE = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")


def _sanitize_for_excel(df: pd.DataFrame, max_len: int = 32767) -> pd.DataFrame:
    out = df.copy()
    cols = out.select_dtypes(include=["object", "string"]).columns
    for col in cols:
        s = out[col]
        mask = s.map(lambda v: isinstance(v, str))
        if not bool(mask.any()):
            continue
        v = s.loc[mask].astype(str).str.replace(_OPENPYXL_ILLEGAL_RE, "", regex=True).str.slice(0, max_len)
        out.loc[mask, col] = v
    return out


def _to_excel_sheet_safe(df: pd.DataFrame, writer: pd.ExcelWriter, sheet_name: str, index: bool = False) -> None:
    _sanitize_for_excel(df).to_excel(writer, sheet_name=sheet_name, index=index)


def run_pattern_monitor(
    cfg: ProjectConfig,
    tag: str,
    label_source: str,
    date_from: str | None = None,
    date_to: str | None = None,
    month: str | None = None,
    force_materialize: bool = False,
) -> tuple[Path, Path, Path]:
    pm = cfg.analysis.pattern_monitoring
    fit_paths = PatternFitPaths(tag=tag, interim_dir=pm.interim_dir, exports_dir=pm.exports_dir, reports_dir=pm.reports_dir)
    if not fit_paths.fit_bundle.exists():
        raise FileNotFoundError(f"fit bundle not found: {fit_paths.fit_bundle}")
    fit_bundle = joblib.load(fit_paths.fit_bundle)

    out = PatternMonitorPaths(tag=tag, interim_dir=pm.interim_dir, exports_dir=pm.exports_dir, reports_dir=pm.reports_dir)
    out.root.mkdir(parents=True, exist_ok=True)
    out.report.parent.mkdir(parents=True, exist_ok=True)
    out.export.parent.mkdir(parents=True, exist_ok=True)

    if month:
        src = Path(pm.interim_dir) / f"month_{month}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"month parquet not found: {src}")
        df = pd.read_parquet(src)
    else:
        if label_source == "pred":
            pred_path = Path(pm.interim_dir) / "all_predicted.parquet"
            if force_materialize or not pred_path.exists():
                materialize_predictions(cfg, cfg.prepare.output_parquet, pred_path)
            df = pd.read_parquet(pred_path)
            prepared_full = pd.read_parquet(cfg.prepare.output_parquet)
            keep_cols = [c for c in ["row_id", "event_time", "month", pm.text_field, "client_first_message"] if c in prepared_full.columns]
            prepared = prepared_full[keep_cols]
            if "row_id" in df.columns and "row_id" in prepared.columns:
                df = df.merge(prepared.drop_duplicates("row_id"), on="row_id", how="left", suffixes=("", "_src"))
        else:
            df = pd.read_parquet(cfg.prepare.output_parquet)
        if date_from:
            df = df[pd.to_datetime(df.get("event_time"), errors="coerce") >= pd.to_datetime(date_from)]
        if date_to:
            to_ts = pd.to_datetime(date_to)
            if len(str(date_to)) <= 10:
                to_ts = to_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[pd.to_datetime(df.get("event_time"), errors="coerce") <= to_ts]

    is_complaint, category = get_label_columns(df, label_source)
    df = df.copy()
    df["is_complaint"] = is_complaint
    df["category"] = category
    if pm.complaints_only:
        df = df[df["is_complaint"] == True]
    if not pm.include_other_category:
        df = df[df["category"] != "OTHER"]
    if df.empty:
        raise ValueError("No rows selected for pattern-monitor")

    df["text_original"] = df.get(pm.text_field, df.get("client_first_message", "")).fillna("").astype(str)
    df["text_clean"] = build_text_clean(df, cfg, pm.text_field, pm.use_first_message_only, pm.strip_system_speakers)

    scored = score_target_rows(df, fit_bundle, pm)
    scored.to_parquet(out.root / "scored_rows.parquet", index=False)

    prev_state = None
    hist_path = out.root / "overall_daily_state.parquet"
    if hist_path.exists() and (date_from and date_to and date_from == date_to):
        prev = pd.read_parquet(hist_path)
        if len(prev):
            prev_state = float(prev.sort_values("date").iloc[-1]["smoothed_state"])

    cat_daily, overall = aggregate_daily_pressure(scored, pm, previous_state=prev_state)
    cat_daily.to_parquet(out.root / "category_daily_pressure.parquet", index=False)
    overall.to_parquet(out.root / "overall_daily_state.parquet", index=False)

    monitor_meta = {
        "created_at": datetime.utcnow().isoformat(),
        "tag": tag,
        "label_source": label_source,
        "date_from": date_from,
        "date_to": date_to,
        "month": month,
        "rows": int(len(df)),
    }
    (out.root / "monitor_meta.json").write_text(json.dumps(monitor_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.ExcelWriter(out.export) as writer:
        _to_excel_sheet_safe(scored, writer, sheet_name="scored_rows", index=False)
        _to_excel_sheet_safe(cat_daily, writer, sheet_name="category_daily_pressure", index=False)
        _to_excel_sheet_safe(overall, writer, sheet_name="overall_daily_state", index=False)
        _to_excel_sheet_safe(scored[scored["is_pattern_alert"] == True].head(200), writer, sheet_name="alert_examples", index=False)

    render_pattern_monitor_report(
        out.report,
        {
            "tag": tag,
            "label_source": label_source,
            "date_from": date_from,
            "date_to": date_to,
            "month": month,
            "alerts": scored.sort_values("pattern_like_score", ascending=False).head(30).to_dict(orient="records"),
            "category_pressure": cat_daily.sort_values("category_pressure", ascending=False).head(30).to_dict(orient="records"),
            "state_rows": overall.to_dict(orient="records"),
        },
    )
    return out.root / "scored_rows.parquet", out.root / "overall_daily_state.parquet", out.report
