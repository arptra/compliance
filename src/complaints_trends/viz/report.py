from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..infer_month import _enforce_subcategory_taxonomy, _load_effective_taxonomy, _predict_subcategories_by_category
from ..reports.render import write_md
from ..text_cleaning import clean_for_model, load_tokens
from .plots import (
    plot_delta_bars_or_waterfall,
    plot_heatmap_dow_hour,
    plot_pareto,
    plot_share_lines,
    plot_stacked_area,
)
from .state import VizPaths, build_viz_state, save_meta


def materialize_predictions(cfg: ProjectConfig, in_parquet: str | Path, out_parquet: str | Path) -> Path:
    df = pd.read_parquet(in_parquet)
    text_col = cfg.training.text_field
    if text_col not in df.columns:
        text_col = "client_first_message"

    deny = load_tokens(cfg.files.deny_tokens_path) | load_tokens(cfg.files.extra_stopwords_path)
    text = df[text_col].fillna("").astype(str).apply(lambda x: clean_for_model(x, deny))

    mdir = Path(cfg.training.model_dir)
    vec = joblib.load(mdir / "vectorizers.joblib")
    complaint_model = joblib.load(mdir / "complaint_model.joblib")
    cat_model = joblib.load(mdir / "category_model.joblib")
    enc = joblib.load(mdir / "label_encoder.joblib")
    subcat_model_path = mdir / "subcategory_model.joblib"
    subcat_enc_path = mdir / "subcategory_label_encoder.joblib"
    subcat_by_cat_path = mdir / "subcategory_models_by_category.joblib"
    subcat_model = joblib.load(subcat_model_path) if subcat_model_path.exists() else None
    subcat_enc = joblib.load(subcat_enc_path) if subcat_enc_path.exists() else None
    subcat_models_by_category = joblib.load(subcat_by_cat_path) if subcat_by_cat_path.exists() else None

    x = vec.transform(text)
    if hasattr(complaint_model, "predict_proba"):
        score = complaint_model.predict_proba(x)[:, 1]
    else:
        raw = complaint_model.decision_function(x)
        score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    is_complaint = score >= cfg.training.complaint_threshold

    cat_pred = np.array(["OTHER"] * len(df), dtype=object)
    if is_complaint.any() and cat_model is not None:
        cat_pred[is_complaint] = enc.inverse_transform(cat_model.predict(x[is_complaint]))

    subcat_pred = _predict_subcategories_by_category(
        x=x,
        idx_mask=is_complaint,
        categories=cat_pred,
        fallback_subcat_model=subcat_model,
        fallback_subcat_enc=subcat_enc,
        models_by_category=subcat_models_by_category,
    )
    taxonomy = _load_effective_taxonomy(cfg)
    subcat_pred = _enforce_subcategory_taxonomy(cat_pred, subcat_pred, taxonomy)

    out = pd.DataFrame(
        {
            "row_id": df.get("row_id", pd.Series([f"row_{i}" for i in range(len(df))])),
            "event_time": pd.to_datetime(df.get("event_time"), errors="coerce"),
            "month": df.get("month", pd.to_datetime(df.get("event_time"), errors="coerce").dt.strftime("%Y-%m")),
            "is_complaint_pred": is_complaint,
            "complaint_score": score,
            "category_pred": cat_pred,
            "subcategory_pred": subcat_pred,
        }
    )
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path


def _prepare_viz_frame(df: pd.DataFrame, label_source: str) -> pd.DataFrame:
    if label_source == "pred":
        out = df.rename(
            columns={
                "is_complaint_pred": "is_complaint_flag",
                "category_pred": "category",
                "subcategory_pred": "subcategory",
            }
        ).copy()
    else:
        out = df.rename(
            columns={
                "is_complaint_llm": "is_complaint_flag",
                "complaint_category_llm": "category",
                "complaint_subcategory_llm": "subcategory",
            }
        ).copy()
        if "subcategory" not in out.columns:
            out["subcategory"] = "UNKNOWN"
    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    out["category"] = out["category"].fillna("OTHER").astype(str)
    out["subcategory"] = out["subcategory"].fillna("UNKNOWN").astype(str)
    out["is_complaint_flag"] = out["is_complaint_flag"].fillna(False).astype(bool)
    return out


def _merge_with_infer_month_predictions(frame: pd.DataFrame, new_month: str | None, label_source: str) -> tuple[pd.DataFrame, dict]:
    info = {"infer_month_loaded": False, "infer_month_rows": 0, "infer_month_path": ""}
    if label_source != "pred" or not new_month:
        return frame, info

    infer_path = Path("data/interim") / f"month_{new_month}.parquet"
    if not infer_path.exists():
        return frame, info

    month_df = pd.read_parquet(infer_path)
    month_viz = _prepare_viz_frame(month_df, "pred")
    month_viz["month"] = month_viz.get("month", pd.to_datetime(month_viz.get("event_time"), errors="coerce").dt.strftime("%Y-%m"))

    merged = pd.concat([frame, month_viz], ignore_index=True)
    dedup_keys = [k for k in ["row_id", "month"] if k in merged.columns]
    if dedup_keys:
        order_cols = [k for k in ["event_time", "month"] if k in merged.columns]
        merged = merged.sort_values(order_cols).drop_duplicates(subset=dedup_keys, keep="last")

    info = {
        "infer_month_loaded": True,
        "infer_month_rows": int(len(month_viz)),
        "infer_month_path": str(infer_path),
    }
    return merged, info


def build_visual_report(
    cfg: ProjectConfig,
    tag: str,
    label_source: str,
    date_from: str | None,
    date_to: str | None,
    baseline_range: str | None,
    new_month: str | None,
    top_n: int,
    freq: str,
) -> tuple[Path, Path]:
    paths = VizPaths(tag)
    src_path = Path("data/interim/all_predicted.parquet") if label_source == "pred" else Path(cfg.prepare.output_parquet)

    frame = pd.read_parquet(src_path)
    frame = _prepare_viz_frame(frame, label_source)
    frame, infer_info = _merge_with_infer_month_predictions(frame, new_month, label_source)

    if date_from:
        frame = frame[frame["event_time"] >= pd.to_datetime(date_from)]
    if date_to:
        frame = frame[frame["event_time"] <= pd.to_datetime(date_to)]

    state = build_viz_state(frame, label_source=label_source, freq=freq, top_n=top_n)
    paths.state_parquet.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(paths.state_parquet, index=False)

    complaints_state = state[state["is_complaint_flag"] == True].copy()

    p1 = plot_stacked_area(complaints_state, paths.report_dir / "stacked_area_counts.png", "Complaints by category over time")
    p2 = plot_share_lines(complaints_state, paths.report_dir / "share_lines.png", "Category share over time")
    p3 = plot_pareto(complaints_state, paths.report_dir / "pareto_categories.png", "Pareto categories over period")
    p4 = plot_heatmap_dow_hour(frame[frame["is_complaint_flag"] == True], paths.report_dir / "heatmap_dow_hour.png", "Complaints heatmap weekday x hour")

    delta_df = pd.DataFrame(columns=["category", "delta_pp", "delta_count"])
    if baseline_range and new_month:
        start, end = baseline_range.split("..")
        base = frame[(frame["month"] >= start) & (frame["month"] <= end) & (frame["is_complaint_flag"] == True)]
        new = frame[(frame["month"] == new_month) & (frame["is_complaint_flag"] == True)]
        b = base["category"].value_counts(normalize=True)
        n = new["category"].value_counts(normalize=True)
        merged = pd.DataFrame({"baseline": b, "new": n}).fillna(0)
        merged["delta_pp"] = (merged["new"] - merged["baseline"]) * 100
        merged["delta_count"] = new["category"].value_counts().reindex(merged.index).fillna(0) - base["category"].value_counts().reindex(merged.index).fillna(0)
        delta_df = merged.reset_index().rename(columns={"index": "category"})
    p5 = plot_delta_bars_or_waterfall(delta_df, paths.report_dir / "delta_bars.png", "Category contribution delta")

    kpi_total = int(len(frame))
    kpi_complaints = int(frame["is_complaint_flag"].sum()) if len(frame) else 0
    kpi_share = float(kpi_complaints / kpi_total) if kpi_total else 0.0
    top_table = complaints_state.groupby("category")["metric_count"].sum().sort_values(ascending=False).head(15)

    infer_month_summary = pd.DataFrame(columns=["category", "count", "share"])
    if new_month:
        nm = frame[(frame.get("month", "") == new_month) & (frame["is_complaint_flag"] == True)]
        if len(nm):
            cnt = nm["category"].value_counts()
            infer_month_summary = pd.DataFrame(
                {
                    "category": cnt.index.astype(str),
                    "count": cnt.values.astype(int),
                    "share": (cnt.values / max(1, int(cnt.sum()))),
                }
            )

    md = [
        f"# Visual report: {tag}",
        "",
        f"- label_source: `{label_source}`",
        f"- rows_total: **{kpi_total}**",
        f"- complaints_count: **{kpi_complaints}**",
        f"- complaints_share: **{kpi_share:.2%}**",
        "",
        "## Charts",
        f"![stacked]({Path(p1).name})",
        f"![share]({Path(p2).name})",
        f"![pareto]({Path(p3).name})",
        f"![heatmap]({Path(p4).name})",
        f"![delta]({Path(p5).name})",
        "",
        "## Top categories",
        "| category | count |",
        "|---|---:|",
    ]
    for cat, cnt in top_table.items():
        md.append(f"| {cat} | {int(cnt)} |")

    md.extend(["", "## Интерпретация infer-month", ""])
    if new_month:
        md.append(f"- new_month: **{new_month}**")
        md.append(f"- infer-month parquet подключен: **{infer_info['infer_month_loaded']}**")
        if infer_info["infer_month_loaded"]:
            md.append(f"- путь: `{infer_info['infer_month_path']}`")
            md.append(f"- строк из infer-month: **{infer_info['infer_month_rows']}**")
        if len(infer_month_summary):
            md.append("")
            md.append("### Топ категорий в infer-month")
            md.append("| category | count | share_of_complaints |")
            md.append("|---|---:|---:|")
            for _, r in infer_month_summary.head(15).iterrows():
                md.append(f"| {r['category']} | {int(r['count'])} | {float(r['share']):.2%} |")
            md.append("")
            md.append("Интерпретация: count = абсолютное число жалоб категории в новом месяце; share_of_complaints = доля среди всех жалоб нового месяца.")
        else:
            md.append("- В данных отчёта не найдено жалоб для new_month (или отсутствует month-поле).")
    else:
        md.append("- Параметр --new-month не задан; отдельная интерпретация по infer-month не построена.")

    report_path = paths.report_dir / "report.md"
    write_md(str(report_path), "\n".join(md))

    save_meta(
        paths.meta_json,
        {
            "tag": tag,
            "label_source": label_source,
            "freq": freq,
            "top_n": top_n,
            "date_from": date_from,
            "date_to": date_to,
            "baseline_range": baseline_range,
            "new_month": new_month,
            "source_path": str(src_path),
            "state_path": str(paths.state_parquet),
            "report_path": str(report_path),
            "infer_month_loaded": infer_info["infer_month_loaded"],
            "infer_month_rows": infer_info["infer_month_rows"],
            "infer_month_path": infer_info["infer_month_path"],
        },
    )
    return report_path, paths.state_parquet
