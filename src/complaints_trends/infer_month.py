from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProjectConfig
from .extract_client_first import extract_client_first_message
from .io_excel import load_excel_with_month
from .reports.render import render_template
from .text_cleaning import clean_for_model, load_tokens
from .taxonomy import load_taxonomy





def _enforce_subcategory_taxonomy(categories: np.ndarray, subcategories: np.ndarray, taxonomy: dict) -> np.ndarray:
    allowed = taxonomy.get("subcategories_by_category", {})
    out = np.array(subcategories, dtype=object).copy()
    for i, (cat, sub) in enumerate(zip(categories.tolist(), out.tolist())):
        if sub in {"NOT_COMPLAINT", "UNKNOWN"}:
            continue
        if sub not in set(allowed.get(str(cat), [])):
            out[i] = "UNKNOWN"
    return out


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = [c for c in out.columns if out[c].dtype == "object"]
    for c in obj_cols:
        out[c] = out[c].apply(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return out

def _infer_dialog_column(df: pd.DataFrame, cfg: ProjectConfig) -> str:
    candidates: list[str] = []
    if cfg.input.dialog_columns:
        candidates.extend(cfg.input.dialog_columns)
    if cfg.input.dialog_column:
        candidates.append(cfg.input.dialog_column)
    for c in candidates:
        if c in df.columns:
            return c
    for c in ("dialog_text", "call_text", "comment_text", "summary_text"):
        if c in df.columns:
            return c
    raise KeyError("No dialog column found for infer-month: configure input.dialog_column/dialog_columns")

def infer_month(cfg: ProjectConfig, excel_path: str, month: str) -> pd.DataFrame:
    df = load_excel_with_month(Path(excel_path), cfg.input)
    df["month"] = month
    df["row_id"] = [f"new_{i}" for i in range(len(df))]
    dialog_col = _infer_dialog_column(df, cfg)
    df["raw_dialog"] = df[dialog_col].fillna("").astype(str)
    df["client_first_message"] = df["raw_dialog"].apply(lambda x: extract_client_first_message(x, cfg.client_first_extraction))

    deny = load_tokens(cfg.files.deny_tokens_path) | load_tokens(cfg.files.extra_stopwords_path)
    df["text_clean"] = df["client_first_message"].astype(str).apply(lambda x: clean_for_model(x, deny))

    taxonomy = load_taxonomy(cfg.files.categories_seed_path)

    mdir = Path(cfg.training.model_dir)
    vec = joblib.load(mdir / "vectorizers.joblib")
    complaint_model = joblib.load(mdir / "complaint_model.joblib")
    cat_model = joblib.load(mdir / "category_model.joblib")
    enc = joblib.load(mdir / "label_encoder.joblib")
    subcat_model_path = mdir / "subcategory_model.joblib"
    subcat_enc_path = mdir / "subcategory_label_encoder.joblib"
    subcat_model = joblib.load(subcat_model_path) if subcat_model_path.exists() else None
    subcat_enc = joblib.load(subcat_enc_path) if subcat_enc_path.exists() else None

    x = vec.transform(df["text_clean"])
    if hasattr(complaint_model, "predict_proba"):
        score = complaint_model.predict_proba(x)[:, 1]
    else:
        raw = complaint_model.decision_function(x)
        score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    df["is_complaint_pred"] = score >= cfg.training.complaint_threshold
    df["complaint_score"] = score

    cat_pred = np.array(["OTHER"] * len(df), dtype=object)
    idx = df["is_complaint_pred"].values
    if idx.any() and cat_model is not None:
        cat_pred[idx] = enc.inverse_transform(cat_model.predict(x[idx]))
    elif idx.any() and len(enc.classes_) > 0:
        cat_pred[idx] = enc.classes_[0]
    df["category_pred"] = cat_pred

    subcat_pred = np.array(["NOT_COMPLAINT"] * len(df), dtype=object)
    if idx.any() and subcat_model is not None and subcat_enc is not None:
        subcat_pred[idx] = subcat_enc.inverse_transform(subcat_model.predict(x[idx]))
    subcat_pred = _enforce_subcategory_taxonomy(cat_pred, subcat_pred, taxonomy)
    df["subcategory_pred"] = subcat_pred

    out_xlsx = f"exports/month_labeled_{month}.xlsx"
    out_parq = f"data/interim/month_{month}.parquet"
    Path(out_xlsx).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    _sanitize_for_parquet(df).to_parquet(out_parq, index=False)

    complaints_df = df[df["is_complaint_pred"]].copy()
    top = complaints_df["category_pred"].value_counts().to_dict()
    top_sub = complaints_df["subcategory_pred"].value_counts().to_dict()

    cat_hist_path = Path(f"reports/month_{month}_category_hist.png")
    subcat_hist_path = Path(f"reports/month_{month}_subcategory_hist.png")
    cat_hist_path.parent.mkdir(parents=True, exist_ok=True)

    cat_counts = complaints_df["category_pred"].astype(str).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(cat_counts):
        ax.barh(cat_counts.index.tolist(), cat_counts.values.tolist(), color="#1f77b4")
    else:
        ax.text(0.5, 0.5, "Нет предсказанных жалоб", ha="center", va="center")
    ax.set_title(f"Категории жалоб за {month}")
    ax.set_xlabel("Количество")
    fig.tight_layout()
    fig.savefig(cat_hist_path, dpi=140)
    plt.close(fig)

    subcat_counts = complaints_df["subcategory_pred"].astype(str).value_counts().head(25)
    fig, ax = plt.subplots(figsize=(11, 7))
    if len(subcat_counts):
        ax.barh(subcat_counts.index.tolist(), subcat_counts.values.tolist(), color="#9467bd")
    else:
        ax.text(0.5, 0.5, "Нет предсказанных жалоб", ha="center", va="center")
    ax.set_title(f"Подкатегории жалоб за {month}")
    ax.set_xlabel("Количество")
    fig.tight_layout()
    fig.savefig(subcat_hist_path, dpi=140)
    plt.close(fig)

    empty_dialog_share = float((df["raw_dialog"].str.strip() == "").mean()) if len(df) else 0.0
    empty_client_share = float((df["client_first_message"].str.strip() == "").mean()) if len(df) else 0.0

    render_template("month_report.html.j2", f"reports/month_report_{month}.html", {
        "month": month,
        "dialog_column_used": dialog_col,
        "rows_total": int(len(df)),
        "rows_complaints": int(complaints_df.shape[0]),
        "empty_dialog_share": empty_dialog_share,
        "empty_client_message_share": empty_client_share,
        "complaint_share": float(df["is_complaint_pred"].mean()),
        "top_categories": top,
        "top_subcategories": top_sub,
        "subcategory_filtering": "taxonomy_enforced",
        "category_histogram": str(cat_hist_path).replace("\\", "/"),
        "subcategory_histogram": str(subcat_hist_path).replace("\\", "/"),
        "examples": complaints_df.head(50).to_dict(orient="records"),
    })
    return df
