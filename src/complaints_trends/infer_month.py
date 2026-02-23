from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ProjectConfig
from .extract_client_first import extract_client_first_message
from .io_excel import load_excel_with_month
from .reports.render import render_template
from .text_cleaning import clean_for_model, load_tokens


def infer_month(cfg: ProjectConfig, excel_path: str, month: str) -> pd.DataFrame:
    df = load_excel_with_month(Path(excel_path), cfg.input)
    df["month"] = month
    df["row_id"] = [f"new_{i}" for i in range(len(df))]
    df["raw_dialog"] = df[cfg.input.dialog_column].astype(str)
    df["client_first_message"] = df["raw_dialog"].apply(lambda x: extract_client_first_message(x, cfg.client_first_extraction))

    deny = load_tokens(cfg.files.deny_tokens_path) | load_tokens(cfg.files.extra_stopwords_path)
    df["text_clean"] = df["client_first_message"].astype(str).apply(lambda x: clean_for_model(x, deny))

    mdir = Path(cfg.training.model_dir)
    vec = joblib.load(mdir / "vectorizers.joblib")
    complaint_model = joblib.load(mdir / "complaint_model.joblib")
    cat_model = joblib.load(mdir / "category_model.joblib")
    enc = joblib.load(mdir / "label_encoder.joblib")

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

    out_xlsx = f"exports/month_labeled_{month}.xlsx"
    out_parq = f"data/interim/month_{month}.parquet"
    Path(out_xlsx).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    df.to_parquet(out_parq, index=False)

    top = df[df["is_complaint_pred"]]["category_pred"].value_counts().to_dict()
    render_template("month_report.html.j2", f"reports/month_report_{month}.html", {
        "month": month,
        "complaint_share": float(df["is_complaint_pred"].mean()),
        "top_categories": top,
        "examples": df[df["is_complaint_pred"]].head(50).to_dict(orient="records"),
    })
    return df
