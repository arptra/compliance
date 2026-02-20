from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .config import ProjectConfig
from .features import TextVectorizers
from .reports.render import render_template, write_md
from .text_cleaning import clean_for_model, load_tokens


def train(cfg: ProjectConfig) -> dict:
    df = pd.read_parquet(cfg.prepare.output_parquet)
    try:
        gold = pd.read_excel(cfg.prepare.pilot_review_xlsx)
        gold = gold[["row_id", "is_complaint_gold", "category_gold"]]
        df = df.merge(gold, on="row_id", how="left")
    except Exception:
        df["is_complaint_gold"] = np.nan
        df["category_gold"] = np.nan

    y_bin = np.where(df["is_complaint_gold"].notna() & (df["is_complaint_gold"] != ""), df["is_complaint_gold"], df["is_complaint_llm"])
    y_cat = np.where(df["category_gold"].notna() & (df["category_gold"] != ""), df["category_gold"], df["complaint_category_llm"])

    deny = load_tokens(cfg.files.deny_tokens_path) | load_tokens(cfg.files.extra_stopwords_path)
    text_col = cfg.training.text_field
    df["text_clean"] = df[text_col].astype(str).apply(lambda x: clean_for_model(x, deny))

    train_df, val_df = _split(df, cfg)
    y_train = y_bin[train_df.index]
    y_val = y_bin[val_df.index]

    vec = TextVectorizers(cfg.training)
    x_train = vec.fit_transform(train_df["text_clean"])
    x_val = vec.transform(val_df["text_clean"])

    if cfg.training.classifier.complaint == "linearsvc":
        base = LinearSVC(class_weight="balanced")
        complaint_model = CalibratedClassifierCV(base)
    else:
        complaint_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    complaint_model.fit(x_train, y_train)

    if hasattr(complaint_model, "predict_proba"):
        complaint_score = complaint_model.predict_proba(x_val)[:, 1]
    else:
        raw = complaint_model.decision_function(x_val)
        complaint_score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    pred_bin = complaint_score >= cfg.training.complaint_threshold

    ctrain = train_df[np.array(y_train).astype(bool)]
    cval = val_df[np.array(y_val).astype(bool)]
    ycat_train = np.array(y_cat)[ctrain.index]
    ycat_val = np.array(y_cat)[cval.index]
    enc = LabelEncoder().fit(ycat_train)
    x_cat_train = vec.transform(ctrain["text_clean"])
    x_cat_val = vec.transform(cval["text_clean"])

    if len(enc.classes_) < 2:
        cat_model = None
        cat_pred = np.array([enc.classes_[0]] * len(cval)) if len(cval) else np.array([])
    else:
        if cfg.training.classifier.category == "logreg":
            cat_model = LogisticRegression(max_iter=2500, class_weight="balanced", multi_class="multinomial")
        else:
            cat_model = LinearSVC(class_weight="balanced")
        cat_model.fit(x_cat_train, enc.transform(ycat_train))
        cat_pred = enc.inverse_transform(cat_model.predict(x_cat_val)) if len(cval) else np.array([])

    metrics = {
        "complaint_report": classification_report(y_val, pred_bin, output_dict=True),
        "complaint_f1": f1_score(y_val, pred_bin),
        "category_macro_f1": f1_score(ycat_val, cat_pred, average="macro") if len(cval) else 0.0,
    }

    mdir = Path(cfg.training.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, mdir / "vectorizers.joblib")
    joblib.dump(complaint_model, mdir / "complaint_model.joblib")
    joblib.dump(cat_model, mdir / "category_model.joblib")
    joblib.dump(enc, mdir / "label_encoder.joblib")
    (mdir / "training_metadata.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    render_template("training_report.html.j2", "reports/training_report.html", {"metrics": metrics})
    write_md("reports/training_report.md", "# Training report\n\nСм. HTML.")
    return metrics


def _split(df: pd.DataFrame, cfg: ProjectConfig):
    if cfg.training.validation.split_mode == "time":
        if "event_time" not in df.columns:
            return train_test_split(df, test_size=0.2, random_state=42)
        d = df.sort_values("event_time").copy()
        vf = cfg.training.validation.val_from
        vt = cfg.training.validation.val_to
        if vf or vt:
            m = pd.Series(True, index=d.index)
            if vf:
                m &= d["event_time"] >= pd.to_datetime(vf)
            if vt:
                m &= d["event_time"] <= pd.to_datetime(vt)
            val_df = d[m].copy()
            train_df = d[~m].copy()
            if len(val_df) > 0 and len(train_df) > 0:
                return train_df, val_df
        cut = int(len(d) * 0.8)
        return d.iloc[:cut].copy(), d.iloc[cut:].copy()
    return train_test_split(df, test_size=0.2, random_state=42)
