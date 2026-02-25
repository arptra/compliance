from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .config import ProjectConfig
from .features import TextVectorizers
from .reports.render import render_template, write_md
from .text_cleaning import clean_for_model, load_tokens



def _save_training_charts(x_cat_val, ycat_val, subcat_val, pred_bin, cat_pred, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Гистограмма жалоба/не жалоба (предсказание).
    bin_counts = pd.Series(pred_bin).map({True: "Жалоба", False: "Не жалоба"}).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bin_counts.index.tolist(), bin_counts.values.tolist(), color=["#d62728", "#2ca02c"][: len(bin_counts)])
    ax.set_title("Гистограмма предсказаний: жалоба / не жалоба")
    ax.set_ylabel("Количество")
    fig.tight_layout()
    bin_path = out_dir / "training_predicted_complaint_distribution.png"
    fig.savefig(bin_path, dpi=140)
    plt.close(fig)

    # 2) Гистограмма категорий (валидация, жалобы).
    cat_series = pd.Series(ycat_val) if len(ycat_val) else pd.Series([], dtype=object)
    cat_counts = cat_series.value_counts().head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    if len(cat_counts):
        ax.barh(cat_counts.index.astype(str).tolist(), cat_counts.values.tolist(), color="#1f77b4")
    ax.set_title("Гистограмма категорий (валидационная выборка, жалобы)")
    ax.set_xlabel("Количество")
    fig.tight_layout()
    cat_path = out_dir / "training_category_hist_ru.png"
    fig.savefig(cat_path, dpi=140)
    plt.close(fig)

    # 3) Гистограмма подкатегорий (валидация, жалобы).
    subcat_series = pd.Series(subcat_val) if len(subcat_val) else pd.Series([], dtype=object)
    subcat_counts = subcat_series.replace({None: "UNKNOWN", "": "UNKNOWN"}).astype(str).value_counts().head(25)
    fig, ax = plt.subplots(figsize=(10, 7))
    if len(subcat_counts):
        ax.barh(subcat_counts.index.tolist(), subcat_counts.values.tolist(), color="#9467bd")
    ax.set_title("Гистограмма подкатегорий (валидационная выборка, жалобы)")
    ax.set_xlabel("Количество")
    fig.tight_layout()
    subcat_path = out_dir / "training_subcategory_hist_ru.png"
    fig.savefig(subcat_path, dpi=140)
    plt.close(fig)

    # 4) Скопление точек (2D SVD) по категориям и подкатегориям.
    scatter_path = out_dir / "training_category_subcategory_scatter_ru.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    if x_cat_val.shape[0] >= 2:
        svd = TruncatedSVD(n_components=2, random_state=42)
        z = svd.fit_transform(x_cat_val)
        cats = pd.Series(ycat_val).astype(str)
        subcats = pd.Series(subcat_val).replace({None: "UNKNOWN", "": "UNKNOWN"}).astype(str)
        unique_cats = sorted(cats.unique().tolist())
        cmap = plt.get_cmap("tab20")
        markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
        top_sub = subcats.value_counts().head(len(markers)).index.tolist()
        marker_map = {s: markers[i % len(markers)] for i, s in enumerate(top_sub)}
        for i, cat in enumerate(unique_cats):
            m = cats == cat
            for sub in subcats[m].unique().tolist():
                ms = m & (subcats == sub)
                ax.scatter(
                    z[ms.values, 0],
                    z[ms.values, 1],
                    color=cmap(i % 20),
                    marker=marker_map.get(sub, "o"),
                    alpha=0.75,
                    s=40,
                    label=f"{cat} / {sub}",
                )
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 20:
            handles, labels = handles[:20], labels[:20]
        ax.legend(handles, labels, fontsize=7, loc="best")
        ax.set_title("Скопление точек по категориям/подкатегориям (SVD 2D)")
        ax.set_xlabel("Компонента 1")
        ax.set_ylabel("Компонента 2")
    else:
        ax.text(0.5, 0.5, "Недостаточно данных для scatter", ha="center", va="center")
        ax.set_title("Скопление точек по категориям/подкатегориям")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=140)
    plt.close(fig)

    return {
        "binary_distribution": str(bin_path).replace('\\', '/'),
        "category_distribution": str(cat_path).replace('\\', '/'),
        "subcategory_distribution": str(subcat_path).replace('\\', '/'),
        "category_subcategory_scatter": str(scatter_path).replace('\\', '/'),
    }

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

    subcat_val = cval["complaint_subcategory_llm"].astype(str).values if "complaint_subcategory_llm" in cval.columns else np.array(["UNKNOWN"] * len(cval), dtype=object)
    charts = _save_training_charts(x_cat_val=x_cat_val, ycat_val=ycat_val, subcat_val=subcat_val, pred_bin=pred_bin, cat_pred=cat_pred, out_dir=Path("reports"))

    mdir = Path(cfg.training.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, mdir / "vectorizers.joblib")
    joblib.dump(complaint_model, mdir / "complaint_model.joblib")
    joblib.dump(cat_model, mdir / "category_model.joblib")
    joblib.dump(enc, mdir / "label_encoder.joblib")
    (mdir / "training_metadata.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    render_template("training_report.html.j2", "reports/training_report.html", {"metrics": metrics, "charts": charts})

    human_md = """# Training report

## Кратко
- Бинарная модель жалоб обучена и провалидирована на отложенной выборке.
- Категориальная модель обучена только на записях, пред/факт которых относится к жалобам.

## Основные метрики
- complaint_f1: {complaint_f1:.4f}
- category_macro_f1: {category_macro_f1:.4f}

## Что смотреть в графиках
1. `training_predicted_complaint_distribution.png` — сколько модель отнесла к жалобам/не-жалобам.
2. `training_category_hist_ru.png` — распределение категорий.
3. `training_subcategory_hist_ru.png` — распределение подкатегорий.
4. `training_category_subcategory_scatter_ru.png` — скопление точек по категориям/подкатегориям.

## Файлы
- HTML: `reports/training_report.html`
- Графики: `reports/training_predicted_complaint_distribution.png`, `reports/training_category_hist_ru.png`, `reports/training_subcategory_hist_ru.png`, `reports/training_category_subcategory_scatter_ru.png`
- Метаданные: `models/training_metadata.json`
""".format(
        complaint_f1=metrics["complaint_f1"],
        category_macro_f1=metrics["category_macro_f1"],
    )
    write_md("reports/training_report.md", human_md)
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
