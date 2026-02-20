from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .extract_client_first import extract_client_first_message
from .gigachat_mtls import GigaChatNormalizer
from .io_excel import read_all_excels
from .pii_redaction import redact_pii
from .reports.render import render_template, write_md
from .taxonomy import load_taxonomy


def prepare_dataset(cfg: ProjectConfig, pilot: bool = False, month: str | None = None, limit: int | None = None, llm_mock: bool = False) -> pd.DataFrame:
    df = read_all_excels(cfg.input)
    if df.empty:
        raise ValueError("No input files found")
    if cfg.input.id_column and cfg.input.id_column in df.columns:
        df["row_id"] = df[cfg.input.id_column].astype(str)
    else:
        df["row_id"] = [f"row_{i}" for i in range(len(df))]
    if month:
        df = df[df["month"] == month].copy()
    if pilot and limit:
        df = df.head(limit).copy()

    keep = list(dict.fromkeys([*cfg.input.signal_columns, cfg.input.dialog_column, "month", "source_file", "row_id"]))
    df = df[[c for c in keep if c in df.columns]].copy()
    df["raw_dialog"] = df[cfg.input.dialog_column].astype(str)
    df["client_first_message"] = df["raw_dialog"].apply(lambda x: extract_client_first_message(x, cfg.client_first_extraction))
    if cfg.pii.enabled:
        df["client_first_message_redacted"] = df["client_first_message"].apply(lambda x: redact_pii(x, cfg.pii))
    else:
        df["client_first_message_redacted"] = df["client_first_message"]

    taxonomy = load_taxonomy(cfg.files.categories_seed_path)
    normalizer = GigaChatNormalizer(cfg.llm, taxonomy, mock=llm_mock or (not cfg.llm.enabled))

    llm_rows = []
    for _, row in df.iterrows():
        payload = {
            "client_first_message": row["client_first_message_redacted"][: cfg.llm.max_text_chars],
            "subject": row.get("subject"),
            "product": row.get("product"),
            "channel": row.get("channel"),
            "status": row.get("status"),
        }
        out = normalizer.normalize(payload)
        llm_rows.append(out.model_dump())
    llm_df = pd.DataFrame(llm_rows)
    out_df = pd.concat([df.reset_index(drop=True), llm_df.add_suffix("_llm")], axis=1)

    out_path = cfg.prepare.pilot_parquet if pilot else cfg.prepare.output_parquet
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    if pilot:
        review_cols = [
            "row_id", "month", "raw_dialog", "client_first_message", "short_summary_llm", "is_complaint_llm", "complaint_category_llm",
            "complaint_subcategory_llm", "severity_llm", "keywords_llm", "confidence_llm",
        ]
        review = out_df[[c for c in review_cols if c in out_df.columns]].copy()
        review["is_complaint_gold"] = ""
        review["category_gold"] = ""
        review["subcategory_gold"] = ""
        review["comment"] = ""
        Path(cfg.prepare.pilot_review_xlsx).parent.mkdir(parents=True, exist_ok=True)
        review.to_excel(cfg.prepare.pilot_review_xlsx, index=False)
        _pilot_report(out_df, cfg)
    return out_df


def _pilot_report(df: pd.DataFrame, cfg: ProjectConfig) -> None:
    complaints = df[df["is_complaint_llm"] == True]
    non_complaints = df[df["is_complaint_llm"] == False]
    top = complaints["complaint_category_llm"].value_counts().head(10).to_dict()
    warn = df["short_summary_llm"].astype(str).str.contains(r"CLIENT|OPERATOR|CHATBOT", case=False).any()
    context = {
        "n": len(df),
        "complaint_share": float(df["is_complaint_llm"].mean()),
        "top_categories": top,
        "complaint_examples": complaints.head(30).to_dict(orient="records"),
        "non_examples": non_complaints.head(30).to_dict(orient="records"),
        "warning": warn,
    }
    render_template("pilot_report.html.j2", "reports/pilot_report.html", context)
    write_md("reports/pilot_report.md", "# Pilot report\n\nПроверить чеклист, категории и примеры.")
