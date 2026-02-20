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


def _non_empty(v) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return bool(s and s.lower() not in {"nan", "none", "null"})


def _get_dialog_fields(cfg: ProjectConfig, df: pd.DataFrame) -> list[str]:
    fields = []
    if cfg.input.dialog_columns:
        fields.extend(cfg.input.dialog_columns)
    if cfg.input.dialog_column:
        fields.append(cfg.input.dialog_column)
    dedup = []
    for c in fields:
        if c not in dedup and c in df.columns:
            dedup.append(c)
    if not dedup:
        raise ValueError("No configured dialog columns found in input file")
    return dedup


def _select_primary_dialog(row: pd.Series, dialog_fields: list[str]) -> tuple[str, str, dict[str, str]]:
    snippets: dict[str, str] = {}
    best_field = dialog_fields[0]
    best_text = ""
    best_len = -1
    for c in dialog_fields:
        txt = str(row.get(c, "") or "").strip()
        if _non_empty(txt):
            snippets[c] = txt
            if len(txt) > best_len:
                best_len = len(txt)
                best_text = txt
                best_field = c
    return best_field, best_text, snippets




def _build_signal_payload(row: pd.Series, signal_columns: list[str], dialog_fields: list[str]) -> dict[str, str]:
    skip = set(dialog_fields)
    out: dict[str, str] = {}
    for c in signal_columns:
        if c in skip:
            continue
        v = row.get(c)
        if _non_empty(v):
            out[c] = str(v).strip()
    return out

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

    dialog_fields = _get_dialog_fields(cfg, df)
    keep = list(dict.fromkeys([*cfg.input.signal_columns, *dialog_fields, "month", "source_file", "row_id"]))
    df = df[[c for c in keep if c in df.columns]].copy()

    selected = df.apply(lambda r: _select_primary_dialog(r, dialog_fields), axis=1)
    df["dialog_source_field"] = selected.apply(lambda x: x[0])
    df["raw_dialog"] = selected.apply(lambda x: x[1])
    df["dialog_context_map"] = selected.apply(lambda x: json.dumps(x[2], ensure_ascii=False))

    df["client_first_message"] = df["raw_dialog"].apply(lambda x: extract_client_first_message(x, cfg.client_first_extraction))
    if cfg.pii.enabled:
        df["client_first_message_redacted"] = df["client_first_message"].apply(lambda x: redact_pii(x, cfg.pii))
        df["dialog_context_map_redacted"] = df["dialog_context_map"].apply(
            lambda x: json.dumps({k: redact_pii(v, cfg.pii) for k, v in json.loads(x).items()}, ensure_ascii=False)
        )
    else:
        df["client_first_message_redacted"] = df["client_first_message"]
        df["dialog_context_map_redacted"] = df["dialog_context_map"]

    taxonomy = load_taxonomy(cfg.files.categories_seed_path)
    normalizer = GigaChatNormalizer(cfg.llm, taxonomy, mock=llm_mock or (not cfg.llm.enabled))

    llm_rows = []
    for _, row in df.iterrows():
        signal_fields = _build_signal_payload(row, cfg.input.signal_columns, dialog_fields)
        payload = {
            "client_first_message": row["client_first_message_redacted"][: cfg.llm.max_text_chars],
            "dialog_source_field": row.get("dialog_source_field"),
            "dialog_context": json.loads(row.get("dialog_context_map_redacted", "{}")),
            "signal_fields": signal_fields,
            "subject": signal_fields.get("subject"),
            "product": signal_fields.get("product"),
            "channel": signal_fields.get("channel"),
            "status": signal_fields.get("status"),
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
            "row_id", "month", "dialog_source_field", "raw_dialog", "client_first_message", "short_summary_llm", "is_complaint_llm", "complaint_category_llm",
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
