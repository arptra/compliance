from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .extract_client_first import extract_client_first_message
from .gigachat_mtls import GigaChatNormalizer
from .gigachat_schema import NormalizeTicket
from .io_excel import read_all_excels
from .pii_redaction import redact_pii
from .reports.render import render_template, write_md
from .taxonomy import load_taxonomy


logger = logging.getLogger(__name__)


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


def _fallback_llm_row(payload: dict, err: Exception) -> dict:
    txt = str(payload.get("full_dialog_text", "") or payload.get("dialog_context", "") or "")
    return {
        "client_first_message": txt,
        "short_summary": txt[:120],
        "is_complaint": False,
        "complaint_category": "OTHER",
        "complaint_subcategory": None,
        "product_area": payload.get("product"),
        "loan_product": "NONE",
        "severity": "low",
        "keywords": ["вопрос", "инфо", "уточнение"],
        "confidence": 0.0,
        "notes": f"LLM_ERROR: {err}",
    }


def _split_payload_batches(normalizer: GigaChatNormalizer, indexed_payloads: list[tuple[int, dict]], token_batch_size: int) -> list[list[tuple[int, dict]]]:
    if not indexed_payloads:
        return []
    batches: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    current_tokens = 0
    limit = max(1, int(token_batch_size))
    for idx, payload in indexed_payloads:
        row_tokens = normalizer.estimate_tokens(payload)
        if current and current_tokens + row_tokens > limit:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((idx, payload))
        current_tokens += row_tokens
    if current:
        batches.append(current)
    return batches


def _run_one(normalizer: GigaChatNormalizer, idx: int, row_id: str, payload: dict) -> tuple[int, dict, dict[str, str] | None]:
    try:
        out = normalizer.normalize(payload)
        return idx, out.model_dump(), None
    except Exception as e:
        return idx, _fallback_llm_row(payload, e), {"row_id": row_id, "error": str(e)}


def _run_sync_payloads(
    cfg: ProjectConfig,
    normalizer: GigaChatNormalizer,
    payloads: list[tuple[int, str, dict]],
) -> tuple[list[dict], list[dict[str, str]]]:
    total_rows = len(payloads)
    llm_rows: list[dict | None] = [None] * total_rows
    llm_errors: list[dict[str, str]] = []

    if cfg.llm.batch_mode:
        indexed_payloads = [(idx, payload) for idx, _, payload in payloads]
        batches = _split_payload_batches(normalizer, indexed_payloads, cfg.llm.token_batch_size)
        processed = 0
        for batch in batches:
            batch_indexes = [idx for idx, _ in batch]
            batch_payloads = [p for _, p in batch]
            try:
                outs = normalizer.normalize_batch(batch_payloads)
                if len(outs) != len(batch):
                    raise RuntimeError(f"Batch output size mismatch: expected={len(batch)} got={len(outs)}")
                for idx, out in zip(batch_indexes, outs):
                    llm_rows[idx] = out.model_dump()
            except Exception:
                for idx, payload in batch:
                    rid = payloads[idx][1]
                    _, row_out, err = _run_one(normalizer, idx, rid, payload)
                    llm_rows[idx] = row_out
                    if err:
                        llm_errors.append(err)
            processed += len(batch)
            if processed == len(batch) or processed % 50 == 0 or processed == total_rows:
                logger.info("[stage=prepare/llm] processed rows %s/%s (remaining=%s)", processed, total_rows, total_rows - processed)
    else:
        for i, (idx, row_id, payload) in enumerate(payloads, start=1):
            _, row_out, err = _run_one(normalizer, idx, row_id, payload)
            llm_rows[idx] = row_out
            if err:
                llm_errors.append(err)
            if i == 1 or i % 50 == 0 or i == total_rows:
                logger.info("[stage=prepare/llm] processed rows %s/%s (remaining=%s)", i, total_rows, total_rows - i)

    filled = [
        (llm_rows[idx] if llm_rows[idx] is not None else _fallback_llm_row(payload, RuntimeError("empty llm row")))
        for idx, _, payload in payloads
    ]
    return filled, llm_errors


def _run_parallel_payloads(
    cfg: ProjectConfig,
    normalizer: GigaChatNormalizer,
    payloads: list[tuple[int, str, dict]],
) -> tuple[list[dict], list[dict[str, str]]]:
    total_rows = len(payloads)
    llm_rows: list[dict | None] = [None] * total_rows
    llm_errors: list[dict[str, str]] = []
    max_workers = max(1, int(cfg.llm.max_workers))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        if cfg.llm.batch_mode:
            indexed_payloads = [(idx, payload) for idx, _, payload in payloads]
            batches = _split_payload_batches(normalizer, indexed_payloads, cfg.llm.token_batch_size)

            def _process_batch(batch: list[tuple[int, dict]]):
                idxs = [idx for idx, _ in batch]
                batch_payloads = [p for _, p in batch]
                try:
                    outs = normalizer.normalize_batch(batch_payloads)
                    if len(outs) != len(batch):
                        raise RuntimeError(f"Batch output size mismatch: expected={len(batch)} got={len(outs)}")
                    return [(i, o.model_dump(), None) for i, o in zip(idxs, outs)]
                except Exception:
                    out = []
                    for idx, payload in batch:
                        rid = payloads[idx][1]
                        out.append(_run_one(normalizer, idx, rid, payload))
                    return out

            futs = [ex.submit(_process_batch, b) for b in batches]
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                for idx, row_out, err in fut.result():
                    llm_rows[idx] = row_out
                    if err:
                        llm_errors.append(err)
                    done += 1
                if done == 1 or done % 50 == 0 or done == total_rows:
                    logger.info("[stage=prepare/llm] processed rows %s/%s (remaining=%s)", done, total_rows, total_rows - done)
        else:
            futs = [ex.submit(_run_one, normalizer, idx, row_id, payload) for idx, row_id, payload in payloads]
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                idx, row_out, err = fut.result()
                llm_rows[idx] = row_out
                if err:
                    llm_errors.append(err)
                done += 1
                if done == 1 or done % 50 == 0 or done == total_rows:
                    logger.info("[stage=prepare/llm] processed rows %s/%s (remaining=%s)", done, total_rows, total_rows - done)

    filled = [
        (llm_rows[idx] if llm_rows[idx] is not None else _fallback_llm_row(payload, RuntimeError("empty llm row")))
        for idx, _, payload in payloads
    ]
    return filled, llm_errors


async def _run_async_payloads(
    cfg: ProjectConfig,
    normalizer: GigaChatNormalizer,
    payloads: list[tuple[int, str, dict]],
) -> tuple[list[dict], list[dict[str, str]]]:
    sem = asyncio.Semaphore(max(1, int(cfg.llm.max_workers)))
    total_rows = len(payloads)
    llm_rows: list[dict | None] = [None] * total_rows
    llm_errors: list[dict[str, str]] = []

    if cfg.llm.batch_mode:
        indexed_payloads = [(idx, payload) for idx, _, payload in payloads]
        batches = _split_payload_batches(normalizer, indexed_payloads, cfg.llm.token_batch_size)

        async def _batch_task(batch):
            idxs = [idx for idx, _ in batch]
            b_payloads = [p for _, p in batch]
            async with sem:
                try:
                    outs = await asyncio.to_thread(normalizer.normalize_batch, b_payloads)
                    if len(outs) != len(batch):
                        raise RuntimeError(f"Batch output size mismatch: expected={len(batch)} got={len(outs)}")
                    return [(i, o.model_dump(), None) for i, o in zip(idxs, outs)]
                except Exception:
                    ret = []
                    for idx, payload in batch:
                        rid = payloads[idx][1]
                        ret.append(await asyncio.to_thread(_run_one, normalizer, idx, rid, payload))
                    return ret

        done = 0
        for coro in asyncio.as_completed([_batch_task(b) for b in batches]):
            for idx, row_out, err in await coro:
                llm_rows[idx] = row_out
                if err:
                    llm_errors.append(err)
                done += 1
            if done == 1 or done % 50 == 0 or done == total_rows:
                logger.info("[stage=prepare/llm] processed rows %s/%s (remaining=%s)", done, total_rows, total_rows - done)
    else:
        async def _single_task(item):
            idx, row_id, payload = item
            async with sem:
                return await asyncio.to_thread(_run_one, normalizer, idx, row_id, payload)

        done = 0
        for coro in asyncio.as_completed([_single_task(p) for p in payloads]):
            idx, row_out, err = await coro
            llm_rows[idx] = row_out
            if err:
                llm_errors.append(err)
            done += 1
            if done == 1 or done % 50 == 0 or done == total_rows:
                logger.info("[stage=prepare/llm] processed rows %s/%s (remaining=%s)", done, total_rows, total_rows - done)

    filled = [
        (llm_rows[idx] if llm_rows[idx] is not None else _fallback_llm_row(payload, RuntimeError("empty llm row")))
        for idx, _, payload in payloads
    ]
    return filled, llm_errors

def prepare_dataset(cfg: ProjectConfig, pilot: bool = False, limit: int | None = None, llm_mock: bool = False, date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    df = read_all_excels(cfg.input)
    if df.empty:
        raise ValueError("No input files found")
    if cfg.input.id_column and cfg.input.id_column in df.columns:
        df["row_id"] = df[cfg.input.id_column].astype(str)
    else:
        df["row_id"] = [f"row_{i}" for i in range(len(df))]
    eff_from = date_from or cfg.prepare.date_from
    eff_to = date_to or cfg.prepare.date_to
    if eff_from:
        df = df[df["event_time"] >= pd.to_datetime(eff_from)].copy()
    if eff_to:
        df = df[df["event_time"] <= pd.to_datetime(eff_to)].copy()
    if pilot and limit:
        df = df.head(limit).copy()

    dialog_fields = _get_dialog_fields(cfg, df)
    keep = list(dict.fromkeys([*cfg.input.signal_columns, *dialog_fields, "event_time", "month", "source_file", "row_id"]))
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

    payloads: list[tuple[int, str, dict]] = []
    total_rows = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        signal_fields = _build_signal_payload(row, cfg.input.signal_columns, dialog_fields)
        dialog_context = json.loads(row.get("dialog_context_map_redacted", "{}"))
        full_dialog_text = "\n".join(f"{k}: {v}" for k, v in dialog_context.items())
        payload = {
            "client_first_message": row["client_first_message_redacted"][: cfg.llm.max_text_chars],
            "full_dialog_text": full_dialog_text[: cfg.llm.max_text_chars * 3],
            "dialog_source_field": row.get("dialog_source_field"),
            "dialog_context": dialog_context,
            "signal_fields": signal_fields,
            "subject": signal_fields.get("subject"),
            "product": signal_fields.get("product"),
            "channel": signal_fields.get("channel"),
            "status": signal_fields.get("status"),
        }
        payloads.append((i - 1, str(row.get("row_id", i)), payload))

    if cfg.llm.async_mode:
        if cfg.llm.parallel_mode:
            logger.info("[stage=prepare/llm] async_mode=true (parallel_mode ignored)")
        llm_rows_filled, llm_errors = asyncio.run(_run_async_payloads(cfg, normalizer, payloads))
    elif cfg.llm.parallel_mode:
        llm_rows_filled, llm_errors = _run_parallel_payloads(cfg, normalizer, payloads)
    else:
        llm_rows_filled, llm_errors = _run_sync_payloads(cfg, normalizer, payloads)
    llm_df = pd.DataFrame(llm_rows_filled)
    if llm_df.empty:
        llm_df = pd.DataFrame(columns=list(NormalizeTicket.model_fields.keys()))
    out_df = pd.concat([df.reset_index(drop=True), llm_df.add_suffix("_llm")], axis=1)
    out_df["llm_error"] = out_df.get("notes_llm", "").astype(str).where(out_df.get("notes_llm", "").astype(str).str.startswith("LLM_ERROR:"), "")

    out_path = cfg.prepare.pilot_parquet if pilot else cfg.prepare.output_parquet
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    if llm_errors:
        err_path = Path(cfg.analysis.reports_dir) / "llm_errors.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(json.dumps(llm_errors, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.warning("[stage=prepare/llm] rows with LLM errors: %s; saved to %s", len(llm_errors), err_path)
        logger.warning("[stage=prepare/llm] sample errors: %s", llm_errors[:3])

    if pilot:
        review_cols = [
            "row_id", "month", "dialog_source_field", "raw_dialog", "client_first_message", "short_summary_llm", "is_complaint_llm", "complaint_category_llm",
            "complaint_subcategory_llm", "loan_product_llm", "severity_llm", "keywords_llm", "confidence_llm", "llm_error",
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
    if "is_complaint_llm" not in df.columns:
        df = df.copy()
        df["is_complaint_llm"] = False
    if "complaint_category_llm" not in df.columns:
        df["complaint_category_llm"] = "OTHER"
    if "short_summary_llm" not in df.columns:
        df["short_summary_llm"] = ""

    complaints = df[df["is_complaint_llm"] == True]
    non_complaints = df[df["is_complaint_llm"] == False]
    taxonomy = load_taxonomy(cfg.files.categories_seed_path)
    cat_labels = taxonomy.get("category_labels", {})
    sub_labels = taxonomy.get("subcategory_labels", {})
    loan_labels = taxonomy.get("loan_product_labels", {})

    top = complaints["complaint_category_llm"].value_counts().head(10).to_dict()
    top_sub = complaints["complaint_subcategory_llm"].value_counts().head(10).to_dict() if "complaint_subcategory_llm" in complaints.columns else {}
    top_loan = complaints["loan_product_llm"].value_counts().head(10).to_dict() if "loan_product_llm" in complaints.columns else {}
    warn = df["short_summary_llm"].astype(str).str.contains(r"CLIENT|OPERATOR|CHATBOT", case=False).any()
    context = {
        "n": len(df),
        "complaint_share": float(df["is_complaint_llm"].mean()) if len(df) else 0.0,
        "top_categories": top,
        "top_categories_ru": {f"{k} ({cat_labels.get(k, k)})": v for k, v in top.items()},
        "top_subcategories_ru": {
            f"{k} ({sub_labels.get(complaints.loc[complaints['complaint_subcategory_llm'] == k, 'complaint_category_llm'].iloc[0], {}).get(k, k)})": v
            for k, v in top_sub.items()
            if "complaint_subcategory_llm" in complaints.columns and not complaints.loc[complaints['complaint_subcategory_llm'] == k, 'complaint_category_llm'].empty
        },
        "top_loan_products_ru": {f"{k} ({loan_labels.get(k, k)})": v for k, v in top_loan.items()},
        "complaint_examples": complaints.head(30).to_dict(orient="records"),
        "non_examples": non_complaints.head(30).to_dict(orient="records"),
        "warning": warn,
    }
    render_template("pilot_report.html.j2", "reports/pilot_report.html", context)
    write_md("reports/pilot_report.md", "# Pilot report\n\nПроверить чеклист, категории и примеры.")
