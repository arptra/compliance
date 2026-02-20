from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from app.extract import extract_first_client_message
from app.preprocess import TextPreprocessor


@dataclass
class DataSplits:
    baseline: pd.DataFrame
    december: pd.DataFrame
    combined: pd.DataFrame


class SchemaError(ValueError):
    pass


def read_excel(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c and c not in df.columns]
    if miss:
        raise SchemaError(f"Missing required columns: {miss}")


def get_message_columns(input_cfg: Dict[str, Any]) -> list[str]:
    cols = input_cfg.get("message_cols")
    if cols:
        return cols
    col = input_cfg.get("message_col")
    if not col:
        raise SchemaError("Set input.message_col or input.message_cols")
    return [col]


def _prepare_dates(df: pd.DataFrame, input_cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    date_col = input_cfg.get("date_col")
    fmt = input_cfg.get("date_format")
    month_col = input_cfg.get("month_col")
    if date_col and date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], format=fmt, errors="coerce")
        out["year_month"] = out[date_col].dt.to_period("M").astype("string")
    elif month_col and month_col in out.columns:
        out["year_month"] = out[month_col].astype("string")
    return out


def _load_from_paths(paths: list[str], split_name: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = read_excel(p)
        d["_source_split"] = split_name
        d["_source_file"] = p
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_input_df(config: Dict[str, Any]) -> pd.DataFrame:
    i = config["input"]
    if i.get("path"):
        return read_excel(i["path"])

    baseline_paths = i.get("baseline_paths") or []
    december_paths = i.get("december_paths") or ([] if not i.get("december_path") else [i["december_path"]])
    if not baseline_paths and not december_paths:
        raise SchemaError("Provide input.path or baseline_paths/december_path(s)")
    return pd.concat(
        [_load_from_paths(baseline_paths, "baseline"), _load_from_paths(december_paths, "december")],
        ignore_index=True,
    )


def _apply_text_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    out_cfg = config.get("output", {})
    i_cfg = config["input"]
    raw_col = out_cfg.get("raw_text_col_name", "message_raw")
    analyzed_col = out_cfg.get("analyzed_text_col_name", "message_client_first")

    msg_cols = get_message_columns(i_cfg)
    _require_columns(df, msg_cols)

    merged = (
        df[msg_cols].fillna("").astype(str).apply(lambda r: " ".join([x.strip() for x in r.tolist() if x and x.strip()]), axis=1)
    )
    df[raw_col] = merged
    df[analyzed_col] = df[raw_col].map(lambda t: extract_first_client_message(t, config))

    pre = TextPreprocessor(config)
    df["message_clean"] = [" ".join(pre.analyzer(t)) for t in df[analyzed_col].fillna("").astype(str)]
    return df


def _filter_by_mode(df: pd.DataFrame, cfg: Dict[str, Any], date_col: Optional[str]) -> pd.Series:
    mode = cfg.get("mode")
    if mode == "explicit_filter":
        return df[cfg["filter_col"]].astype("string") == str(cfg["filter_value"])
    if mode == "month_value":
        return df["year_month"].astype("string") == str(cfg.get("month_value"))
    if mode == "date_range":
        start = pd.to_datetime(cfg.get("start"), errors="coerce")
        end = pd.to_datetime(cfg.get("end"), errors="coerce")
        if not date_col:
            raise SchemaError("date_range requires input.date_col")
        return (df[date_col] >= start) & (df[date_col] <= end)
    if mode == "last_n_months":
        months = df["year_month"].dropna().sort_values().unique().tolist()
        n = int(cfg.get("n_months", 6))
        return df["year_month"].isin(set(months[-n:]))
    raise SchemaError(f"Unsupported mode: {mode}")


def split_baseline_december(df: pd.DataFrame, cfg: Dict[str, Any]) -> DataSplits:
    i_cfg = cfg["input"]
    date_col = i_cfg.get("date_col")
    df = _prepare_dates(df, i_cfg)
    df = _apply_text_columns(df, cfg)

    if "_source_split" in df.columns and not i_cfg.get("path"):
        bmask = df["_source_split"] == "baseline"
        dmask = df["_source_split"] == "december"
    else:
        bmask = _filter_by_mode(df, i_cfg["baseline"], date_col)
        dmask = _filter_by_mode(df, i_cfg["december"], date_col)

    baseline = df[bmask].copy()
    december = df[dmask].copy()
    baseline["split"] = "baseline"
    december["split"] = "december"
    combined = pd.concat([baseline, december], ignore_index=True)
    return DataSplits(baseline=baseline, december=december, combined=combined)


def load_and_split(config: Dict[str, Any]) -> DataSplits:
    return split_baseline_december(load_input_df(config), config)


def save_labeled_outputs(baseline_df: pd.DataFrame, december_df: pd.DataFrame, combined_df: pd.DataFrame, output_cfg: Dict[str, Any]) -> None:
    Path(output_cfg["labeled_baseline_xlsx"]).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_cfg["labeled_baseline_xlsx"], engine="openpyxl") as w:
        baseline_df.to_excel(w, index=False, sheet_name="baseline")
    with pd.ExcelWriter(output_cfg["labeled_december_xlsx"], engine="openpyxl") as w:
        december_df.to_excel(w, index=False, sheet_name="december")
    with pd.ExcelWriter(output_cfg["combined_xlsx"], engine="openpyxl") as w:
        combined_df.to_excel(w, index=False, sheet_name="combined")
