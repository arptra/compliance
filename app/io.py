from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class DataSplits:
    baseline: pd.DataFrame
    december: pd.DataFrame
    combined: pd.DataFrame


class SchemaError(ValueError):
    """Raised when required input columns are missing."""


def read_excel(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}. Available: {list(df.columns)}")


def get_message_columns(input_cfg: Dict[str, Any]) -> list[str]:
    cols = input_cfg.get("message_cols")
    if cols:
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise SchemaError("input.message_cols must be a list of column names")
        return cols

    col = input_cfg.get("message_col")
    if not isinstance(col, str) or not col:
        raise SchemaError("Provide input.message_col or input.message_cols in config")
    return [col]


def merge_message_columns(df: pd.DataFrame, message_cols: list[str], out_col: str = "message_joined") -> pd.Series:
    _require_columns(df, message_cols)
    merged = (
        df[message_cols]
        .fillna("")
        .astype(str)
        .apply(lambda r: " ".join([x.strip() for x in r.tolist() if x and x.strip()]), axis=1)
    )
    if out_col not in df.columns:
        df[out_col] = merged
    else:
        df[out_col] = merged
    return df[out_col]


def _prepare_dates(df: pd.DataFrame, date_col: Optional[str], date_format: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if date_col and date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], format=date_format, errors="coerce")
        out["year_month"] = out[date_col].dt.to_period("M").astype("string")
    elif "month" in out.columns and "year_month" not in out.columns:
        out["year_month"] = out["month"].astype("string")
    return out


def _filter_by_mode(df: pd.DataFrame, cfg: Dict[str, Any], date_col: Optional[str]) -> pd.Series:
    mode = cfg.get("mode")
    if mode == "explicit_filter":
        col = cfg.get("filter_col")
        val = cfg.get("filter_value")
        if col is None or col not in df.columns:
            raise SchemaError("explicit_filter mode requires existing filter_col")
        return df[col].astype("string") == str(val)
    if mode == "date_range":
        if not date_col or date_col not in df.columns:
            raise SchemaError("date_range mode requires date_col")
        start = pd.to_datetime(cfg.get("start"), errors="coerce")
        end = pd.to_datetime(cfg.get("end"), errors="coerce")
        return (df[date_col] >= start) & (df[date_col] <= end)
    if mode == "month_value":
        month_value = str(cfg.get("month_value"))
        if "year_month" not in df.columns:
            raise SchemaError("month_value mode requires parsed year_month column")
        return df["year_month"].astype("string") == month_value
    if mode == "last_n_months":
        if "year_month" not in df.columns:
            raise SchemaError("last_n_months mode requires parsed year_month column")
        months = df["year_month"].dropna().sort_values().unique().tolist()
        n = int(cfg.get("n_months", 6))
        selected = set(months[-n:])
        return df["year_month"].isin(selected)
    raise SchemaError(f"Unsupported split mode: {mode}")


def _load_from_paths(paths: list[str], source_tag: str) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        dfi = read_excel(p)
        dfi["_source_file"] = str(p)
        dfi["_source_split"] = source_tag
        frames.append(dfi)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_input_data(config: Dict[str, Any]) -> pd.DataFrame:
    input_cfg = config["input"]

    # mode 1: single dataset path with in-file splitting rules
    if input_cfg.get("path"):
        return read_excel(input_cfg["path"])

    # mode 2: multiple baseline files + separate december file/list
    baseline_paths = input_cfg.get("baseline_paths") or []
    december_paths = input_cfg.get("december_paths") or ([] if not input_cfg.get("december_path") else [input_cfg["december_path"]])

    if not baseline_paths and not december_paths:
        raise SchemaError("Provide either input.path OR input.baseline_paths / input.december_path(s)")

    baseline_df = _load_from_paths(baseline_paths, "baseline")
    december_df = _load_from_paths(december_paths, "december")
    return pd.concat([baseline_df, december_df], ignore_index=True)


def split_baseline_december(df: pd.DataFrame, cfg: Dict[str, Any]) -> DataSplits:
    input_cfg = cfg["input"]
    message_cols = get_message_columns(input_cfg)
    date_col = input_cfg.get("date_col")
    month_col = input_cfg.get("month_col")

    required = list(message_cols)
    if date_col:
        required.append(date_col)
    if month_col:
        required.append(month_col)
    _require_columns(df, required)

    work = _prepare_dates(df, date_col, input_cfg.get("date_format"))
    if month_col and month_col in work.columns:
        work["year_month"] = work[month_col].astype("string")

    merge_message_columns(work, message_cols, out_col="message_joined")

    if "_source_split" in work.columns and not input_cfg.get("path"):
        baseline_mask = work["_source_split"] == "baseline"
        december_mask = work["_source_split"] == "december"
    else:
        baseline_mask = _filter_by_mode(work, input_cfg["baseline"], date_col)
        december_mask = _filter_by_mode(work, input_cfg["december"], date_col)

    baseline = work[baseline_mask].copy()
    december = work[december_mask].copy()
    baseline["split"] = "baseline"
    december["split"] = "december"

    combined = pd.concat([baseline, december], axis=0, ignore_index=True)
    return DataSplits(baseline=baseline, december=december, combined=combined)


def load_and_split(config: Dict[str, Any]) -> DataSplits:
    df = _load_input_data(config)
    return split_baseline_december(df, config)


def save_labeled_outputs(
    baseline_df: pd.DataFrame,
    december_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    output_cfg: Dict[str, Any],
) -> None:
    Path(output_cfg["labeled_baseline_xlsx"]).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_cfg["labeled_baseline_xlsx"], engine="openpyxl") as w:
        baseline_df.to_excel(w, index=False, sheet_name="baseline")
    with pd.ExcelWriter(output_cfg["labeled_december_xlsx"], engine="openpyxl") as w:
        december_df.to_excel(w, index=False, sheet_name="december")
    with pd.ExcelWriter(output_cfg["combined_xlsx"], engine="openpyxl") as w:
        combined_df.to_excel(w, index=False, sheet_name="combined")
