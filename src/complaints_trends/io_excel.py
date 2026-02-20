from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import InputConfig


def discover_excel_files(cfg: InputConfig) -> list[Path]:
    base = Path(cfg.input_dir)
    if cfg.file_names:
        files = [base / name for name in cfg.file_names]
    else:
        files = sorted(base.glob(cfg.file_glob))
    return [p for p in files if p.exists()]


def _resolve_datetime_column(cfg: InputConfig, columns: list[str]) -> str:
    candidates = [cfg.datetime_column, cfg.month_column]
    for c in candidates:
        if c and c in columns:
            return c
    raise ValueError(
        f"No datetime column found. Configure input.datetime_column (current={cfg.datetime_column!r}); "
        f"available columns: {columns[:20]}"
    )


def parse_event_time(series: pd.Series, dt_format: str | None = None) -> pd.Series:
    if dt_format:
        dt = pd.to_datetime(series, format=dt_format, errors="coerce")
    else:
        dt = pd.to_datetime(series, errors="coerce")
    return dt




def extract_month_from_filename(path: Path, pattern: str | None = None, patterns: list[str] | None = None) -> str | None:
    import re
    candidates = patterns or ([] if pattern is None else [pattern])
    for p in candidates:
        m = re.search(p, path.name)
        if m and m.lastindex and m.lastindex >= 2:
            return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(\d{4})[-_](\d{2})", path.name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None


def extract_month_from_column(series: pd.Series, explicit_format: str | None = None) -> pd.Series:
    dt = parse_event_time(series, explicit_format)
    out = dt.dt.strftime("%Y-%m")
    return out.fillna("")

def load_excel_with_month(path: Path, cfg: InputConfig) -> pd.DataFrame:
    df = pd.read_excel(path)
    if cfg.month_source == "filename":
        month = extract_month_from_filename(path, cfg.month_regex, cfg.month_regexes)
        if not month:
            raise ValueError(
                f"Cannot parse month from filename: {path.name}. "
                f"Check input.month_regex/month_regexes (current: {cfg.month_regex!r}, {cfg.month_regexes!r}) "
                "or set month_source=column with month_column."
            )
        df["month"] = month
        # still require event_time for downstream period filtering
        dt_col = _resolve_datetime_column(cfg, list(df.columns))
        df["event_time"] = parse_event_time(df[dt_col], cfg.datetime_format or cfg.month_column_datetime_format)
    else:
        dt_col = _resolve_datetime_column(cfg, list(df.columns))
        df["event_time"] = parse_event_time(df[dt_col], cfg.datetime_format or cfg.month_column_datetime_format)
    bad = int(df["event_time"].isna().sum())
    if bad:
        raise ValueError(
            f"Cannot parse {bad} values in datetime column '{dt_col}' for file {path.name}. "
            "Expected values like '2025-01-09 12:55:29'."
        )
    if "month" not in df.columns:
        df["month"] = df["event_time"].dt.strftime("%Y-%m")
    df["source_file"] = path.name
    return df


def read_all_excels(cfg: InputConfig) -> pd.DataFrame:
    frames = [load_excel_with_month(path, cfg) for path in discover_excel_files(cfg)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
