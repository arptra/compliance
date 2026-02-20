from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import InputConfig


MONTH_RE_FALLBACK = re.compile(r"(\d{4})[-_](\d{2})")


def discover_excel_files(cfg: InputConfig) -> list[Path]:
    base = Path(cfg.input_dir)
    if cfg.file_names:
        files = [base / name for name in cfg.file_names]
    else:
        files = sorted(base.glob(cfg.file_glob))
    return [p for p in files if p.exists()]


def _month_from_regex_match(match: re.Match[str] | None) -> str | None:
    if not match:
        return None
    if match.lastindex and match.lastindex >= 2:
        return f"{match.group(1)}-{match.group(2)}"
    val = match.group(1)
    m2 = MONTH_RE_FALLBACK.search(val)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}"
    return None


def extract_month_from_filename(path: Path, pattern: str | None = None, patterns: list[str] | None = None) -> str | None:
    candidates = patterns or ([] if pattern is None else [pattern])
    for p in candidates:
        m = re.search(p, path.name)
        month = _month_from_regex_match(m)
        if month:
            return month
    # fallback for names like 2025-01-09_report.xlsx
    m = MONTH_RE_FALLBACK.search(path.name)
    return _month_from_regex_match(m)


def extract_month_from_column(series: pd.Series, explicit_format: str | None = None) -> pd.Series:
    if explicit_format:
        dt = pd.to_datetime(series, format=explicit_format, errors="coerce")
    else:
        dt = pd.to_datetime(series, errors="coerce")
    out = dt.dt.strftime("%Y-%m")
    mask = out.isna()
    if mask.any():
        raw = series.astype(str)
        out.loc[mask] = raw.loc[mask].str.extract(r"(\d{4}[-_]\d{2})", expand=False).str.replace("_", "-", regex=False)
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
    else:
        if not cfg.month_column:
            raise ValueError("month_column must be set when month_source=column")
        df["month"] = extract_month_from_column(df[cfg.month_column], cfg.month_column_datetime_format)
    df["source_file"] = path.name
    return df


def read_all_excels(cfg: InputConfig) -> pd.DataFrame:
    frames = [load_excel_with_month(path, cfg) for path in discover_excel_files(cfg)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
