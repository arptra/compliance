from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import InputConfig


def discover_excel_files(cfg: InputConfig) -> list[Path]:
    return sorted(Path(cfg.input_dir).glob(cfg.file_glob))


def extract_month_from_filename(path: Path, pattern: str) -> str | None:
    m = re.search(pattern, path.name)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def load_excel_with_month(path: Path, cfg: InputConfig) -> pd.DataFrame:
    df = pd.read_excel(path)
    if cfg.month_source == "filename":
        month = extract_month_from_filename(path, cfg.month_regex)
        df["month"] = month
    else:
        if not cfg.month_column:
            raise ValueError("month_column must be set when month_source=column")
        df["month"] = df[cfg.month_column].astype(str).str[:7]
    df["source_file"] = path.name
    return df


def read_all_excels(cfg: InputConfig) -> pd.DataFrame:
    frames = [load_excel_with_month(path, cfg) for path in discover_excel_files(cfg)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
