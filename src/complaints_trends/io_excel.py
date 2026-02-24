from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

from .config import InputConfig


logger = logging.getLogger(__name__)


def discover_excel_files(cfg: InputConfig) -> list[Path]:
    base = Path(cfg.input_dir)
    if cfg.file_names:
        files = [base / name for name in cfg.file_names]
    else:
        files = sorted(base.glob(cfg.file_glob))
    return [p for p in files if p.exists()]


def parse_event_time(series: pd.Series, dt_format: str | None = None) -> pd.Series:
    if dt_format:
        return pd.to_datetime(series, format=dt_format, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def load_excel_with_month(path: Path, cfg: InputConfig) -> pd.DataFrame:
    df = pd.read_excel(path)
    if cfg.datetime_column not in df.columns:
        raise ValueError(
            f"No datetime column '{cfg.datetime_column}' in {path.name}. "
            f"Available: {list(df.columns)[:20]}"
        )

    df["event_time"] = parse_event_time(df[cfg.datetime_column], cfg.datetime_format)
    bad = int(df["event_time"].isna().sum())
    if bad:
        raise ValueError(
            f"Cannot parse {bad} values in datetime column '{cfg.datetime_column}' for file {path.name}. "
            "Expected values like '2025-01-09 12:55:29'."
        )

    df["month"] = df["event_time"].dt.strftime("%Y-%m")
    df["source_file"] = path.name
    return df


def read_all_excels(cfg: InputConfig) -> pd.DataFrame:
    files = discover_excel_files(cfg)
    total = len(files)
    frames = []
    for idx, path in enumerate(files, start=1):
        remaining = total - idx
        logger.info("[stage=prepare/read] loading file %s/%s: %s (remaining=%s)", idx, total, path.name, remaining)
        frames.append(load_excel_with_month(path, cfg))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
