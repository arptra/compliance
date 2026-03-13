from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ...config import ProjectConfig


@dataclass
class ArtifactPaths:
    prepare_parquet: Path
    interim_dir: Path


class DataLoader:
    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self.paths = ArtifactPaths(
            prepare_parquet=Path(cfg.prepare.output_parquet),
            interim_dir=Path(cfg.analysis.pattern_monitoring.interim_dir),
        )
        self._cache: dict[str, tuple[float, Any]] = {}

    def _load_cached(self, path: Path, loader) -> Any:
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        key = str(path)
        cached = self._cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        value = loader(path)
        self._cache[key] = (mtime, value)
        return value

    def read_parquet(self, path: Path) -> pd.DataFrame:
        df = self._load_cached(path, pd.read_parquet)
        if df is None:
            return pd.DataFrame()
        return df.copy()

    def read_json(self, path: Path) -> dict[str, Any]:
        data = self._load_cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))
        return dict(data or {})

    def read_joblib(self, path: Path) -> Any:
        return self._load_cached(path, joblib.load)

    def _normalize_prepare_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()

        # Normalize category/subcategory naming from prepare artifacts.
        if "category" not in out.columns and "complaint_category_llm" in out.columns:
            out["category"] = out["complaint_category_llm"]
        if "subcategory" not in out.columns and "complaint_subcategory_llm" in out.columns:
            out["subcategory"] = out["complaint_subcategory_llm"]

        # Normalize complaint flag naming.
        if "is_complaint_flag" not in out.columns and "is_complaint_llm" in out.columns:
            out["is_complaint_flag"] = out["is_complaint_llm"]

        # Normalize date column naming.
        if "event_time" not in out.columns and "created_at" in out.columns:
            out["event_time"] = out["created_at"]

        if "category" in out.columns:
            out["category"] = out["category"].fillna("UNKNOWN").astype(str)
        if "subcategory" in out.columns:
            out["subcategory"] = out["subcategory"].fillna("UNKNOWN").astype(str)
        return out

    def find_viz_tags(self) -> list[str]:
        return sorted(p.stem.replace("viz_state_", "") for p in self.paths.interim_dir.glob("viz_state_*.parquet"))

    def find_pattern_fit_tags(self) -> list[str]:
        return sorted(p.name.replace("pattern_fit_", "") for p in self.paths.interim_dir.glob("pattern_fit_*") if p.is_dir())

    def find_pattern_monitor_tags(self) -> list[str]:
        return sorted(p.name.replace("pattern_monitor_", "") for p in self.paths.interim_dir.glob("pattern_monitor_*") if p.is_dir())

    def resolve_tag(self, family: str, tag: str) -> str:
        if tag != "latest":
            return tag
        if family == "pattern_fit":
            tags = self.find_pattern_fit_tags()
        elif family == "pattern_monitor":
            tags = self.find_pattern_monitor_tags()
        elif family == "viz":
            tags = self.find_viz_tags()
        else:
            return tag
        return tags[-1] if tags else tag

    def _normalize_common_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "date" not in out.columns:
            for candidate in ("event_date", "event_time", "created_at", "timestamp"):
                if candidate in out.columns:
                    out["date"] = out[candidate]
                    break
        if "category" not in out.columns and "complaint_category_llm" in out.columns:
            out["category"] = out["complaint_category_llm"]
        if "subcategory" not in out.columns and "complaint_subcategory_llm" in out.columns:
            out["subcategory"] = out["complaint_subcategory_llm"]
        if "category" in out.columns:
            out["category"] = out["category"].fillna("UNKNOWN").astype(str)
        if "subcategory" in out.columns:
            out["subcategory"] = out["subcategory"].fillna("UNKNOWN").astype(str)
        return out

    def load_viz_state(self, tag: str) -> pd.DataFrame:
        return self.read_parquet(self.paths.interim_dir / f"viz_state_{tag}.parquet")

    def load_prepare(self) -> pd.DataFrame:
        return self._normalize_prepare_columns(self.read_parquet(self.paths.prepare_parquet))

    def load_pattern_fit_growth(self, tag: str) -> pd.DataFrame:
        resolved = self.resolve_tag("pattern_fit", tag)
        return self._normalize_common_columns(self.read_parquet(self.paths.interim_dir / f"pattern_fit_{resolved}" / "category_growth_summary.parquet"))

    def load_pattern_fit_seed_pool(self, tag: str) -> pd.DataFrame:
        resolved = self.resolve_tag("pattern_fit", tag)
        return self._normalize_common_columns(self.read_parquet(self.paths.interim_dir / f"pattern_fit_{resolved}" / "seed_pool.parquet"))

    def load_pattern_fit_clusters(self, tag: str) -> pd.DataFrame:
        resolved = self.resolve_tag("pattern_fit", tag)
        return self._normalize_common_columns(self.read_parquet(self.paths.interim_dir / f"pattern_fit_{resolved}" / "cluster_members.parquet"))

    def load_pattern_fit_profiles(self, tag: str) -> dict[str, Any]:
        resolved = self.resolve_tag("pattern_fit", tag)
        return self.read_json(self.paths.interim_dir / f"pattern_fit_{resolved}" / "cluster_profiles.json")

    def load_pattern_monitor_scored(self, tag: str) -> pd.DataFrame:
        resolved = self.resolve_tag("pattern_monitor", tag)
        return self._normalize_common_columns(self.read_parquet(self.paths.interim_dir / f"pattern_monitor_{resolved}" / "scored_rows.parquet"))

    def load_pattern_monitor_pressure(self, tag: str) -> pd.DataFrame:
        resolved = self.resolve_tag("pattern_monitor", tag)
        return self._normalize_common_columns(self.read_parquet(self.paths.interim_dir / f"pattern_monitor_{resolved}" / "category_daily_pressure.parquet"))

    def load_pattern_monitor_state(self, tag: str) -> pd.DataFrame:
        resolved = self.resolve_tag("pattern_monitor", tag)
        return self._normalize_common_columns(self.read_parquet(self.paths.interim_dir / f"pattern_monitor_{resolved}" / "overall_daily_state.parquet"))
