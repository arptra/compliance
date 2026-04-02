from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import pandas as pd
import pyarrow.parquet as pq

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
        self._cache: OrderedDict[str, tuple[tuple[int, int], Any]] = OrderedDict()
        self._lock = RLock()
        self._cache_max_items = 24
        self._prepare_cache: tuple[tuple[int, int], pd.DataFrame] | None = None
        self._prepare_columns_cache: tuple[tuple[int, int], set[str]] | None = None

    @staticmethod
    def _file_stamp(path: Path) -> tuple[int, int]:
        st = path.stat()
        return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size)

    def _cache_get(self, key: str, stamp: tuple[int, int]) -> Any | None:
        cached = self._cache.get(key)
        if not cached or cached[0] != stamp:
            return None
        self._cache.move_to_end(key)
        return cached[1]

    def _cache_set(self, key: str, stamp: tuple[int, int], value: Any) -> None:
        self._cache[key] = (stamp, value)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max_items:
            self._cache.popitem(last=False)

    def _load_cached(self, path: Path, cache_key: str, loader) -> Any:
        if not path.exists():
            return None
        stamp = self._file_stamp(path)
        with self._lock:
            cached_value = self._cache_get(cache_key, stamp)
            if cached_value is not None:
                return cached_value
        value = loader(path)
        with self._lock:
            self._cache_set(cache_key, stamp, value)
        return value

    def read_many_parquet(self, paths: list[Path]) -> dict[str, pd.DataFrame]:
        def _one(p: Path) -> tuple[str, pd.DataFrame]:
            return str(p), self.read_parquet(p)

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as ex:
            return dict(ex.map(_one, paths))

    def read_parquet(self, path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        projection = tuple(columns or ())
        cache_key = f"parquet::{path}::{','.join(projection)}"
        if projection:
            requested = list(projection)

            def loader(p: Path) -> pd.DataFrame:
                try:
                    return pd.read_parquet(p, columns=requested)
                except Exception:
                    available = set(pq.ParquetFile(p).schema.names)
                    present = [c for c in requested if c in available]
                    if not present:
                        return pd.DataFrame(columns=requested)
                    out = pd.read_parquet(p, columns=present)
                    for c in requested:
                        if c not in out.columns:
                            out[c] = pd.NA
                    return out[requested]
        else:
            loader = pd.read_parquet
        df = self._load_cached(path, cache_key, loader)
        if df is None:
            return pd.DataFrame()
        # shallow copy dramatically reduces latency/memory for big parquet reads
        return df.copy(deep=False)

    def read_json(self, path: Path) -> dict[str, Any]:
        data = self._load_cached(path, f"json::{path}", lambda p: json.loads(p.read_text(encoding="utf-8")))
        return dict(data or {})

    def read_joblib(self, path: Path) -> Any:
        return self._load_cached(path, f"joblib::{path}", joblib.load)

    def source_mtime(self, viz_tag: str | None = None) -> float:
        if viz_tag:
            p = self.paths.interim_dir / f"viz_state_{viz_tag}.parquet"
        else:
            p = self.paths.prepare_parquet
        return float(self._file_stamp(p)[0]) if p.exists() else 0.0

    def _normalize_prepare_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "category" not in out.columns and "complaint_category_llm" in out.columns:
            out["category"] = out["complaint_category_llm"]
        if "subcategory" not in out.columns and "complaint_subcategory_llm" in out.columns:
            out["subcategory"] = out["complaint_subcategory_llm"]
        if "is_complaint_flag" not in out.columns and "is_complaint_llm" in out.columns:
            out["is_complaint_flag"] = out["is_complaint_llm"]
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
        p = self.paths.prepare_parquet
        if not p.exists():
            return pd.DataFrame()
        stamp = self._file_stamp(p)
        with self._lock:
            if self._prepare_cache and self._prepare_cache[0] == stamp:
                return self._prepare_cache[1].copy(deep=False)
        normalized = self._normalize_prepare_columns(self.read_parquet(p))
        with self._lock:
            self._prepare_cache = (stamp, normalized)
        return normalized.copy(deep=False)

    def load_prepare_timeseries(self) -> pd.DataFrame:
        columns = ["event_time", "created_at", "date", "count", "metric_count", "category", "subcategory", "complaint_category_llm", "complaint_subcategory_llm"]
        if not self.paths.prepare_parquet.exists():
            return pd.DataFrame()
        stamp = self._file_stamp(self.paths.prepare_parquet)
        with self._lock:
            if self._prepare_columns_cache and self._prepare_columns_cache[0] == stamp:
                available = self._prepare_columns_cache[1]
            else:
                available = set(pq.ParquetFile(self.paths.prepare_parquet).schema_arrow.names)
                self._prepare_columns_cache = (stamp, available)
        present = [c for c in columns if c in available]
        if not present:
            return self.load_prepare()
        return self._normalize_prepare_columns(self.read_parquet(self.paths.prepare_parquet, columns=present))

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
