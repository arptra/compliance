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

    def find_viz_tags(self) -> list[str]:
        return sorted(p.stem.replace("viz_state_", "") for p in self.paths.interim_dir.glob("viz_state_*.parquet"))

    def find_pattern_fit_tags(self) -> list[str]:
        return sorted(p.name.replace("pattern_fit_", "") for p in self.paths.interim_dir.glob("pattern_fit_*") if p.is_dir())

    def find_pattern_monitor_tags(self) -> list[str]:
        return sorted(p.name.replace("pattern_monitor_", "") for p in self.paths.interim_dir.glob("pattern_monitor_*") if p.is_dir())

    def load_viz_state(self, tag: str) -> pd.DataFrame:
        return self.read_parquet(self.paths.interim_dir / f"viz_state_{tag}.parquet")

    def load_prepare(self) -> pd.DataFrame:
        return self.read_parquet(self.paths.prepare_parquet)

    def load_pattern_fit_growth(self, tag: str) -> pd.DataFrame:
        return self.read_parquet(self.paths.interim_dir / f"pattern_fit_{tag}" / "category_growth_summary.parquet")

    def load_pattern_fit_seed_pool(self, tag: str) -> pd.DataFrame:
        return self.read_parquet(self.paths.interim_dir / f"pattern_fit_{tag}" / "seed_pool.parquet")

    def load_pattern_fit_clusters(self, tag: str) -> pd.DataFrame:
        return self.read_parquet(self.paths.interim_dir / f"pattern_fit_{tag}" / "cluster_members.parquet")

    def load_pattern_fit_profiles(self, tag: str) -> dict[str, Any]:
        return self.read_json(self.paths.interim_dir / f"pattern_fit_{tag}" / "cluster_profiles.json")

    def load_pattern_monitor_scored(self, tag: str) -> pd.DataFrame:
        return self.read_parquet(self.paths.interim_dir / f"pattern_monitor_{tag}" / "scored_rows.parquet")

    def load_pattern_monitor_pressure(self, tag: str) -> pd.DataFrame:
        return self.read_parquet(self.paths.interim_dir / f"pattern_monitor_{tag}" / "category_daily_pressure.parquet")

    def load_pattern_monitor_state(self, tag: str) -> pd.DataFrame:
        return self.read_parquet(self.paths.interim_dir / f"pattern_monitor_{tag}" / "overall_daily_state.parquet")
