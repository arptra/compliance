from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatternFitPaths:
    tag: str
    interim_dir: str
    exports_dir: str
    reports_dir: str

    @property
    def root(self) -> Path:
        return Path(self.interim_dir) / f"pattern_fit_{self.tag}"

    @property
    def fit_meta(self) -> Path:
        return self.root / "fit_meta.json"

    @property
    def fit_bundle(self) -> Path:
        return self.root / "fit_bundle.joblib"

    @property
    def growth_summary(self) -> Path:
        return self.root / "category_growth_summary.parquet"

    @property
    def report(self) -> Path:
        return Path(self.reports_dir) / f"pattern_fit_{self.tag}.html"

    @property
    def export(self) -> Path:
        return Path(self.exports_dir) / f"pattern_fit_{self.tag}.xlsx"


@dataclass(frozen=True)
class PatternMonitorPaths:
    tag: str
    interim_dir: str
    exports_dir: str
    reports_dir: str

    @property
    def root(self) -> Path:
        return Path(self.interim_dir) / f"pattern_monitor_{self.tag}"

    @property
    def report(self) -> Path:
        return Path(self.reports_dir) / f"pattern_monitor_{self.tag}.html"

    @property
    def export(self) -> Path:
        return Path(self.exports_dir) / f"pattern_monitor_{self.tag}.xlsx"
