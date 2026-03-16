from __future__ import annotations

from ..schemas import ClusterProfileResponse, PatternFitSummaryResponse
from .data_loader import DataLoader


class PatternFitService:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def _resolved_tag(self, tag: str) -> str:
        return self.loader.resolve_tag("pattern_fit", tag)

    def summary(self, tag: str) -> PatternFitSummaryResponse:
        resolved = self._resolved_tag(tag)
        g = self.loader.load_pattern_fit_growth(resolved)
        cats = sorted(g["category"].dropna().astype(str).unique().tolist()) if "category" in g.columns else []
        return PatternFitSummaryResponse(tag=resolved, growth_summary=g.to_dict(orient="records"), categories=cats)

    def categories(self, tag: str) -> list[str]:
        return self.summary(tag).categories

    def category(self, tag: str, category: str) -> dict:
        resolved = self._resolved_tag(tag)
        g = self.loader.load_pattern_fit_growth(resolved)
        if "category" in g.columns:
            g = g[g["category"] == category]
        return {"tag": resolved, "category": category, "rows": g.to_dict(orient="records")}

    def clusters(self, tag: str, category: str) -> ClusterProfileResponse:
        resolved = self._resolved_tag(tag)
        profiles = self.loader.load_pattern_fit_profiles(resolved)
        clusters = profiles.get(category, []) if isinstance(profiles, dict) else []
        if not clusters:
            members = self.loader.load_pattern_fit_clusters(resolved)
            if not members.empty and "category" in members.columns:
                clusters = members[members["category"] == category].head(200).to_dict(orient="records")
        return ClusterProfileResponse(category=category, clusters=clusters)

    def seeds(self, tag: str, category: str) -> dict:
        resolved = self._resolved_tag(tag)
        seeds = self.loader.load_pattern_fit_seed_pool(resolved)
        if "category" in seeds.columns:
            seeds = seeds[seeds["category"] == category]
        return {"tag": resolved, "category": category, "rows": seeds.head(200).to_dict(orient="records")}
