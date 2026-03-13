from __future__ import annotations

from ..schemas import ClusterProfileResponse, PatternFitSummaryResponse
from .data_loader import DataLoader


class PatternFitService:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def summary(self, tag: str) -> PatternFitSummaryResponse:
        g = self.loader.load_pattern_fit_growth(tag)
        cats = sorted(g["category"].dropna().astype(str).unique().tolist()) if "category" in g.columns else []
        return PatternFitSummaryResponse(tag=tag, growth_summary=g.to_dict(orient="records"), categories=cats)

    def categories(self, tag: str) -> list[str]:
        return self.summary(tag).categories

    def category(self, tag: str, category: str) -> dict:
        g = self.loader.load_pattern_fit_growth(tag)
        if "category" in g.columns:
            g = g[g["category"] == category]
        return {"category": category, "rows": g.to_dict(orient="records")}

    def clusters(self, tag: str, category: str) -> ClusterProfileResponse:
        profiles = self.loader.load_pattern_fit_profiles(tag)
        clusters = profiles.get(category, []) if isinstance(profiles, dict) else []
        if not clusters:
            members = self.loader.load_pattern_fit_clusters(tag)
            if not members.empty and "category" in members.columns:
                clusters = members[members["category"] == category].head(200).to_dict(orient="records")
        return ClusterProfileResponse(category=category, clusters=clusters)

    def seeds(self, tag: str, category: str) -> dict:
        seeds = self.loader.load_pattern_fit_seed_pool(tag)
        if "category" in seeds.columns:
            seeds = seeds[seeds["category"] == category]
        return {"category": category, "rows": seeds.head(200).to_dict(orient="records")}
