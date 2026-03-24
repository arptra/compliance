from __future__ import annotations

from typing import Any


class TaxonomyLabelService:
    def __init__(self, category_labels: dict[str, str], subcategory_labels: dict[str, dict[str, str]]) -> None:
        self.category_labels = category_labels
        self.subcategory_labels = subcategory_labels

    def category_label_ru(self, category: str | None) -> str | None:
        if category is None:
            return None
        key = str(category)
        return self.category_labels.get(key, key)

    def subcategory_label_ru(self, category: str | None, subcategory: str | None) -> str | None:
        if subcategory is None:
            return None
        category_key = str(category or "")
        subcategory_key = str(subcategory)
        scoped = self.subcategory_labels.get(category_key, {})
        return scoped.get(subcategory_key, subcategory_key)

    def enrich_row(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        category = out.get("category")
        subcategory = out.get("subcategory")
        out["category_label_ru"] = self.category_label_ru(category)
        out["subcategory_label_ru"] = self.subcategory_label_ru(category, subcategory)
        return out

    def enrich_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.enrich_row(row) for row in rows]
