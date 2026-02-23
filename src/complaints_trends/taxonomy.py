from __future__ import annotations

from pathlib import Path

import yaml


def load_taxonomy(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    categories = data.get("categories", {}) or {}
    loan_products = data.get("loan_products", {}) or {}

    category_codes = list(categories.keys())
    subcategories_by_category: dict[str, list[str]] = {}

    for cat, body in categories.items():
        if isinstance(body, dict):
            sub = body.get("subcategories", {})
            if isinstance(sub, dict):
                subcategories_by_category[cat] = list(sub.keys())
            elif isinstance(sub, list):
                subcategories_by_category[cat] = [str(x) for x in sub]
            else:
                subcategories_by_category[cat] = []
        else:
            subcategories_by_category[cat] = []

    return {
        "category_codes": category_codes,
        "subcategories_by_category": subcategories_by_category,
        "loan_products": list(loan_products.keys()),
        "raw": data,
    }
