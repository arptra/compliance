from __future__ import annotations

from pathlib import Path

import yaml


def load_taxonomy(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    categories = data.get("categories", {}) or {}
    loan_products = data.get("loan_products", {}) or {}

    category_codes = list(categories.keys())
    subcategories_by_category: dict[str, list[str]] = {}
    category_labels: dict[str, str] = {}
    subcategory_labels: dict[str, dict[str, str]] = {}
    loan_product_labels: dict[str, str] = {}

    for cat, body in categories.items():
        category_labels[cat] = body.get("label_ru", cat) if isinstance(body, dict) else cat
        if isinstance(body, dict):
            sub = body.get("subcategories", {})
            if isinstance(sub, dict):
                subcategories_by_category[cat] = list(sub.keys())
                subcategory_labels[cat] = {
                    code: (meta.get("label_ru", code) if isinstance(meta, dict) else str(meta))
                    for code, meta in sub.items()
                }
            elif isinstance(sub, list):
                subcategories_by_category[cat] = [str(x) for x in sub]
                subcategory_labels[cat] = {str(x): str(x) for x in sub}
            else:
                subcategories_by_category[cat] = []
                subcategory_labels[cat] = {}
        else:
            subcategories_by_category[cat] = []
            subcategory_labels[cat] = {}

    for code, meta in loan_products.items():
        loan_product_labels[code] = meta.get("label_ru", code) if isinstance(meta, dict) else code

    return {
        "category_codes": category_codes,
        "subcategories_by_category": subcategories_by_category,
        "loan_products": list(loan_products.keys()),
        "category_labels": category_labels,
        "subcategory_labels": subcategory_labels,
        "loan_product_labels": loan_product_labels,
        "raw": data,
    }
