from pathlib import Path

import numpy as np
import pandas as pd

from complaints_trends.compare import compare_month
from complaints_trends.config import load_config
from complaints_trends.train_models import _load_effective_taxonomy, _normalize_subcategory_by_taxonomy


def test_load_effective_taxonomy_merges_discovered_file(tmp_path):
    cfg = load_config("configs/project.yaml").model_copy(deep=True)
    disc = tmp_path / "discovered.json"
    disc.write_text(
        '{"categories": ["new_cat"], "subcategories_by_category": {"new_cat": ["new_sub"]}}',
        encoding="utf-8",
    )
    cfg.llm.discovered_taxonomy_file = str(disc)

    taxonomy = _load_effective_taxonomy(cfg)
    assert "new_cat" in taxonomy["category_codes"]
    assert "new_sub" in taxonomy["subcategories_by_category"]["new_cat"]


def test_normalize_subcategory_keeps_value_for_discovered_category_without_allowlist():
    df = pd.DataFrame(
        [
            {"complaint_category_llm": "discovered_cat", "complaint_subcategory_llm": "fresh_sub"},
            {"complaint_category_llm": "discovered_cat", "complaint_subcategory_llm": ""},
        ]
    )
    taxonomy = {"subcategories_by_category": {"TECHNICAL": ["login_issue"]}}
    out = _normalize_subcategory_by_taxonomy(df, taxonomy)
    assert out.tolist() == ["fresh_sub", "UNKNOWN"]


def test_compare_month_builds_category_and_subcategory_tables_for_dynamic_labels(tmp_path, monkeypatch):
    cfg = load_config("configs/project.yaml").model_copy(deep=True)
    cfg.prepare.output_parquet = str(tmp_path / "prepared.parquet")
    cfg.training.model_dir = str(tmp_path / "models")

    base = pd.DataFrame(
        [
            {"month": "2025-01", "is_complaint_llm": True, "complaint_category_llm": "legacy_cat", "complaint_subcategory_llm": "legacy_sub", "client_first_message": "a"},
            {"month": "2025-01", "is_complaint_llm": True, "complaint_category_llm": "new_dynamic", "complaint_subcategory_llm": "new_sub", "client_first_message": "b"},
        ]
    )
    base.to_parquet(cfg.prepare.output_parquet, index=False)

    new_m = pd.DataFrame(
        [
            {"is_complaint_pred": True, "category_pred": "new_dynamic", "subcategory_pred": "new_sub", "client_first_message": "x"},
            {"is_complaint_pred": True, "category_pred": "other_dynamic", "subcategory_pred": "other_sub", "client_first_message": "y"},
        ]
    )
    month_parq = Path("data/interim/month_2025-02.parquet")
    month_parq.parent.mkdir(parents=True, exist_ok=True)
    new_m.to_parquet(month_parq, index=False)

    class _Vec:
        def transform(self, s):
            return np.zeros((len(s), 2))

    monkeypatch.setattr("complaints_trends.compare.joblib.load", lambda _: _Vec())
    monkeypatch.setattr(
        "complaints_trends.compare.compute_novelty_scores",
        lambda xb, xn, **kwargs: (np.zeros(xn.shape[0]), 0.5, np.zeros((xn.shape[0], 2))),
    )
    monkeypatch.setattr(
        "complaints_trends.compare.cluster_novel_texts",
        lambda z, mask: np.array([-1] * len(mask)),
    )

    out = compare_month(cfg, "2025-02", "2025-01..2025-01")
    assert len(out) == 2

    html = Path("reports/compare_2025-02_vs_baseline.html").read_text(encoding="utf-8")
    assert "Subcategory share delta" in html
    assert "other_dynamic" in html
    assert "other_sub" in html
