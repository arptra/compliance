from pathlib import Path

import pandas as pd

from complaints_trends.viz.report import _merge_with_infer_month_predictions


def test_merge_with_infer_month_predictions_adds_month_rows_for_pred(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data/interim").mkdir(parents=True, exist_ok=True)

    base = pd.DataFrame(
        [
            {"row_id": "r1", "month": "2025-10", "event_time": "2025-10-01 10:00:00", "is_complaint_flag": True, "category": "A", "subcategory": "a"},
        ]
    )
    month_df = pd.DataFrame(
        [
            {"row_id": "new_0", "month": "2025-12", "event_time": "2025-12-05 12:00:00", "is_complaint_pred": True, "category_pred": "NEW_CAT", "subcategory_pred": "new_sub"},
        ]
    )
    month_df.to_parquet("data/interim/month_2025-12.parquet", index=False)

    merged, info = _merge_with_infer_month_predictions(base, "2025-12", "pred")
    assert info["infer_month_loaded"] is True
    assert info["infer_month_rows"] == 1
    assert (merged["month"] == "2025-12").any()
    assert (merged["category"] == "NEW_CAT").any()


def test_merge_with_infer_month_predictions_skips_for_llm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = pd.DataFrame(
        [
            {"row_id": "r1", "month": "2025-10", "event_time": "2025-10-01 10:00:00", "is_complaint_flag": True, "category": "A", "subcategory": "a"},
        ]
    )
    merged, info = _merge_with_infer_month_predictions(base, "2025-12", "llm")
    assert info["infer_month_loaded"] is False
    assert len(merged) == 1
