import pandas as pd

from complaints_trends.viz.state import build_viz_state


def test_build_viz_state_has_expected_columns():
    df = pd.DataFrame(
        [
            {"event_time": "2025-01-01 10:00:00", "month": "2025-01", "is_complaint_flag": True, "category": "A", "subcategory": "a1"},
            {"event_time": "2025-01-01 11:00:00", "month": "2025-01", "is_complaint_flag": True, "category": "B", "subcategory": "b1"},
            {"event_time": "2025-01-01 12:00:00", "month": "2025-01", "is_complaint_flag": False, "category": "OTHER", "subcategory": "UNKNOWN"},
            {"event_time": "2025-01-02 10:00:00", "month": "2025-01", "is_complaint_flag": True, "category": "A", "subcategory": "a1"},
        ]
    )
    out = build_viz_state(df, label_source="pred", freq="D", top_n=2)
    for col in [
        "date",
        "month",
        "category",
        "subcategory",
        "metric_count",
        "metric_share",
        "metric_share_of_all",
        "label_source",
        "is_complaint_flag",
    ]:
        assert col in out.columns
    assert len(out) > 0


def test_build_viz_state_handles_empty_input():
    out = build_viz_state(pd.DataFrame(), label_source="llm", freq="D", top_n=3)
    assert len(out) == 0
    assert "metric_count" in out.columns
