from complaints_trends.api.services.model_quality_service import compute_precision_at_k, compute_precision_curve


def test_precision_at_k_real_top_k_logic():
    rows = [
        {"verdict": "true", "base_score": 0.95},
        {"verdict": "false", "base_score": 0.92},
        {"verdict": "true", "base_score": 0.91},
    ]
    assert compute_precision_at_k(rows, 2, "base_score") == 0.5
    assert compute_precision_at_k(rows, 10, "base_score") == 2 / 3


def test_precision_curve_points():
    rows = [{"verdict": "true", "base_score": 0.5}]
    curve = compute_precision_curve(rows, [10, 20], "base_score")
    assert [p["k"] for p in curve] == [10, 20]
    assert curve[0]["precision"] == 1.0
