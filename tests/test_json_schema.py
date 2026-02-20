import pytest

from complaints_trends.gigachat_schema import NormalizeTicket


def test_schema_forbidden_keyword():
    with pytest.raises(Exception):
        NormalizeTicket(
            client_first_message="x",
            short_summary="x",
            is_complaint=True,
            complaint_category="OTHER",
            complaint_subcategory=None,
            product_area=None,
            severity="low",
            keywords=["client", "a", "b"],
            confidence=0.5,
            notes=None,
        )
