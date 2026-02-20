import pandas as pd

from complaints_trends.config import ClientFirstConfig
from complaints_trends.extract_client_first import extract_client_first_message
from complaints_trends.prepare_dataset import _select_primary_dialog


def test_select_primary_dialog_picks_longest_non_empty_and_extracts_client_first():
    row = pd.Series(
        {
            "dialog_text": "",
            "call_text": "CLIENT: Не могу войти в приложение\nOPERATOR: Проверю",
            "comment_text": "коротко",
            "summary_text": None,
        }
    )
    field, txt, ctx = _select_primary_dialog(row, ["dialog_text", "call_text", "comment_text", "summary_text"])
    assert field == "call_text"
    assert "Не могу войти" in txt
    assert set(ctx.keys()) == {"call_text", "comment_text"}

    cfg = ClientFirstConfig(
        client_markers=["CLIENT"],
        operator_markers=["OPERATOR"],
        chatbot_markers=["CHATBOT"],
        stop_on_markers=["OPERATOR"],
    )
    assert extract_client_first_message(txt, cfg) == "Не могу войти в приложение"
