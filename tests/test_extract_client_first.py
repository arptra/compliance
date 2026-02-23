from complaints_trends.config import ClientFirstConfig
from complaints_trends.extract_client_first import extract_client_first_message


def test_extract_client_first_message():
    cfg = ClientFirstConfig(
        client_markers=["CLIENT"], operator_markers=["OPERATOR"], chatbot_markers=["CHATBOT"], stop_on_markers=["OPERATOR"]
    )
    dialog = "CHATBOT: Привет\nCLIENT: Не работает платеж\nOPERATOR: Приняли"
    assert extract_client_first_message(dialog, cfg) == "Не работает платеж"
