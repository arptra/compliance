from app.extract import extract_first_client_message


def test_extract_first_client_ru_en_markers():
    cfg = {
        "input": {
            "role_parsing": {
                "enabled": True,
                "client_markers": ["CLIENT", "КЛИЕНТ"],
                "client_prefix_regexes": [r"(?im)^\s*(CLIENT|КЛИЕНТ)\s*[:\-]\s*"],
                "stop_on_markers": ["OPERATOR", "ОПЕРАТОР", "CHATBOT", "БОТ"],
                "fallback_mode": "first_paragraph",
            }
        }
    }
    text = "CHATBOT: hello\nCLIENT: не могу войти\nOPERATOR: уточните номер"
    out = extract_first_client_message(text, cfg)
    assert "не могу войти" in out
    assert "OPERATOR" not in out


def test_extract_fallback_first_paragraph():
    cfg = {"input": {"role_parsing": {"enabled": True, "fallback_mode": "first_paragraph"}}}
    text = "первый абзац\n\nвторой абзац"
    assert extract_first_client_message(text, cfg) == "первый абзац"
