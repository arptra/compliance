import pandas as pd

from app.preprocess import TextPreprocessor


def _cfg():
    return {
        "preprocess": {
            "use_razdel": False,
            "lemmatize": False,
            "remove_placeholders": True,
            "remove_urls_emails_phones": True,
            "normalize_numbers": "remove",
            "min_token_len": 2,
            "max_token_len": 30,
            "drop_tokens_with_digits": True,
            "drop_tokens_with_underscores": True,
            "drop_repeated_char_tokens": True,
            "extra_stopwords_path": "configs/extra_stopwords.txt",
            "deny_tokens_path": "configs/deny_tokens.txt",
            "placeholder_patterns": [
                r"<[^>]{1,80}>",
                r"\*{2,}",
                r"(?i)\\b(x{2,}|х{2,}|_+|#+)\\b",
                r"\\b\\d{4,}\\b",
                r"\\b[0-9a-f]{8,}\\b",
                r"\\b[0-9a-f]{8}-[0-9a-f-]{8,}\\b",
                r"(?i)\\b(ticket|case|id|uuid|guid)\\b\\s*[:=]?\\s*[\\w-]+",
            ],
        }
    }


def test_clean_text_removes_urls_and_digits():
    p = TextPreprocessor(_cfg())
    text = "Смотрите https://test.ru и пишите x@y.com номер +79991234567, код 12345"
    cleaned = p.clean_text(text)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "12345" not in cleaned


def test_preprocess_series_stable():
    p = TextPreprocessor(_cfg())
    out = p.preprocess_series(pd.Series(["Не работает вход 12"]))
    assert len(out) == 1
    assert "работает" in out[0]


def test_placeholder_and_phone_removed_from_tokens():
    p = TextPreprocessor(_cfg())
    tokens = p.analyzer("верните деньги <phone> +7 999 123 45 67")
    assert "phone" not in " ".join(tokens)
    assert all(not any(ch.isdigit() for ch in t) for t in tokens)


def test_id_and_masking_artifacts_removed():
    p = TextPreprocessor(_cfg())
    tokens = p.analyzer("ID=ABCD1234-FFFF-9999 *** xxxx ####")
    assert tokens == []
