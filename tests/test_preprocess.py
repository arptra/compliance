from app.preprocess import TextPreprocessor


def test_clean_text_removes_urls_and_digits():
    cfg = {"preprocess": {"use_razdel": False, "lemmatize": False, "max_token_len": 30}}
    p = TextPreprocessor(cfg)
    text = "Смотрите https://test.ru и пишите x@y.com номер +79991234567, код 12345"
    cleaned = p.clean_text(text)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "<num>" in cleaned


def test_preprocess_series_stable():
    cfg = {"preprocess": {"use_razdel": False, "lemmatize": False, "max_token_len": 30}}
    p = TextPreprocessor(cfg)
    out = p.preprocess_series(__import__("pandas").Series(["Не работает вход 12"]))
    assert len(out) == 1
    assert "работает" in out[0]
