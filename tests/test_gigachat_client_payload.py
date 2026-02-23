import json

from complaints_trends.config import LLMConfig
from complaints_trends.gigachat_mtls import GigaChatNormalizer


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class DummyClient:
    def chat(self, payload):
        assert isinstance(payload, dict)
        assert "messages" in payload
        assert "model" in payload
        body = {
            "client_first_message": "тест",
            "short_summary": "тест",
            "is_complaint": False,
            "complaint_category": "OTHER",
            "complaint_subcategory": None,
            "product_area": None,
            "loan_product": "NONE",
            "severity": "low",
            "keywords": ["вопрос", "инфо", "уточнение"],
            "confidence": 0.9,
            "notes": None,
        }
        return _Resp(json.dumps(body, ensure_ascii=False))


class FailTLSClient:
    def chat(self, payload):
        raise RuntimeError("TLSV13_ALERT_CERTIFICATE_REQUIRED")


def test_normalizer_uses_payload_dict_for_chat(tmp_path):
    cfg = LLMConfig(
        enabled=True,
        mode="mtls",
        base_url="https://x",
        ca_bundle_file="ca.pem",
        cert_file="cert.pem",
        key_file="key.pem",
        verify_ssl_certs=True,
        model="GigaChat",
        cache_db=str(tmp_path / "cache.sqlite"),
    )
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n.mock = False
    n.client = DummyClient()
    out = n.normalize({"client_first_message": "привет"})
    assert out.complaint_category == "OTHER"


def test_tls_mode_initializes_client_without_mtls_files(monkeypatch, tmp_path):
    captured = {}

    class FakeGigaChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("complaints_trends.gigachat_mtls.GigaChat", FakeGigaChat)

    cfg = LLMConfig(
        enabled=True,
        mode="tls",
        base_url="https://x",
        verify_ssl_certs=True,
        model="GigaChat",
        cache_db=str(tmp_path / "cache.sqlite"),
    )
    GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=False)

    assert captured["base_url"] == "https://x"
    assert "cert_file" not in captured
    assert "key_file" not in captured


def test_tls_mode_raises_explicit_message_when_server_requires_client_cert(tmp_path):
    cfg = LLMConfig(
        enabled=True,
        mode="tls",
        base_url="https://x",
        verify_ssl_certs=True,
        model="GigaChat",
        cache_db=str(tmp_path / "cache.sqlite"),
    )
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n.mock = False
    n.client = FailTLSClient()
    try:
        n.normalize({"client_first_message": "привет"})
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "Switch llm.mode to mtls" in str(e)
