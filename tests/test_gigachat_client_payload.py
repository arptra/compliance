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


def test_mtls_mode_uses_ssl_context_for_client(monkeypatch, tmp_path):
    ca = tmp_path / "ca.pem"
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    ca.write_text("ca", encoding="utf-8")
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")

    captured = {}

    class FakeContext:
        def __init__(self):
            self.loaded = None
            self.check_hostname = True
            self.verify_mode = None

        def load_cert_chain(self, certfile, keyfile, password=None):
            self.loaded = (certfile, keyfile, password)

    fake_context = FakeContext()

    def fake_create_default_context(*, cafile=None):
        captured["cafile"] = cafile
        return fake_context

    class FakeGigaChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("complaints_trends.gigachat_mtls.ssl.create_default_context", fake_create_default_context)
    monkeypatch.setattr("complaints_trends.gigachat_mtls.GigaChat", FakeGigaChat)

    cfg = LLMConfig(
        enabled=True,
        mode="mtls",
        base_url="https://x",
        ca_bundle_file=str(ca),
        cert_file=str(cert),
        key_file=str(key),
        verify_ssl_certs=True,
        model="GigaChat",
        cache_db=str(tmp_path / "cache.sqlite"),
    )

    GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=False)

    assert captured["ssl_context"] is fake_context
    assert "cert_file" not in captured
    assert "key_file" not in captured
    assert captured["cafile"] == str(ca)
    assert fake_context.loaded == (str(cert), str(key), None)



def test_client_forces_no_env_auth_credentials_in_mtls(monkeypatch, tmp_path):
    ca = tmp_path / "ca.pem"
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    ca.write_text("ca", encoding="utf-8")
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")

    captured = {}

    class FakeContext:
        def load_cert_chain(self, certfile, keyfile, password=None):
            return None

    def fake_create_default_context(*, cafile=None):
        return FakeContext()

    class FakeGigaChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("complaints_trends.gigachat_mtls.ssl.create_default_context", fake_create_default_context)
    monkeypatch.setattr("complaints_trends.gigachat_mtls.GigaChat", FakeGigaChat)
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "should_not_be_used")

    cfg = LLMConfig(
        enabled=True,
        mode="mtls",
        base_url="https://x",
        ca_bundle_file=str(ca),
        cert_file=str(cert),
        key_file=str(key),
        verify_ssl_certs=True,
        cache_db=str(tmp_path / "cache.sqlite"),
    )

    GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=False)

    assert captured["credentials"] == ""
    assert captured["user"] == ""
    assert captured["password"] == ""
