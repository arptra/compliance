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
    out = n.normalize({"full_dialog_text": "привет"})
    assert out.complaint_category == "OTHER"


def test_tls_mode_initializes_httpx_client_without_mtls_files(monkeypatch, tmp_path):
    captured = {}

    class FakeHTTPXClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def post(self, *args, **kwargs):
            raise AssertionError("not expected")

    monkeypatch.setattr("complaints_trends.gigachat_mtls.httpx.Client", FakeHTTPXClient)

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
    assert captured["verify"] is True
    assert captured["trust_env"] is False


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
        n.normalize({"full_dialog_text": "привет"})
        assert False, "Expected RuntimeError"
    except RuntimeError as e:
        assert "Switch llm.mode to mtls" in str(e)


def test_mtls_mode_uses_ssl_context_for_httpx_client(monkeypatch, tmp_path):
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

    class FakeHTTPXClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def post(self, *args, **kwargs):
            raise AssertionError("not expected")

    monkeypatch.setattr("complaints_trends.gigachat_mtls.ssl.create_default_context", fake_create_default_context)
    monkeypatch.setattr("complaints_trends.gigachat_mtls.httpx.Client", FakeHTTPXClient)

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

    assert captured["verify"] is fake_context
    assert captured["cafile"] == str(ca)
    assert fake_context.loaded == (str(cert), str(key), None)



def test_compact_api_response_is_coerced_to_schema(tmp_path):
    class CompactClient:
        def chat(self, payload):
            body = {
                "category": "CREDITING",
                "product": "CONSUMER_LOAN",
            }
            return _Resp(json.dumps(body, ensure_ascii=False))

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
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER", "CREDITING"], "subcategories_by_category": {"OTHER": [], "CREDITING": []}, "loan_products": ["NONE", "CONSUMER_LOAN"]}, mock=True)
    n.mock = False
    n.client = CompactClient()
    out = n.normalize({"full_dialog_text": "Не получается оплатить"})
    assert out.complaint_category == "CREDITING"
    assert out.loan_product == "CONSUMER_LOAN"
    assert out.client_first_message == "Не получается оплатить"


def test_request_metrics_can_be_disabled(tmp_path):
    class NoMetricsClient:
        def count_tokens(self, *, model, input_text):
            raise AssertionError("count_tokens must not be called when request_metrics_enabled=false")

        def chat(self, payload):
            body = {
                "complaint_category": "OTHER",
                "is_complaint": False,
                "loan_product": "NONE",
                "severity": "low",
                "keywords": ["вопрос", "инфо", "уточнение"],
                "confidence": 0.9,
            }
            return _Resp(json.dumps(body, ensure_ascii=False))

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
        request_metrics_enabled=False,
    )
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n.mock = False
    n.client = NoMetricsClient()

    out = n.normalize({"full_dialog_text": "привет"})
    assert out.complaint_category == "OTHER"


def test_discover_mode_prompt_contains_existing_categories(tmp_path):
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
        category_mode="discover",
    )
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n.discovered_categories = ["payments_issue"]
    p = n._single_user_prompt({"full_dialog_text": "text"})
    assert "normalize_ticket_discover_categories" in p
    assert "payments_issue" in p


def test_discover_mode_remembers_new_categories(tmp_path):
    class DiscoverClient:
        def chat(self, payload):
            body = {
                "category": "new billing issue",
                "is_complaint": True,
                "loan_product": "NONE",
                "severity": "medium",
                "keywords": ["ошибка", "оплата", "billing"],
                "confidence": 0.8,
            }
            return _Resp(json.dumps(body, ensure_ascii=False))

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
        category_mode="discover",
    )
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n.mock = False
    n.client = DiscoverClient()
    out = n.normalize({"full_dialog_text": "проблема с биллингом"})
    assert out.complaint_category == "new billing issue"
    assert "new_billing_issue" in n.discovered_categories
