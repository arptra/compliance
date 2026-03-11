import json
from pathlib import Path

from complaints_trends.config import LLMConfig
from complaints_trends.gigachat_mtls import GigaChatNormalizer
from complaints_trends.questions_loader import load_questions


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
    n.discovered_subcategories_by_category = {"payments_issue": ["card_declined"]}
    p = n._single_user_prompt({"full_dialog_text": "text"})
    assert "discover_and_normalize_ticket" in p
    assert "you_must_discover_categories_yourself" in p
    assert "payments_issue" in p
    assert "card_declined" in p
    assert "allowed_categories" not in p


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
    discovered_file = tmp_path / "disc.json"
    cfg2 = cfg.model_copy(deep=True)
    cfg2.discovered_taxonomy_file = str(discovered_file)
    n2 = GigaChatNormalizer(cfg2, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n2.mock = False
    n2.client = DiscoverClient()
    n2.normalize({"full_dialog_text": "другая проблема"})
    assert discovered_file.exists()
    data = json.loads(discovered_file.read_text(encoding="utf-8"))
    assert "new_billing_issue" in data.get("categories", [])


def test_discover_mode_creates_empty_discovery_file_when_missing(tmp_path):
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
        discovered_taxonomy_file=str(tmp_path / "disc.json"),
    )
    GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    assert (tmp_path / "disc.json").exists()
    data = json.loads((tmp_path / "disc.json").read_text(encoding="utf-8"))
    assert data.get("categories") == []


def test_questions_mode_prompt_and_mapping_file(tmp_path):
    qf = tmp_path / "questions.json"
    qf.write_text(
        json.dumps(
            {
                "version": 1,
                "categories": [{"question_ru": "Есть ли жалоба на переводы?"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
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
        category_mode="questions",
        questions_file=str(qf),
    )
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    p = n._single_user_prompt({"full_dialog_text": "text"})
    assert "questionnaire_normalize_ticket" in p
    assert "allowed_categories" in p
    assert "OTHER" in p
    assert Path("data/interim/questions_taxonomy.json").exists()


def test_questions_mode_cache_key_depends_on_questions_hash(tmp_path):
    q1 = tmp_path / "q1.json"
    q2 = tmp_path / "q2.json"
    q1.write_text(json.dumps({"version": 1, "categories": [{"question_ru": "Q1?"}]}, ensure_ascii=False), encoding="utf-8")
    q2.write_text(json.dumps({"version": 1, "categories": [{"question_ru": "Q2?"}]}, ensure_ascii=False), encoding="utf-8")

    base_kwargs = dict(
        enabled=True,
        mode="mtls",
        base_url="https://x",
        ca_bundle_file="ca.pem",
        cert_file="cert.pem",
        key_file="key.pem",
        verify_ssl_certs=True,
        model="GigaChat",
        cache_db=str(tmp_path / "cache.sqlite"),
        category_mode="questions",
    )
    cfg1 = LLMConfig(**base_kwargs, questions_file=str(q1))
    cfg2 = LLMConfig(**base_kwargs, questions_file=str(q2))
    n1 = GigaChatNormalizer(cfg1, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n2 = GigaChatNormalizer(cfg2, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    payload = {"full_dialog_text": "x"}
    assert n1._key(payload) != n2._key(payload)


def test_questions_mode_batch_prompt_is_used(tmp_path):
    class BatchClient:
        def chat(self, payload):
            user = payload["messages"][1]["content"]
            assert "questionnaire_normalize_batch" in user
            body = {
                "items": [
                    {
                        "_batch_index": 0,
                        "primary_category_code": "OTHER",
                        "triggered_codes": [],
                        "keywords": ["вопрос", "инфо", "уточнение"],
                    }
                ]
            }
            return _Resp(json.dumps(body, ensure_ascii=False))

    qf = tmp_path / "questions.json"
    qf.write_text(json.dumps({"version": 1, "categories": [{"question_ru": "Есть ли жалоба на переводы?"}]}, ensure_ascii=False), encoding="utf-8")
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
        category_mode="questions",
        questions_file=str(qf),
    )
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n.mock = False
    n.client = BatchClient()
    out = n.normalize_batch([{"full_dialog_text": "test"}])
    assert len(out) == 1
    assert out[0].complaint_category == "OTHER"
    assert out[0].is_complaint is False


def test_questions_mode_learns_category_name_per_question_and_reuses_it(tmp_path):
    q_code_holder = {"code": ""}

    class QClient:
        def __init__(self):
            self.n = 0

        def chat(self, payload):
            self.n += 1
            # second response tries to rename same question category, should be ignored/reused
            cat_name = "Проблема с переводом" if self.n == 1 else "Совсем другое имя"
            q_code = q_code_holder["code"]
            body = {
                "primary_question_code": q_code,
                "category_name": cat_name,
                "triggered_codes": [q_code],
                "keywords": ["перевод", "ошибка", "жалоба"],
            }
            return _Resp(json.dumps(body, ensure_ascii=False))

    qf = tmp_path / "questions.json"
    qf.write_text(
        json.dumps(
            {
                "version": 1,
                "categories": [{"question_ru": "Есть ли жалоба на переводы?"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    q_code_holder["code"] = load_questions(qf)["items"][0]["code"]
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
        category_mode="questions",
        questions_file=str(qf),
    )
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n.mock = False
    n.client = QClient()

    out1 = n.normalize({"full_dialog_text": "перевод не прошел"})
    out2 = n.normalize({"full_dialog_text": "еще одна проблема с переводом"})
    assert out1.is_complaint is True
    assert out2.is_complaint is True
    assert out1.complaint_category == out2.complaint_category
    state = n.export_questions_categories_json()
    assert q_code_holder["code"] in state["question_category_map"]


def test_questions_mode_uses_short_name_when_model_repeats_question_text(tmp_path):
    q_code_holder = {"code": ""}

    class QClient:
        def chat(self, payload):
            q_code = q_code_holder["code"]
            body = {
                "primary_question_code": q_code,
                "category_name": "Есть ли жалоба на переводы?",
                "triggered_codes": [q_code],
                "keywords": ["перевод", "ошибка", "жалоба"],
            }
            return _Resp(json.dumps(body, ensure_ascii=False))

    qf = tmp_path / "questions.json"
    qf.write_text(json.dumps({"version": 1, "categories": [{"question_ru": "Есть ли жалоба на переводы?"}]}, ensure_ascii=False), encoding="utf-8")
    q_code_holder["code"] = load_questions(qf)["items"][0]["code"]

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
        category_mode="questions",
        questions_file=str(qf),
    )
    n = GigaChatNormalizer(cfg, {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]}, mock=True)
    n.mock = False
    n.client = QClient()

    out = n.normalize({"full_dialog_text": "перевод не прошел"})
    assert out.is_complaint is True
    state = n.export_questions_categories_json()
    qmap = state["question_category_map"][q_code_holder["code"]]
    assert qmap["category_name"].strip().lower() != "есть ли жалоба на переводы?"
