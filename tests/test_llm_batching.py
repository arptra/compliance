from complaints_trends.prepare_dataset import _split_payload_batches
from complaints_trends.config import LLMConfig
from complaints_trends.gigachat_mtls import GigaChatNormalizer


class _TokNormalizer:
    def __init__(self, token_map):
        self.token_map = token_map

    def estimate_tokens(self, payload):
        return self.token_map[payload["id"]]


def test_split_payload_batches_by_token_limit():
    n = _TokNormalizer({"a": 10, "b": 20, "c": 15})
    rows = [
        (0, {"id": "a"}),
        (1, {"id": "b"}),
        (2, {"id": "c"}),
    ]
    batches = _split_payload_batches(n, rows, token_batch_size=30)
    assert [[i for i, _ in b] for b in batches] == [[0, 1], [2]]


def test_normalize_batch_uses_single_chat_request(tmp_path):
    class BatchClient:
        def __init__(self):
            self.chat_calls = 0

        def count_tokens(self, *, model, input_text):
            return 100

        def chat(self, payload):
            self.chat_calls += 1
            assert payload["model"] == "GigaChat"
            content = (
                '[{"complaint_category":"OTHER","is_complaint":false,"loan_product":"NONE","severity":"low","keywords":["вопрос","инфо","уточнение"],"confidence":0.9},'
                '{"complaint_category":"OTHER","is_complaint":false,"loan_product":"NONE","severity":"low","keywords":["вопрос","инфо","уточнение"],"confidence":0.8}]'
            )

            class _Msg:
                def __init__(self, c):
                    self.content = c

            class _Choice:
                def __init__(self, c):
                    self.message = _Msg(c)

            class _Resp:
                def __init__(self, c):
                    self.choices = [_Choice(c)]

            return _Resp(content)

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
    n = GigaChatNormalizer(
        cfg,
        {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]},
        mock=True,
    )
    n.mock = False
    n.client = BatchClient()

    out = n.normalize_batch([
        {"full_dialog_text": "one", "dialog_context": {"dialog_text": "one"}},
        {"full_dialog_text": "two", "dialog_context": {"dialog_text": "two"}},
    ])
    assert len(out) == 2
    assert n.client.chat_calls == 1


def test_normalize_batch_retries_only_missing_rows(tmp_path):
    class PartialBatchClient:
        def __init__(self):
            self.chat_calls = 0

        def count_tokens(self, *, model, input_text):
            return 100

        def chat(self, payload):
            self.chat_calls += 1
            if self.chat_calls == 1:
                # first attempt returns only one record out of two
                content = '[{"_batch_index":0,"complaint_category":"OTHER","is_complaint":false,"loan_product":"NONE","severity":"low","keywords":["вопрос","инфо","уточнение"],"confidence":0.9}]'
            else:
                # retry should request and return only missing index=1
                content = '[{"_batch_index":1,"complaint_category":"OTHER","is_complaint":false,"loan_product":"NONE","severity":"low","keywords":["вопрос","инфо","уточнение"],"confidence":0.8}]'

            class _Msg:
                def __init__(self, c):
                    self.content = c

            class _Choice:
                def __init__(self, c):
                    self.message = _Msg(c)

            class _Resp:
                def __init__(self, c):
                    self.choices = [_Choice(c)]

            return _Resp(content)

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
    n = GigaChatNormalizer(
        cfg,
        {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]},
        mock=True,
    )
    n.mock = False
    n.client = PartialBatchClient()

    out = n.normalize_batch([
        {"full_dialog_text": "one", "dialog_context": {"dialog_text": "one"}},
        {"full_dialog_text": "two", "dialog_context": {"dialog_text": "two"}},
    ])
    assert len(out) == 2
    assert n.client.chat_calls == 2


def test_cache_threadsafe_for_parallel_mode(tmp_path):
    import concurrent.futures

    class FastClient:
        def chat(self, payload):
            content = '{"complaint_category":"OTHER","is_complaint":false,"loan_product":"NONE","severity":"low","keywords":["вопрос","инфо","уточнение"],"confidence":0.9}'

            class _Msg:
                def __init__(self, c):
                    self.content = c

            class _Choice:
                def __init__(self, c):
                    self.message = _Msg(c)

            class _Resp:
                def __init__(self, c):
                    self.choices = [_Choice(c)]

            return _Resp(content)

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
    n = GigaChatNormalizer(
        cfg,
        {"category_codes": ["OTHER"], "subcategories_by_category": {"OTHER": []}, "loan_products": ["NONE"]},
        mock=True,
    )
    n.mock = False
    n.client = FastClient()

    payloads = [{"full_dialog_text": f"msg-{i}", "dialog_context": {"dialog_text": f"msg-{i}"}} for i in range(20)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        outs = list(ex.map(n.normalize, payloads))

    assert len(outs) == 20
    assert all(o.complaint_category == "OTHER" for o in outs)
