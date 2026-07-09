from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path

from complaints_trends.config import LLMConfig
from complaints_trends.gigachat_api.token_transport import AuthorizationKeyTokenProvider, TokenAuthorizedHTTPXClient, build_verify_arg


class _Resp:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}
        self.reason_phrase = "OK" if status_code < 400 else "Too Many Requests"
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_token_transport_reads_authorization_key_and_caches_access_token(monkeypatch, tmp_path: Path):
    token_posts: list[dict] = []
    model_gets: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.base_url = kwargs.get("base_url")

        def post(self, url, headers=None, data=None, json=None):
            if str(url).endswith("/oauth"):
                token_posts.append({"url": url, "headers": headers, "data": data})
                return _Resp({"access_token": "token-123", "expires_in": 1800})
            return _Resp({"choices": [{"message": {"content": "{\"complaint_category\":\"OTHER\",\"is_complaint\":false,\"loan_product\":\"NONE\",\"severity\":\"low\",\"keywords\":[\"вопрос\",\"инфо\",\"уточнение\"],\"confidence\":0.9}"}}]})

        def get(self, url, headers=None):
            model_gets.append({"url": url, "headers": headers})
            return _Resp({"data": [{"id": "GigaChat"}, {"id": "GigaChat-Pro"}]})

    monkeypatch.setattr("complaints_trends.gigachat_api.token_transport.httpx.Client", FakeClient)

    key_file = tmp_path / "key"
    key_file.write_text("basic-key-value", encoding="utf-8")
    cfg = LLMConfig(
        enabled=True,
        mode="token",
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        oauth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        authorization_key_file=str(key_file),
        verify_ssl_certs=True,
        model="GigaChat",
        cache_db=str(tmp_path / "cache.sqlite"),
    )

    provider = AuthorizationKeyTokenProvider(
        oauth_url=cfg.oauth_url,
        authorization_key_file=cfg.authorization_key_file,
        scope=cfg.oauth_scope,
        verify=build_verify_arg(cfg),
    )
    client = TokenAuthorizedHTTPXClient(base_url=cfg.base_url, token_provider=provider, verify=build_verify_arg(cfg))

    first = client.list_models()
    second = client.list_models()

    assert first == ["GigaChat", "GigaChat-Pro"]
    assert second == ["GigaChat", "GigaChat-Pro"]
    assert len(token_posts) == 1
    assert token_posts[0]["headers"]["Authorization"] == "Basic basic-key-value"
    assert model_gets[0]["headers"]["Authorization"] == "Bearer token-123"


def test_token_provider_refresh_is_thread_safe(monkeypatch, tmp_path: Path):
    token_posts: list[dict] = []
    token_posts_lock = threading.Lock()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def post(self, url, headers=None, data=None, json=None):
            if str(url).endswith("/oauth"):
                time.sleep(0.02)
                with token_posts_lock:
                    token_posts.append({"url": url, "headers": headers, "data": data})
                return _Resp({"access_token": "shared-token", "expires_in": 1800})
            return _Resp({})

    monkeypatch.setattr("complaints_trends.gigachat_api.token_transport.httpx.Client", FakeClient)

    key_file = tmp_path / "key"
    key_file.write_text("basic-key-value", encoding="utf-8")
    provider = AuthorizationKeyTokenProvider(
        oauth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        authorization_key_file=str(key_file),
        scope="GIGACHAT_API_PERS",
        verify=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        tokens = list(executor.map(lambda _: provider.get_access_token(), range(8)))

    assert tokens == ["shared-token"] * 8
    assert len(token_posts) == 1


def test_token_transport_retries_chat_after_rate_limit(monkeypatch, tmp_path: Path):
    import complaints_trends.gigachat_api.rate_limit as rate_limit

    rate_limit._LIMITERS.clear()
    monkeypatch.setenv("GIGACHAT_MIN_REQUEST_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("GIGACHAT_MAX_REQUEST_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(rate_limit, "rate_limit_delay", lambda response, attempt: 0)
    monkeypatch.setattr(rate_limit.time, "sleep", lambda delay: None)

    chat_posts: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.base_url = kwargs.get("base_url")

        def post(self, url, headers=None, data=None, json=None):
            if str(url).endswith("/oauth"):
                return _Resp({"access_token": "token-123", "expires_in": 1800})
            chat_posts.append({"url": url, "headers": headers, "json": json})
            if len(chat_posts) < 3:
                return _Resp({}, status_code=429)
            return _Resp({"choices": [{"message": {"content": "{\"ok\": true}"}}]})

    monkeypatch.setattr("complaints_trends.gigachat_api.token_transport.httpx.Client", FakeClient)

    key_file = tmp_path / "key"
    key_file.write_text("basic-key-value", encoding="utf-8")
    provider = AuthorizationKeyTokenProvider(
        oauth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        authorization_key_file=str(key_file),
        scope="GIGACHAT_API_PERS",
        verify=True,
    )
    client = TokenAuthorizedHTTPXClient(
        base_url="https://gigachat.devices.sberbank.ru/api/v1/retry-test",
        token_provider=provider,
        verify=True,
    )

    result = client.chat({"messages": []})

    assert result.choices[0].message.content == "{\"ok\": true}"
    assert len(chat_posts) == 3


def test_build_verify_arg_falls_back_to_system_store_when_ca_bundle_is_missing(tmp_path: Path):
    cfg = LLMConfig(
        enabled=True,
        mode="token",
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        ca_bundle_file=str(tmp_path / "missing-ca.pem"),
        oauth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        authorization_key_file=str(tmp_path / "key"),
        verify_ssl_certs=True,
        model="GigaChat",
        cache_db=str(tmp_path / "cache.sqlite"),
    )

    assert build_verify_arg(cfg) is True
