from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
from uuid import uuid4

import httpx

from ..config import LLMConfig
from .rate_limit import get_rate_limiter


@dataclass
class OAuthToken:
    access_token: str
    expires_at: datetime


def resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    return Path(path_value).expanduser()


def build_verify_arg(cfg: LLMConfig) -> bool | str:
    verify: bool | str = cfg.verify_ssl_certs
    if cfg.ca_bundle_file:
        ca_bundle_path = resolve_path(cfg.ca_bundle_file)
        if ca_bundle_path is not None and ca_bundle_path.exists():
            verify = str(ca_bundle_path)
    return verify


class AuthorizationKeyTokenProvider:
    def __init__(
        self,
        *,
        oauth_url: str,
        authorization_key_file: str,
        scope: str,
        verify: bool | str,
        timeout: float = 60.0,
    ) -> None:
        self.oauth_url = oauth_url
        self.authorization_key_file = authorization_key_file
        self.scope = scope
        self.verify = verify
        self.timeout = timeout
        self._token: OAuthToken | None = None
        self._token_lock = threading.Lock()
        self._oauth_limiter = get_rate_limiter(f"oauth:{oauth_url}")
        self._client = httpx.Client(verify=verify, timeout=timeout, trust_env=False)

    def _read_authorization_key(self) -> str:
        path = resolve_path(self.authorization_key_file)
        if path is None or not path.exists():
            raise FileNotFoundError(
                "Authorization key file is missing for token transport: "
                f"{self.authorization_key_file}. Place the Basic auth key into this file."
            )
        key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"Authorization key file is empty: {path}")
        return key

    @staticmethod
    def _coerce_expiry(payload: dict) -> datetime:
        now = datetime.now(timezone.utc)
        default_expiry = now + timedelta(minutes=29)

        expires_in = payload.get("expires_in")
        if expires_in not in (None, ""):
            try:
                seconds = max(int(float(expires_in)) - 30, 60)
                return now + timedelta(seconds=seconds)
            except Exception:
                return default_expiry

        for field in ("expires_at", "expired_at", "token_expires_at"):
            raw = payload.get(field)
            if raw in (None, ""):
                continue
            if isinstance(raw, (int, float)):
                try:
                    ts = float(raw)
                    if ts > 10_000_000_000:
                        ts = ts / 1000.0
                    return datetime.fromtimestamp(ts, tz=timezone.utc) - timedelta(seconds=30)
                except Exception:
                    return default_expiry
            if isinstance(raw, str):
                value = raw.strip()
                if not value:
                    continue
                try:
                    normalized = value.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(normalized)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc) - timedelta(seconds=30)
                except Exception:
                    continue
        return default_expiry

    def get_access_token(self) -> str:
        now = datetime.now(timezone.utc)
        if self._token is not None and now < self._token.expires_at:
            return self._token.access_token

        with self._token_lock:
            now = datetime.now(timezone.utc)
            if self._token is not None and now < self._token.expires_at:
                return self._token.access_token

            authorization_key = self._read_authorization_key()
            response = self._oauth_limiter.execute(
                lambda: self._client.post(
                    self.oauth_url,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Authorization": f"Basic {authorization_key}",
                        "RqUID": str(uuid4()),
                    },
                    data={"scope": self.scope},
                )
            )
            response.raise_for_status()
            payload = response.json()
            access_token = str(payload.get("access_token") or "").strip()
            if not access_token:
                raise RuntimeError("OAuth response does not contain access_token")
            self._token = OAuthToken(access_token=access_token, expires_at=self._coerce_expiry(payload))
            return access_token


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _ChatResp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class TokenAuthorizedHTTPXClient:
    def __init__(self, *, base_url: str, token_provider: AuthorizationKeyTokenProvider, verify: bool | str, timeout: float = 60.0):
        self._api_client = httpx.Client(base_url=base_url, verify=verify, timeout=timeout, trust_env=False)
        self._token_provider = token_provider
        self._api_limiter = get_rate_limiter(f"api:{base_url}")

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token_provider.get_access_token()}",
        }

    def count_tokens(self, *, model: str, input_text: str) -> int | None:
        try:
            response = self._post_api_with_backoff("/tokens/count", json={"model": model, "input": [input_text]})
            if response.status_code >= 400:
                return None
            data = response.json()
            return self._extract_token_count(data)
        except Exception:
            return None

    def _post_api_with_backoff(self, path: str, *, json: dict) -> httpx.Response:
        return self._api_limiter.execute(
            lambda: self._api_client.post(path, json=json, headers=self._headers())
        )

    @classmethod
    def _extract_token_count(cls, data: object) -> int | None:
        if isinstance(data, list):
            counts = [cls._extract_token_count(item) for item in data]
            valid_counts = [item for item in counts if item is not None]
            return sum(valid_counts) if valid_counts else None
        if isinstance(data, dict):
            for key in ("tokens", "count", "token_count", "total_tokens"):
                value = data.get(key)
                if isinstance(value, (int, float, str)) and str(value).strip():
                    try:
                        return int(float(value))
                    except Exception:
                        pass
                nested = cls._extract_token_count(value)
                if nested is not None:
                    return nested
            for key in ("data", "items", "result", "results"):
                nested = cls._extract_token_count(data.get(key))
                if nested is not None:
                    return nested
        return None

    def chat(self, payload: dict) -> _ChatResp:
        response = self._post_api_with_backoff("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _ChatResp(content)

    def list_models(self) -> list[str]:
        response = self._api_client.get("/models", headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        items = payload
        if isinstance(payload, dict):
            items = payload.get("data") or payload.get("models") or payload.get("items") or []
        if not isinstance(items, list):
            return []
        models: list[str] = []
        for item in items:
            if isinstance(item, dict):
                model_name = item.get("id") or item.get("name")
                if model_name:
                    models.append(str(model_name))
            elif item:
                models.append(str(item))
        return models
