from __future__ import annotations

from ..config import LLMConfig
from .token_transport import AuthorizationKeyTokenProvider, TokenAuthorizedHTTPXClient, build_verify_arg


def build_gigachat_transport_client(cfg: LLMConfig, transport: str | None = None):
    selected = str(transport or cfg.mode or "mtls").strip().lower()
    if selected == "token":
        verify = build_verify_arg(cfg)
        provider = AuthorizationKeyTokenProvider(
            oauth_url=cfg.oauth_url,
            authorization_key_file=cfg.authorization_key_file,
            scope=cfg.oauth_scope,
            verify=verify,
            timeout=60.0,
        )
        return TokenAuthorizedHTTPXClient(base_url=cfg.base_url, token_provider=provider, verify=verify, timeout=60.0)

    from ..gigachat_mtls import _HTTPXChatClient, _build_mtls_ssl_context, _validate_mtls_files
    import os

    if selected == "mtls":
        _validate_mtls_files(cfg.ca_bundle_file, cfg.cert_file, cfg.key_file)
        key_password = (os.getenv(cfg.key_file_password_env) if cfg.key_file_password_env else os.getenv("GIGACHAT_KEY_PASSWORD")) or None
        ssl_context = _build_mtls_ssl_context(
            ca_bundle_file=cfg.ca_bundle_file,
            cert_file=cfg.cert_file,
            key_file=cfg.key_file,
            key_file_password=key_password,
            verify_ssl_certs=cfg.verify_ssl_certs,
        )
        return _HTTPXChatClient(base_url=cfg.base_url, verify=ssl_context, timeout=60.0)

    verify: bool | str = cfg.verify_ssl_certs
    if cfg.ca_bundle_file:
        verify = cfg.ca_bundle_file
    return _HTTPXChatClient(base_url=cfg.base_url, verify=verify, timeout=60.0)
