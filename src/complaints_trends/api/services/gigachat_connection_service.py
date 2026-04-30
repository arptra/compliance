from __future__ import annotations

from pathlib import Path

from ...config import ProjectConfig
from ...gigachat_api import build_gigachat_transport_client, resolve_path
from .gigachat_lab_service import GigaChatLabService
from ..schemas import (
    GigaChatTransportArtifact,
    GigaChatTransportProbeRequest,
    GigaChatTransportProbeResponse,
    GigaChatTransportStatus,
    GigaChatTransportStatusResponse,
)


class GigaChatConnectionService:
    def __init__(self, cfg: ProjectConfig, *, lab_service: GigaChatLabService | None = None) -> None:
        self.cfg = cfg
        self.lab_service = lab_service

    def _llm_cfg(self):
        llm_cfg = self.cfg.llm.model_copy(deep=True)
        if self.lab_service is not None:
            llm_cfg = self.lab_service.apply_llm_overrides(llm_cfg)
        return llm_cfg

    @staticmethod
    def _artifact(label: str, path_value: str | None) -> GigaChatTransportArtifact:
        path = resolve_path(path_value)
        return GigaChatTransportArtifact(
            label=label,
            path=str(path or path_value or ""),
            exists=bool(path and path.exists()),
        )

    @staticmethod
    def _message_for_artifacts(*, ready: bool, missing: list[str], fallback: str, missing_prefix: str) -> str:
        if ready:
            return "Готово к использованию."
        if missing:
            return f"{missing_prefix}: {', '.join(missing)}."
        return fallback

    def status(self) -> GigaChatTransportStatusResponse:
        llm = self._llm_cfg()

        mtls_artifacts = [
            self._artifact("CA bundle", llm.ca_bundle_file),
            self._artifact("Client cert", llm.cert_file),
            self._artifact("Client key", llm.key_file),
        ]
        mtls_ready = all(item.exists for item in mtls_artifacts)
        mtls_configured = all(bool(value) for value in (llm.base_url, llm.ca_bundle_file, llm.cert_file, llm.key_file))
        mtls_missing = [item.label for item in mtls_artifacts if not item.exists]

        token_artifacts = [
            self._artifact("Authorization key", llm.authorization_key_file),
        ]
        token_ready = token_artifacts[0].exists and bool(llm.oauth_url) and bool(llm.base_url)
        token_configured = bool(llm.authorization_key_file) and bool(llm.oauth_url) and bool(llm.base_url)
        token_missing = [item.label for item in token_artifacts if not item.exists]

        transports = [
            GigaChatTransportStatus(
                name="mtls",
                title="mTLS",
                description="Подключение по клиентскому сертификату и ключу.",
                active=llm.mode == "mtls",
                configured=mtls_configured,
                ready=mtls_ready,
                base_url=llm.base_url,
                artifacts=mtls_artifacts,
                message=self._message_for_artifacts(
                    ready=mtls_ready,
                    missing=mtls_missing,
                    fallback="Нужны CA bundle, cert и key файлы.",
                    missing_prefix="Отсутствуют обязательные файлы",
                ),
            ),
            GigaChatTransportStatus(
                name="token",
                title="Token",
                description="OAuth через Basic Authorization key из файла `key` с последующим Bearer token.",
                active=llm.mode == "token",
                configured=token_configured,
                ready=token_ready,
                base_url=llm.base_url,
                oauth_url=llm.oauth_url,
                artifacts=token_artifacts,
                message=self._message_for_artifacts(
                    ready=token_ready,
                    missing=token_missing,
                    fallback="Нужен oauth_url и файл с Authorization key.",
                    missing_prefix="Отсутствуют обязательные файлы",
                ),
            ),
        ]
        return GigaChatTransportStatusResponse(
            configured_mode=llm.mode,
            model=llm.model,
            transports=transports,
        )

    def probe(self, req: GigaChatTransportProbeRequest) -> GigaChatTransportProbeResponse:
        llm_cfg = self._llm_cfg()
        llm_cfg.mode = req.transport
        try:
            client = build_gigachat_transport_client(llm_cfg, transport=req.transport)
            models = client.list_models()
            preview = models[:8]
            message = "Соединение успешно установлено."
            if preview:
                message = f"Соединение успешно установлено, найдено моделей: {len(models)}."
            ok = True
        except Exception as exc:
            preview = []
            message = str(exc)
            ok = False
        return GigaChatTransportProbeResponse(
            transport=req.transport,
            ok=ok,
            base_url=llm_cfg.base_url,
            oauth_url=(llm_cfg.oauth_url if req.transport == "token" else None),
            model=llm_cfg.model,
            message=message,
            models=preview,
        )
