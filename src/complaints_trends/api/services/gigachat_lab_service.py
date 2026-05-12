from __future__ import annotations

import json
import math
import re
import threading
import traceback
import uuid
from contextlib import suppress
from io import BytesIO
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from ...config import LLMConfig, ProjectConfig
from ...gigachat_api import build_gigachat_transport_client
from ...gigachat_mtls import SYSTEM_PROMPT
from ..schemas import (
    GigaChatAnnotatedExportRequest,
    GigaChatBackgroundTaskListResponse,
    GigaChatBackgroundTaskResultResponse,
    GigaChatBackgroundTaskRowRun,
    GigaChatBackgroundTaskStartRequest,
    GigaChatBackgroundTaskSummary,
    GigaChatFinalPromptRequest,
    GigaChatFinalPromptResponse,
    GigaChatLabRowRunRequest,
    GigaChatLabRowRunResponse,
    GigaChatRuleEvaluationRequest,
    GigaChatRuleEvaluationResponse,
    GigaChatRuleEvaluationRow,
    GigaChatRuleHit,
    GigaChatRulePack,
    GigaChatRulePackFilter,
    GigaChatLabSettingField,
    GigaChatLabSettingOption,
    GigaChatLabSettingsResponse,
    GigaChatLabSettingsUpdateRequest,
    GigaChatLabSettingsVersionCreateRequest,
    GigaChatLabSettingsVersionResponse,
    GigaChatLabSettingsVersionsResponse,
    GigaChatLabSettingsVersionSummary,
    GigaChatLabSettingsVersionUpdateRequest,
    GigaChatWorkbookSelectSheetRequest,
    GigaChatWorkbookSheetDataResponse,
    GigaChatWorkbookSheetPreview,
    GigaChatWorkbookUploadResponse,
)

try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE as _OPENPYXL_ILLEGAL_RE
except Exception:  # pragma: no cover
    _OPENPYXL_ILLEGAL_RE = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")


class GigaChatLabService:
    DEFAULT_SYSTEM_PROMPT = (
        "Ты обязан вернуть только валидный JSON без markdown, без пояснений вне JSON и без служебного текста. "
        "Размечай одно клиентское обращение за раз, используй все доступные поля строки и не выдумывай факты, классы или теги."
    )
    DEFAULT_USER_PROMPT_PREFIX = (
        "Разметь одну запись и верни JSON с полями: "
        "client_first_message, short_summary, is_complaint, complaint_category, complaint_subcategory, "
        "product_area, loan_product, severity, keywords, assigned_tags, local_tags, model_added_tags, "
        "model_rejected_tags, reclassified_topic, confirmed_rule_hits, rejected_rule_hits, tag_decisions, "
        "match_type, evidence, confidence, notes, evidence_columns. "
        "Сначала определи, является ли запись жалобой. Затем выбери основную категорию, при необходимости подкатегорию, "
        "назначь только разрешенные теги, при необходимости подтверди или отвергни подсказки rule-based движка, "
        "оцени severity и confidence, а в notes кратко объясни решение. Возвращай только классы и теги из разрешенных списков. "
        "Если ни один разрешенный класс не подходит, верни пустую строку в complaint_category и объясни причину в notes."
    )
    DEFAULT_CONTEXT_NOTES = (
        "Источник данных: Excel/CSV с клиентскими обращениями. "
        "Одна строка таблицы = одно обращение для разметки. "
        "Используй одновременно текст клиента, комментарии, суммаризацию, продукт, тематику, канал и другие непустые поля. "
        "Если поля противоречат друг другу, приоритет у явного текста клиента, затем у суммаризации, затем у служебных атрибутов. "
        "Если обращение не выглядит как жалоба, верни is_complaint=false и не придумывай complaint_category."
    )
    DEFAULT_FALLBACK_ON_BLANK_KEYS = {
        "system_prompt",
        "user_prompt_prefix",
        "context_notes",
        "classification_prompt_notes",
        "tagging_prompt_notes",
        "rule_pack_prompt_notes",
    }
    RULE_FILTER_FIELD_ALIASES: dict[str, list[str]] = {
        "Трайб": ["Во. Группа"],
        "драйвер": ["Во. Тематика"],
    }
    FIELD_DEFS: list[dict[str, Any]] = [
        {"key": "model", "label": "Model", "input_type": "text", "section": "Запрос"},
        {"key": "temperature", "label": "Temperature", "input_type": "number", "section": "Запрос", "help_text": "Насколько вариативным делать ответ модели."},
        {"key": "top_p", "label": "Top P", "input_type": "number", "section": "Запрос", "help_text": "Альтернативный sampling-порог по вероятности токенов."},
        {"key": "max_output_tokens", "label": "Max output tokens", "input_type": "number", "section": "Запрос", "help_text": "Максимальный размер ответа модели."},
        {"key": "system_prompt", "label": "System prompt", "input_type": "textarea", "section": "Промпты", "help_text": "Базовый system prompt для экспериментальной разметки."},
        {"key": "user_prompt_prefix", "label": "User prompt prefix", "input_type": "textarea", "section": "Промпты", "help_text": "Дополнительный текст перед пользовательским payload."},
        {"key": "context_notes", "label": "Context notes", "input_type": "textarea", "section": "Промпты", "help_text": "Текстовые инструкции про контекст, листы Excel и особенности эксперимента."},
        {"key": "classification_prompt_notes", "label": "Классификации", "input_type": "textarea", "section": "Разметка", "help_text": "Правила корзин, категорий и подкатегорий для классификации."},
        {"key": "tagging_prompt_notes", "label": "Теги", "input_type": "textarea", "section": "Разметка", "help_text": "Правила тегирования, словари тегов и требования к их формату."},
        {"key": "rule_pack_prompt_notes", "label": "Rule packs", "input_type": "textarea", "section": "Разметка", "help_text": "Локальные rule-based пакеты: фильтры по полям, словари и действия до GigaChat."},
    ]
    _background_lock = threading.Lock()
    _background_cancel_flags: dict[str, threading.Event] = {}
    _background_threads: dict[str, threading.Thread] = {}

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self.base_dir = Path(cfg.analysis.pattern_monitoring.interim_dir) / "gigachat_lab"
        self.uploads_dir = self.base_dir / "uploads"
        self.background_dir = Path("data/background")
        self.versions_dir = Path("data/gigachat_lab/versions")
        self.settings_path = self.base_dir / "settings.json"
        self.final_prompt_path = self.base_dir / "final_prompt_snapshot.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.background_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _serialize_rule_items(items: list[dict[str, str]]) -> str:
        return json.dumps(items, ensure_ascii=False, indent=2)

    def _default_classification_rules(self) -> str:
        items = [
            {
                "name": "Проблема с выдачей очередного транша по Образовательному кредиту",
                "description": "Единственный класс для правила EDU_RECLASS_TRANCH. Выбирай только когда подтверждена проблема с очередным траншем или семестром по образовательному кредиту.",
            },
            {
                "name": "Проблемы с зачислением средств/ оформлением-рассмотрением заявки",
                "description": "Единственный класс для правила EDU_RECLASS_APPLICATION. Выбирай только когда подтверждена проблема с зачислением средств, оплатой обучения, вузом, периодом обучения, отчислением или заявкой по образовательному кредиту.",
            },
        ]
        return self._serialize_rule_items(items)

    def _default_tag_rules(self) -> str:
        items = [
            {
                "name": "DRA",
                "description": "Единственный разрешенный тег. Назначай только когда подтверждено правило DRA/ДРПА по словам про смерть, наследство, каникулы, реструктуризацию, приставов, СВО, суд, исполнительное производство, военный контур или банкротство.",
            },
            {
                "name": "ИПОТЕКА",
                "description": "Назначай, когда обращение относится к ипотеке, жилищному кредиту, кредиту под залог недвижимости, квартире/дому в залоге, закладной, эскроу, обременению, созаемщику или рефинансированию ипотечного кредита.",
            },
        ]
        return self._serialize_rule_items(items)

    def _default_rule_packs(self) -> str:
        items = [
            {
                "code": "DRA",
                "description": "Проставляет тег DRA/ДРПА по словам про смерть, наследство, каникулы, реструктуризацию, приставов, СВО, суд, исполнительное производство, военный контур и банкротство.",
                "enabled": True,
                "type": "assign_tag",
                "source_fields": ["Во. Описание", "Обр. Результат суммаризации диалога"],
                "keywords": [
                    "умер",
                    "погиб",
                    "смерт",
                    "гибел",
                    "наследни",
                    "наследств",
                    "каникул",
                    "реструктуриз",
                    "пристав",
                    "участник СВО",
                    "участника СВО",
                    "участником СВО",
                    "на СВО",
                    "судебное решение",
                    "по решению суда",
                    "исполнительное производство",
                    "военн",
                    "банкрот",
                ],
                "filters": [],
                "target_tag": "DRA",
                "target_topic": None,
            },
            {
                "code": "IPOTEKA",
                "description": "Проставляет тег ИПОТЕКА по обращениям про ипотеку, жилищный кредит, недвижимость в залоге, закладную, эскроу, обременение и ипотечное рефинансирование.",
                "enabled": True,
                "type": "assign_tag",
                "source_fields": ["Во. Описание", "Обр. Результат суммаризации диалога"],
                "keywords": [
                    "ипотек",
                    "жилищн",
                    "недвижим",
                    "квартир",
                    "дом в залог",
                    "залог недвиж",
                    "закладн",
                    "эскроу",
                    "обременен",
                    "созаемщик",
                    "созаемщ",
                    "рефинансир*ипот",
                    "рефинансирован*ипот",
                    "материнск*капитал",
                    "первоначальн*взнос",
                ],
                "filters": [],
                "target_tag": "ИПОТЕКА",
                "target_topic": None,
            },
            {
                "code": "EDU_RECLASS_TRANCH",
                "description": "Переклассифицирует обращения в тему про очередной транш по образовательному кредиту.",
                "enabled": True,
                "type": "reclass_topic",
                "source_fields": ["Во. Описание", "Обр. Результат суммаризации диалога"],
                "keywords": ["транш", "семестр"],
                "filters": [
                    {"field": "Трайб", "op": "eq", "value": "ПОТРЕБИТЕЛЬСКИЕ КРЕДИТЫ"},
                    {"field": "драйвер", "op": "ne", "value": "ОБРАЗОВАТЕЛЬНЫЙ КРЕДИТ"},
                ],
                "target_tag": None,
                "target_topic": "Проблема с выдачей очередного транша по Образовательному кредиту",
            },
            {
                "code": "EDU_RECLASS_APPLICATION",
                "description": "Переклассифицирует обращения в тему про зачисление средств, оформление или рассмотрение заявки по образовательному кредиту.",
                "enabled": True,
                "type": "reclass_topic",
                "source_fields": ["Во. Описание", "Обр. Результат суммаризации диалога"],
                "keywords": [
                    "Образовательн",
                    "Вуз",
                    "Кредит на образ",
                    "Оплатить обучение",
                    "Период*обучения",
                    "Отчисл",
                ],
                "filters": [
                    {"field": "Трайб", "op": "eq", "value": "ПОТРЕБИТЕЛЬСКИЕ КРЕДИТЫ"},
                    {"field": "драйвер", "op": "ne", "value": "ОБРАЗОВАТЕЛЬНЫЙ КРЕДИТ"},
                ],
                "target_tag": None,
                "target_topic": "Проблемы с зачислением средств/ оформлением-рассмотрением заявки",
            },
        ]
        return json.dumps(items, ensure_ascii=False, indent=2)

    def _default_values(self) -> dict[str, Any]:
        llm = self.cfg.llm.model_dump()
        default_system_prompt = str(llm.get("system_prompt", "") or "").strip()
        if not default_system_prompt or default_system_prompt == LLMConfig.model_fields["system_prompt"].default:
            default_system_prompt = self.DEFAULT_SYSTEM_PROMPT

        default_user_prompt_prefix = str(llm.get("user_prompt_prefix", "") or "").strip() or self.DEFAULT_USER_PROMPT_PREFIX
        default_context_notes = str(llm.get("context_notes", "") or "").strip() or self.DEFAULT_CONTEXT_NOTES
        default_classification_notes = str(llm.get("classification_prompt_notes", "") or "").strip() or self._default_classification_rules()
        default_tagging_notes = str(llm.get("tagging_prompt_notes", "") or "").strip() or self._default_tag_rules()
        default_rule_pack_notes = str(llm.get("rule_pack_prompt_notes", "") or "").strip() or self._default_rule_packs()
        return {
            "model": llm.get("model", "GigaChat"),
            "temperature": llm.get("temperature", 0.2),
            "top_p": llm.get("top_p", 0.95),
            "max_output_tokens": llm.get("max_output_tokens", 2048),
            "system_prompt": default_system_prompt,
            "user_prompt_prefix": default_user_prompt_prefix,
            "context_notes": default_context_notes,
            "classification_prompt_notes": default_classification_notes,
            "tagging_prompt_notes": default_tagging_notes,
            "rule_pack_prompt_notes": default_rule_pack_notes,
        }

    def _load_saved_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return {"saved_at": None, "values": {}}
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {"saved_at": None, "values": {}}
            values = payload.get("values", {})
            if not isinstance(values, dict):
                values = {}
            return {"saved_at": payload.get("saved_at"), "values": values}
        except Exception:
            return {"saved_at": None, "values": {}}

    def _effective_values(self) -> tuple[dict[str, Any], str | None]:
        defaults = self._default_values()
        saved = self._load_saved_settings()
        values = dict(defaults)
        for key, value in dict(saved.get("values", {})).items():
            if key not in defaults:
                continue
            if key in self.DEFAULT_FALLBACK_ON_BLANK_KEYS and (value is None or (isinstance(value, str) and not value.strip())):
                continue
            values[key] = value
        return values, saved.get("saved_at")

    def _fields_from_values(self, values: dict[str, Any]) -> list[GigaChatLabSettingField]:
        fields: list[GigaChatLabSettingField] = []
        for meta in self.FIELD_DEFS:
            options = [GigaChatLabSettingOption(value=v, label=l) for v, l in meta.get("options", [])]
            fields.append(
                GigaChatLabSettingField(
                    key=meta["key"],
                    label=meta["label"],
                    input_type=meta["input_type"],
                    section=meta["section"],
                    help_text=meta.get("help_text"),
                    value=values.get(meta["key"]),
                    options=options,
                )
            )
        return fields

    @staticmethod
    def _version_id_from_title(title: str) -> str:
        raw = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.strip().lower()).strip("_")
        return raw or f"version_{uuid.uuid4().hex[:8]}"

    def _version_path(self, version_id: str) -> Path:
        safe = self._version_id_from_title(version_id)
        return self.versions_dir / f"{safe}.json"

    def _default_version_payload(self) -> dict[str, Any]:
        values, saved_at = self._effective_values()
        now = saved_at or self._now().isoformat()
        return {
            "version_id": "default",
            "title": "Default",
            "description": "Базовая версия из дефолтных и legacy-настроек Lab.",
            "status": "release",
            "created_by": "system",
            "created_at": now,
            "updated_at": saved_at,
            "base_version_id": None,
            "values": values,
        }

    def _read_version_payload(self, version_id: str) -> dict[str, Any]:
        if version_id == "default" and not self._version_path(version_id).exists():
            return self._default_version_payload()
        path = self._version_path(version_id)
        if not path.exists():
            raise FileNotFoundError(f"Settings version not found: {version_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Settings version is not a JSON object: {version_id}")
        return payload

    def _write_version_payload(self, payload: dict[str, Any]) -> None:
        version_id = str(payload.get("version_id") or "")
        if not version_id:
            raise ValueError("version_id is required")
        self._version_path(version_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _version_summary_from_payload(self, payload: dict[str, Any], *, is_default: bool = False) -> GigaChatLabSettingsVersionSummary:
        version_id = str(payload.get("version_id") or "default")
        path = self._version_path(version_id)
        return GigaChatLabSettingsVersionSummary(
            version_id=version_id,
            title=str(payload.get("title") or version_id),
            description=str(payload.get("description") or ""),
            status=payload.get("status") or "draft",
            created_by=str(payload.get("created_by") or ""),
            created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else None,
            updated_at=datetime.fromisoformat(payload["updated_at"]) if payload.get("updated_at") else None,
            base_version_id=payload.get("base_version_id"),
            path=str(path) if path.exists() else None,
            is_default=is_default,
        )

    def _values_from_version_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self._default_values()
        values = dict(defaults)
        raw_values = payload.get("values", {})
        if isinstance(raw_values, dict):
            for key, value in raw_values.items():
                if key in defaults:
                    values[key] = self._coerce_setting_value(key, value)
        return values

    def list_settings_versions(self) -> GigaChatLabSettingsVersionsResponse:
        versions: list[GigaChatLabSettingsVersionSummary] = []
        for path in sorted(self.versions_dir.glob("*.json")):
            with suppress(Exception):
                payload = json.loads(path.read_text(encoding="utf-8"))
                versions.append(self._version_summary_from_payload(payload, is_default=payload.get("version_id") == "default"))
        if not versions:
            versions.append(self._version_summary_from_payload(self._default_version_payload(), is_default=True))
        versions.sort(key=lambda item: (not item.is_default, item.title.lower()))
        return GigaChatLabSettingsVersionsResponse(versions=versions)

    def get_settings_version(self, version_id: str) -> GigaChatLabSettingsVersionResponse:
        payload = self._read_version_payload(version_id)
        values = self._values_from_version_payload(payload)
        return GigaChatLabSettingsVersionResponse(
            version=self._version_summary_from_payload(payload, is_default=version_id == "default"),
            fields=self._fields_from_values(values),
            values=values,
        )

    def create_settings_version(self, req: GigaChatLabSettingsVersionCreateRequest) -> GigaChatLabSettingsVersionResponse:
        version_id = self._version_id_from_title(req.version_id or req.title)
        if self._version_path(version_id).exists():
            raise ValueError(f"Settings version already exists: {version_id}")
        base_payload = self._read_version_payload(req.base_version_id or "default")
        now = self._now().isoformat()
        payload = {
            "version_id": version_id,
            "title": req.title.strip() or version_id,
            "description": req.description,
            "status": req.status,
            "created_by": req.created_by,
            "created_at": now,
            "updated_at": now,
            "base_version_id": req.base_version_id or "default",
            "values": self._values_from_version_payload(base_payload),
        }
        self._write_version_payload(payload)
        return self.get_settings_version(version_id)

    def save_settings_version(self, version_id: str, req: GigaChatLabSettingsVersionUpdateRequest) -> GigaChatLabSettingsVersionResponse:
        if version_id == "default" and not self._version_path(version_id).exists():
            payload = self._default_version_payload()
            payload["created_at"] = self._now().isoformat()
        else:
            payload = self._read_version_payload(version_id)
        if req.title is not None:
            payload["title"] = req.title
        if req.description is not None:
            payload["description"] = req.description
        if req.status is not None:
            payload["status"] = req.status
        values = self._values_from_version_payload(payload)
        for key, value in req.values.items():
            if key in values:
                values[key] = self._coerce_setting_value(key, value)
        payload["values"] = values
        payload["updated_by"] = req.updated_by or payload.get("updated_by") or payload.get("created_by") or ""
        payload["updated_at"] = self._now().isoformat()
        self._write_version_payload(payload)
        return self.get_settings_version(str(payload["version_id"]))

    def _coerce_setting_value(self, key: str, value: Any) -> Any:
        meta = next((x for x in self.FIELD_DEFS if x["key"] == key), None)
        if meta is None:
            return value
        input_type = meta["input_type"]
        if input_type == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)
        if input_type == "number":
            try:
                return float(value) if "." in str(value) and key in {"temperature", "top_p"} else int(value)
            except Exception:
                return 0.0 if key in {"temperature", "top_p"} else 0
        if value is None:
            return ""
        return str(value)

    def get_settings(self) -> GigaChatLabSettingsResponse:
        values, saved_at_raw = self._effective_values()
        saved_at = None
        if isinstance(saved_at_raw, str) and saved_at_raw.strip():
            try:
                saved_at = datetime.fromisoformat(saved_at_raw)
            except Exception:
                saved_at = None
        return GigaChatLabSettingsResponse(
            title="GigaChat Lab Settings",
            fields=self._fields_from_values(values),
            saved_at=saved_at,
        )

    def save_settings(self, req: GigaChatLabSettingsUpdateRequest) -> GigaChatLabSettingsResponse:
        allowed = {meta["key"] for meta in self.FIELD_DEFS}
        saved_values = dict(self._load_saved_settings().get("values", {}))
        for key, value in req.values.items():
            if key not in allowed:
                continue
            saved_values[key] = self._coerce_setting_value(key, value)
        payload = {
            "saved_at": self._now().isoformat(),
            "values": saved_values,
        }
        self.settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.get_settings()

    def apply_llm_overrides(self, llm_cfg: LLMConfig) -> LLMConfig:
        values, _ = self._effective_values()
        fields = llm_cfg.__class__.model_fields
        for key, value in values.items():
            if key in fields:
                setattr(llm_cfg, key, value)
        return llm_cfg

    def build_final_prompt(self, req: GigaChatFinalPromptRequest, *, save_snapshot: bool = False) -> GigaChatFinalPromptResponse:
        values = dict(self._effective_values()[0])
        for key, value in req.values.items():
            values[key] = self._coerce_setting_value(key, value)

        columns = [str(column) for column in req.columns if str(column).strip()]
        payload = self._compose_final_payload(values, columns)
        generated_at = self._now()
        saved_path = None

        if save_snapshot:
            snapshot = {
                "generated_at": generated_at.isoformat(),
                "source_columns": columns,
                "payload": payload,
            }
            self.final_prompt_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            self.save_settings(GigaChatLabSettingsUpdateRequest(values=req.values))
            saved_path = str(self.final_prompt_path)

        return GigaChatFinalPromptResponse(
            generated_at=generated_at,
            source_columns=columns,
            payload=payload,
            saved=save_snapshot,
            saved_path=saved_path,
        )

    def run_row_prompt(self, req: GigaChatLabRowRunRequest) -> GigaChatLabRowRunResponse:
        values = dict(self._effective_values()[0])
        for key, value in req.values.items():
            values[key] = self._coerce_setting_value(key, value)

        columns = [str(column) for column in req.columns if str(column).strip()]
        base_payload = req.payload_override or self._compose_final_payload(values, columns)
        rule_packs = self._parse_rule_pack_items(values.get("rule_pack_prompt_notes", ""))
        rule_evaluation = self._evaluate_rule_hits_for_row(rule_packs, req.row, 0)
        rendered_payload = self._render_payload_for_row(base_payload, req.row)
        rendered_payload = self._inject_row_rule_context(rendered_payload, rule_evaluation)

        llm_cfg = self.cfg.llm.model_copy(deep=True)
        llm_cfg = self.apply_llm_overrides(llm_cfg)
        llm_fields = llm_cfg.__class__.model_fields
        for key, value in values.items():
            if key in llm_fields:
                setattr(llm_cfg, key, value)
        llm_cfg.mode = req.transport

        client = build_gigachat_transport_client(llm_cfg, transport=req.transport)
        token_count = None
        if req.count_tokens and hasattr(client, "count_tokens"):
            token_count = client.count_tokens(model=llm_cfg.model, input_text=self._payload_to_count_input(rendered_payload))
        raw_content = client.chat(rendered_payload).choices[0].message.content
        parsed_payload: dict[str, Any] | list[Any] | None = None
        parse_ok = False
        try:
            parsed_payload = json.loads(raw_content)
            parse_ok = True
        except Exception:
            parsed_payload = None

        return GigaChatLabRowRunResponse(
            transport=req.transport,
            request_payload=rendered_payload,
            response_raw=raw_content,
            response_json=parsed_payload,
            parse_ok=parse_ok,
            request_token_count=token_count,
            rule_evaluation=rule_evaluation,
        )

    def list_background_tasks(self) -> GigaChatBackgroundTaskListResponse:
        tasks: list[GigaChatBackgroundTaskSummary] = []
        for meta_path in sorted(self.background_dir.glob("*/task.json"), reverse=True):
            with suppress(Exception):
                tasks.append(GigaChatBackgroundTaskSummary(**json.loads(meta_path.read_text(encoding="utf-8"))))
        tasks.sort(key=lambda item: item.created_at, reverse=True)
        return GigaChatBackgroundTaskListResponse(tasks=tasks)

    def start_background_labeling(self, req: GigaChatBackgroundTaskStartRequest) -> GigaChatBackgroundTaskSummary:
        task_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        task_dir = self.background_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        summary = GigaChatBackgroundTaskSummary(
            task_id=task_id,
            status="queued",
            filename=req.filename,
            sheet_name=req.sheet_name,
            created_at=self._now(),
            total_rows=len(req.rows),
            current_label="Задача поставлена в очередь",
        )
        (task_dir / "input.json").write_text(req.model_dump_json(indent=2), encoding="utf-8")
        self._write_background_summary(summary)

        cancel_flag = threading.Event()
        worker = threading.Thread(
            target=self._run_background_labeling,
            args=(task_id, req, cancel_flag),
            daemon=True,
            name=f"gigachat-bg-{task_id}",
        )
        with self._background_lock:
            self._background_cancel_flags[task_id] = cancel_flag
            self._background_threads[task_id] = worker
        worker.start()
        return summary

    def cancel_background_task(self, task_id: str) -> GigaChatBackgroundTaskSummary:
        summary = self._read_background_summary(task_id)
        with self._background_lock:
            flag = self._background_cancel_flags.get(task_id)
            if flag:
                flag.set()
        if summary.status in {"queued", "running"}:
            summary.status = "cancelled"
            summary.finished_at = self._now()
            summary.current_label = "Отмена запрошена"
            self._write_background_summary(summary)
        return summary

    def load_background_result(self, task_id: str) -> GigaChatBackgroundTaskResultResponse:
        summary = self._read_background_summary(task_id)
        result_path = self.background_dir / task_id / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Background task result not found: {task_id}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return GigaChatBackgroundTaskResultResponse(
            task=summary,
            workbook=GigaChatWorkbookSheetDataResponse(**payload["workbook"]),
            row_runs=[GigaChatBackgroundTaskRowRun(**row) for row in payload.get("row_runs", [])],
        )

    def _run_background_labeling(self, task_id: str, req: GigaChatBackgroundTaskStartRequest, cancel_flag: threading.Event) -> None:
        summary = self._read_background_summary(task_id)
        task_dir = self.background_dir / task_id
        source_rows = [item.source_row for item in req.rows]
        row_runs: list[dict[str, Any]] = []
        summary.status = "running"
        summary.started_at = self._now()
        summary.current_label = "Фоновая разметка запущена"
        self._write_background_summary(summary)

        try:
            for index, item in enumerate(req.rows):
                if cancel_flag.is_set():
                    summary.status = "cancelled"
                    summary.current_label = "Задача отменена"
                    break
                summary.current_label = f"Размечаем строку {index + 1} из {len(req.rows)}"
                self._write_background_summary(summary)
                try:
                    result = self.run_row_prompt(GigaChatLabRowRunRequest(
                        transport=req.transport,
                        values=req.values,
                        columns=req.columns,
                        row=item.source_row,
                        payload_override=req.payload_override,
                        count_tokens=req.count_tokens,
                    ))
                    row_runs.append({
                        "row_index": item.row_index,
                        "source_row": item.source_row,
                        "result": result.model_dump(mode="json"),
                        "error": None,
                    })
                except Exception as exc:
                    summary.failed_rows += 1
                    row_runs.append({
                        "row_index": item.row_index,
                        "source_row": item.source_row,
                        "result": None,
                        "error": str(exc),
                    })
                summary.completed_rows = index + 1
                summary.progress = summary.completed_rows / summary.total_rows if summary.total_rows else 1
                self._write_background_summary(summary)

            if summary.status != "cancelled":
                summary.status = "completed"
                summary.current_label = "Готово"
            summary.finished_at = self._now()
            workbook = GigaChatWorkbookSheetDataResponse(
                upload_id=f"background-{task_id}",
                filename=req.filename,
                file_format="csv",
                sheet_name=req.sheet_name,
                total_rows=len(source_rows),
                rendered_rows=len(source_rows),
                columns=req.columns,
                rows=source_rows,
            )
            result_path = task_dir / "result.json"
            result_path.write_text(json.dumps({
                "workbook": workbook.model_dump(mode="json"),
                "row_runs": row_runs,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            summary.result_path = str(result_path)
            self._write_background_summary(summary)
        except Exception as exc:
            summary.status = "failed"
            summary.error = f"{exc}\n{traceback.format_exc()}"
            summary.current_label = "Задача завершилась ошибкой"
            summary.finished_at = self._now()
            self._write_background_summary(summary)
        finally:
            with self._background_lock:
                self._background_cancel_flags.pop(task_id, None)
                self._background_threads.pop(task_id, None)

    def _read_background_summary(self, task_id: str) -> GigaChatBackgroundTaskSummary:
        meta_path = self.background_dir / task_id / "task.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Background task not found: {task_id}")
        return GigaChatBackgroundTaskSummary(**json.loads(meta_path.read_text(encoding="utf-8")))

    def _write_background_summary(self, summary: GigaChatBackgroundTaskSummary) -> None:
        task_dir = self.background_dir / summary.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    def export_annotated_workbook(self, req: GigaChatAnnotatedExportRequest) -> tuple[str, bytes]:
        source_columns = [str(column).strip() for column in req.source_columns if str(column).strip()]
        ordered_rows = sorted(
            list(req.rows),
            key=lambda row: (row.row_index is None, row.row_index if row.row_index is not None else 0),
        )
        export_rows: list[dict[str, Any]] = []
        for row in ordered_rows:
            export_row: dict[str, Any] = {
                "Класс": row.classification,
                "Теги": ", ".join([str(tag).strip() for tag in row.tags if str(tag).strip()]),
                "Local tags": ", ".join([str(tag).strip() for tag in row.local_tags if str(tag).strip()]),
                "Model added tags": ", ".join([str(tag).strip() for tag in row.model_added_tags if str(tag).strip()]),
                "Model rejected tags": ", ".join([str(tag).strip() for tag in row.model_rejected_tags if str(tag).strip()]),
                "Match type": row.match_type or "",
                "Evidence": row.evidence or "",
                "Tag decisions": row.tag_decisions or "",
                "Rule hits": ", ".join([str(hit).strip() for hit in row.rule_hits if str(hit).strip()]),
                "Suggested topics": ", ".join([str(topic).strip() for topic in row.suggested_topics if str(topic).strip()]),
                "Confirmed rule hits": ", ".join([str(hit).strip() for hit in row.confirmed_rule_hits if str(hit).strip()]),
                "Rejected rule hits": ", ".join([str(hit).strip() for hit in row.rejected_rule_hits if str(hit).strip()]),
                "Rule decision": row.rule_decision or "",
                "Model decision": row.model_decision or "",
                "Reclassified topic": row.reclassified_topic or "",
                "Final topic": row.final_topic or "",
                "Decision source": row.decision_source or "",
            }
            for column in source_columns:
                export_row[column] = self._normalize_export_cell(row.source_row.get(column))
            export_rows.append(export_row)

        ordered_columns = [
            "Класс",
            "Теги",
            "Local tags",
            "Model added tags",
            "Model rejected tags",
            "Match type",
            "Evidence",
            "Tag decisions",
            "Rule hits",
            "Suggested topics",
            "Confirmed rule hits",
            "Rejected rule hits",
            "Rule decision",
            "Model decision",
            "Reclassified topic",
            "Final topic",
            "Decision source",
            *source_columns,
        ]
        df = pd.DataFrame(export_rows, columns=ordered_columns)
        df = self._sanitize_for_excel(df)

        buffer = BytesIO()
        sheet_name = self._excel_safe_sheet_name(req.sheet_name or "Разметка")
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col_idx in range(1, ws.max_column + 1):
                for row_idx in range(2, ws.max_row + 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    if isinstance(cell.value, datetime):
                        cell.number_format = "yyyy-mm-dd hh:mm:ss"
                    elif isinstance(cell.value, date):
                        cell.number_format = "yyyy-mm-dd"

        buffer.seek(0)
        export_filename = f"{Path(req.filename or 'annotated.xlsx').stem}_annotated.xlsx"
        return export_filename, buffer.getvalue()

    def export_validation_workbook(self, req: GigaChatAnnotatedExportRequest) -> tuple[str, bytes]:
        source_columns = [str(column).strip() for column in req.source_columns if str(column).strip()]
        ordered_rows = sorted(
            list(req.rows),
            key=lambda row: (row.row_index is None, row.row_index if row.row_index is not None else 0),
        )
        export_rows: list[dict[str, Any]] = []
        for row in ordered_rows:
            model_class = row.classification or ""
            model_tags = ", ".join([str(tag).strip() for tag in row.tags if str(tag).strip()])
            model_topic = row.final_topic or row.reclassified_topic or row.classification or ""
            export_row: dict[str, Any] = {
                "Review status": "needs_review",
                "Class matches": "",
                "Tags match": "",
                "Topic matches": "",
                "Expected class": "",
                "Expected tags": "",
                "Expected final topic": "",
                "Reviewer comment": "",
                "Model class": model_class,
                "Model tags": model_tags,
                "Local tags": ", ".join([str(tag).strip() for tag in row.local_tags if str(tag).strip()]),
                "Model added tags": ", ".join([str(tag).strip() for tag in row.model_added_tags if str(tag).strip()]),
                "Model rejected tags": ", ".join([str(tag).strip() for tag in row.model_rejected_tags if str(tag).strip()]),
                "Match type": row.match_type or "",
                "Evidence": row.evidence or "",
                "Tag decisions": row.tag_decisions or "",
                "Rule hits": ", ".join([str(hit).strip() for hit in row.rule_hits if str(hit).strip()]),
                "Suggested topics": ", ".join([str(topic).strip() for topic in row.suggested_topics if str(topic).strip()]),
                "Confirmed rule hits": ", ".join([str(hit).strip() for hit in row.confirmed_rule_hits if str(hit).strip()]),
                "Rejected rule hits": ", ".join([str(hit).strip() for hit in row.rejected_rule_hits if str(hit).strip()]),
                "Rule decision": row.rule_decision or "",
                "Model decision": row.model_decision or "",
                "Reclassified topic": row.reclassified_topic or "",
                "Model final topic": model_topic,
                "Decision source": row.decision_source or "",
            }
            for column in source_columns:
                export_row[column] = self._normalize_export_cell(row.source_row.get(column))
            export_rows.append(export_row)

        ordered_columns = [
            "Review status",
            "Class matches",
            "Tags match",
            "Topic matches",
            "Expected class",
            "Expected tags",
            "Expected final topic",
            "Reviewer comment",
            "Model class",
            "Model tags",
            "Local tags",
            "Model added tags",
            "Model rejected tags",
            "Match type",
            "Evidence",
            "Tag decisions",
            "Rule hits",
            "Suggested topics",
            "Confirmed rule hits",
            "Rejected rule hits",
            "Rule decision",
            "Model decision",
            "Reclassified topic",
            "Model final topic",
            "Decision source",
            *source_columns,
        ]
        df = pd.DataFrame(export_rows, columns=ordered_columns)
        df = self._sanitize_for_excel(df)

        buffer = BytesIO()
        sheet_name = self._excel_safe_sheet_name(req.sheet_name or "Validation")
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "I2"
            ws.auto_filter.ref = ws.dimensions

            from openpyxl.styles import Font, PatternFill
            from openpyxl.worksheet.datavalidation import DataValidation

            manual_fill = PatternFill("solid", fgColor="FFF2CC")
            model_fill = PatternFill("solid", fgColor="D9EAF7")
            source_fill = PatternFill("solid", fgColor="E2F0D9")
            for col_idx in range(1, ws.max_column + 1):
                header = ws.cell(row=1, column=col_idx)
                header.font = Font(bold=True)
                if col_idx <= 8:
                    header.fill = manual_fill
                elif col_idx <= 19:
                    header.fill = model_fill
                else:
                    header.fill = source_fill
                for row_idx in range(2, ws.max_row + 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    if isinstance(cell.value, datetime):
                        cell.number_format = "yyyy-mm-dd hh:mm:ss"
                    elif isinstance(cell.value, date):
                        cell.number_format = "yyyy-mm-dd"

            if ws.max_row >= 2:
                yes_no = DataValidation(type="list", formula1='"yes,no,needs_review"', allow_blank=True)
                ws.add_data_validation(yes_no)
                for col_letter in ("B", "C", "D"):
                    yes_no.add(f"{col_letter}2:{col_letter}{ws.max_row}")
                status = DataValidation(type="list", formula1='"needs_review,approved,rejected"', allow_blank=True)
                ws.add_data_validation(status)
                status.add(f"A2:A{ws.max_row}")

            for col_idx in range(1, ws.max_column + 1):
                max_len = len(str(ws.cell(row=1, column=col_idx).value or ""))
                for row_idx in range(2, min(ws.max_row, 40) + 1):
                    max_len = max(max_len, len(str(ws.cell(row=row_idx, column=col_idx).value or "")))
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max(12, min(48, max_len * 0.8))

        buffer.seek(0)
        export_filename = f"{Path(req.filename or 'annotated.xlsx').stem}_validation.xlsx"
        return export_filename, buffer.getvalue()

    def upload_workbook(self, filename: str, content: bytes) -> GigaChatWorkbookUploadResponse:
        upload_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        upload_dir = self.uploads_dir / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        display_filename = filename
        source_content = content
        ext = (Path(filename).suffix or ".xlsx").lower()
        if ext == ".zip":
            inner_name, source_content = self._extract_supported_file_from_zip(filename, content)
            ext = (Path(inner_name).suffix or ".xlsx").lower()
            display_filename = f"{filename} -> {inner_name}"

        stored_path = upload_dir / f"source{ext}"
        stored_path.write_bytes(source_content)

        meta = self._inspect_workbook(stored_path, display_filename)
        (upload_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return GigaChatWorkbookUploadResponse(**meta)

    def upload_local_workbook(self, filename: str) -> GigaChatWorkbookUploadResponse:
        requested_name = Path(filename).name
        if not requested_name or requested_name != filename:
            raise ValueError("Local workbook fallback accepts only a file name, not a path.")
        allowed_dirs = [
            Path("data/raw"),
            Path("outputs"),
            Path("outputs/gigachat_delivery"),
        ]
        for directory in allowed_dirs:
            candidate = directory / requested_name
            if candidate.is_file():
                return self.upload_workbook(candidate.name, candidate.read_bytes())
        searched = ", ".join(str(directory / requested_name) for directory in allowed_dirs)
        raise FileNotFoundError(f"Local workbook not found. Checked: {searched}")

    def load_sheet(self, upload_id: str, req: GigaChatWorkbookSelectSheetRequest) -> GigaChatWorkbookSheetDataResponse:
        upload_dir = self.uploads_dir / upload_id
        meta_path = upload_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Workbook upload not found: {upload_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        filename = str(meta.get("filename", ""))
        file_format = str(meta.get("file_format", "excel"))
        stored_path = next(upload_dir.glob("source.*"), None)
        if stored_path is None:
            raise FileNotFoundError(f"Workbook source file not found for upload: {upload_id}")

        if file_format == "csv":
            df = self._read_csv_frame(stored_path)
            sheet_name = "data"
        else:
            sheet_name = req.sheet_name
            available_sheets = {str(sheet.get("name", "")) for sheet in meta.get("sheets", [])}
            if sheet_name not in available_sheets:
                raise ValueError(f"Sheet not found in workbook: {sheet_name}")
            df = pd.read_excel(stored_path, sheet_name=sheet_name, dtype=object)
        df = self._drop_empty_unnamed_columns(df)

        columns = [str(c) for c in df.columns]
        row_limit = min(max(int(req.row_limit), 1), len(df))
        rows = self._frame_to_rows(df.head(row_limit))
        return GigaChatWorkbookSheetDataResponse(
            upload_id=upload_id,
            filename=filename,
            file_format="csv" if file_format == "csv" else "excel",
            sheet_name=sheet_name,
            total_rows=int(len(df)),
            rendered_rows=int(len(rows)),
            columns=columns,
            rows=rows,
        )

    def _inspect_workbook(self, stored_path: Path, filename: str) -> dict[str, Any]:
        suffix = stored_path.suffix.lower()
        if suffix == ".csv":
            df = self._read_csv_frame(stored_path)
            df = self._drop_empty_unnamed_columns(df)
            sheet = self._build_sheet_preview("data", df)
            return {
                "upload_id": stored_path.parent.name,
                "filename": filename,
                "file_format": "csv",
                "sheet_count": 1,
                "sheets": [sheet.model_dump()],
            }

        excel = pd.ExcelFile(stored_path)
        sheets = [
            self._build_sheet_preview(
                sheet_name,
                self._drop_empty_unnamed_columns(pd.read_excel(excel, sheet_name=sheet_name, dtype=object)),
            ).model_dump()
            for sheet_name in excel.sheet_names
        ]
        return {
            "upload_id": stored_path.parent.name,
            "filename": filename,
            "file_format": "excel",
            "sheet_count": len(sheets),
            "sheets": sheets,
        }

    @classmethod
    def _extract_supported_file_from_zip(cls, filename: str, content: bytes) -> tuple[str, bytes]:
        supported_suffixes = (".xlsx", ".xls", ".xlsm", ".csv")
        suffix_priority = {".xlsx": 0, ".xlsm": 1, ".xls": 2, ".csv": 3}
        with ZipFile(BytesIO(content)) as archive:
            candidates = [
                info for info in archive.infolist()
                if not info.is_dir()
                and not Path(info.filename).name.startswith("._")
                and "__MACOSX/" not in info.filename
                and info.filename.lower().endswith(supported_suffixes)
            ]
            if not candidates:
                raise ValueError(
                    f"В архиве {filename} не найдено поддерживаемых файлов Excel/CSV. "
                    "Ожидается .xlsx, .xls, .xlsm или .csv."
                )
            candidates.sort(key=lambda info: (
                suffix_priority.get(Path(info.filename).suffix.lower(), 99),
                len(Path(info.filename).parts),
                info.filename.lower(),
            ))
            selected = candidates[0]
            return selected.filename, archive.read(selected)

    @classmethod
    def _read_csv_frame(cls, stored_path: Path) -> pd.DataFrame:
        errors: list[str] = []
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            with suppress(Exception):
                return cls._normalize_csv_columns(pd.read_csv(stored_path, dtype=object, sep=None, engine="python", encoding=encoding))
            for separator in (";", ",", "\t", "|"):
                try:
                    df = pd.read_csv(stored_path, dtype=object, sep=separator, encoding=encoding)
                    if len(df.columns) > 1:
                        return cls._normalize_csv_columns(df)
                except Exception as exc:
                    errors.append(f"{encoding}/{separator}: {exc}")
        message = "; ".join(errors[:4]) or "не удалось определить кодировку или разделитель"
        raise ValueError(
            "Не удалось прочитать CSV. Проверьте разделитель и кавычки в файле. "
            f"Последние ошибки: {message}"
        )

    @staticmethod
    def _normalize_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(column).strip() for column in df.columns]
        return df

    @staticmethod
    def _drop_empty_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty and not len(df.columns):
            return df
        keep_columns = []
        for column in df.columns:
            column_name = str(column)
            is_unnamed = column_name.startswith("Unnamed:")
            if not is_unnamed:
                keep_columns.append(column)
                continue
            series = df[column]
            has_value = bool(series.map(lambda value: not pd.isna(value) and str(value).strip() != "").any())
            if has_value:
                keep_columns.append(column)
        return df.loc[:, keep_columns]

    @staticmethod
    def _parse_rule_items(raw: Any) -> list[dict[str, str]]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return [{"name": "", "description": text}]
        if not isinstance(parsed, list):
            return []
        items: list[dict[str, str]] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            items.append(
                {
                    "name": str(entry.get("name", "") or "").strip(),
                    "description": str(entry.get("description", "") or "").strip(),
                }
            )
        return [item for item in items if item["name"] or item["description"]]

    @staticmethod
    def _split_rule_list(raw: Any) -> list[str]:
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if raw is None:
            return []
        text = str(raw).replace("\r", "\n")
        parts = [part.strip() for chunk in text.split("\n") for part in chunk.split(",")]
        return [part for part in parts if part]

    @classmethod
    def _parse_rule_pack_items(cls, raw: Any) -> list[GigaChatRulePack]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []

        items: list[GigaChatRulePack] = []
        for index, entry in enumerate(parsed):
            if not isinstance(entry, dict):
                continue
            filters: list[GigaChatRulePackFilter] = []
            for filter_entry in entry.get("filters", []) or []:
                if not isinstance(filter_entry, dict):
                    continue
                field = str(filter_entry.get("field", "") or "").strip()
                op = str(filter_entry.get("op", "eq") or "eq").strip().lower()
                value = str(filter_entry.get("value", "") or "").strip()
                if not field or not value or op not in {"eq", "ne"}:
                    continue
                filters.append(GigaChatRulePackFilter(field=field, op=op, value=value))

            code = str(entry.get("code", "") or "").strip() or f"RULE_{index + 1}"
            rule_type = str(entry.get("type", "assign_tag") or "assign_tag").strip().lower()
            if rule_type not in {"assign_tag", "reclass_topic"}:
                rule_type = "assign_tag"

            items.append(
                GigaChatRulePack(
                    code=code,
                    description=str(entry.get("description", "") or "").strip(),
                    enabled=bool(entry.get("enabled", True)),
                    type=rule_type,
                    source_fields=cls._split_rule_list(entry.get("source_fields")),
                    keywords=cls._split_rule_list(entry.get("keywords")),
                    filters=filters,
                    target_tag=str(entry.get("target_tag", "") or "").strip() or None,
                    target_topic=str(entry.get("target_topic", "") or "").strip() or None,
                )
            )
        return items

    @staticmethod
    def _normalize_match_text(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def _keyword_matches_text(cls, keyword: str, text: str) -> bool:
        normalized_keyword = cls._normalize_match_text(keyword)
        normalized_text = cls._normalize_match_text(text)
        if not normalized_keyword or not normalized_text:
            return False
        regex_pattern = ".*".join(re.escape(part) for part in normalized_keyword.split("*"))
        regex_pattern = regex_pattern.replace(r"\ ", r"\s+")
        try:
            return re.search(regex_pattern, normalized_text, flags=re.IGNORECASE) is not None
        except re.error:
            return normalized_keyword in normalized_text

    @classmethod
    def _row_filter_matches(cls, row: dict[str, Any], rule_filter: GigaChatRulePackFilter) -> bool:
        field_names = [rule_filter.field, *cls.RULE_FILTER_FIELD_ALIASES.get(rule_filter.field, [])]
        row_value = next((row.get(field_name) for field_name in field_names if field_name in row), None)
        left = cls._normalize_match_text(row_value)
        right = cls._normalize_match_text(rule_filter.value)
        if rule_filter.op == "eq":
            return left == right
        return left != right

    @classmethod
    def _evaluate_rule_hits_for_row(
        cls,
        rule_packs: list[GigaChatRulePack],
        row: dict[str, Any],
        row_index: int,
    ) -> GigaChatRuleEvaluationRow:
        hits: list[GigaChatRuleHit] = []

        for rule_pack in rule_packs:
            if not rule_pack.enabled:
                continue
            if rule_pack.filters and not all(cls._row_filter_matches(row, rule_filter) for rule_filter in rule_pack.filters):
                continue

            matched_keywords: list[str] = []
            matched_fields: list[str] = []
            for keyword in rule_pack.keywords:
                keyword_matched = False
                for field_name in rule_pack.source_fields:
                    if cls._keyword_matches_text(keyword, row.get(field_name)):
                        keyword_matched = True
                        if field_name not in matched_fields:
                            matched_fields.append(field_name)
                if keyword_matched:
                    matched_keywords.append(keyword)

            if not matched_keywords:
                continue

            hits.append(
                GigaChatRuleHit(
                    code=rule_pack.code,
                    description=rule_pack.description,
                    type=rule_pack.type,
                    matched_keywords=matched_keywords,
                    matched_fields=matched_fields,
                    target_tag=rule_pack.target_tag,
                    target_topic=rule_pack.target_topic,
                )
            )

        suggested_tags = list(dict.fromkeys([hit.target_tag for hit in hits if hit.target_tag]))
        suggested_topics = list(dict.fromkeys([hit.target_topic for hit in hits if hit.target_topic]))

        return GigaChatRuleEvaluationRow(
            row_index=row_index,
            hits=hits,
            suggested_tags=[str(item) for item in suggested_tags],
            suggested_topics=[str(item) for item in suggested_topics],
        )

    def evaluate_rule_packs(self, req: GigaChatRuleEvaluationRequest) -> GigaChatRuleEvaluationResponse:
        values = dict(self._effective_values()[0])
        for key, value in req.values.items():
            values[key] = self._coerce_setting_value(key, value)
        rule_packs = self._parse_rule_pack_items(values.get("rule_pack_prompt_notes", ""))
        evaluations = [
            self._evaluate_rule_hits_for_row(rule_packs, row, row_index)
            for row_index, row in enumerate(req.rows)
        ]
        return GigaChatRuleEvaluationResponse(rule_packs=rule_packs, evaluations=evaluations)

    @classmethod
    def _format_rule_lines(cls, raw: Any, *, label: str) -> str:
        items = cls._parse_rule_items(raw)
        if not items:
            return f"{label}: список пока пустой."
        lines = [f"{label}:"]
        for item in items:
            if item["name"] and item["description"]:
                lines.append(f"- {item['name']}: {item['description']}")
            elif item["name"]:
                lines.append(f"- {item['name']}")
            else:
                lines.append(f"- {item['description']}")
        return "\n".join(lines)

    @classmethod
    def _format_rule_pack_lines(cls, raw: Any) -> str:
        rule_packs = cls._parse_rule_pack_items(raw)
        if not rule_packs:
            return "Rule packs: список пока пустой."
        lines = ["Rule packs:"]
        for item in rule_packs:
            if not item.enabled:
                continue
            filter_text = (
                "; ".join(f"{rule_filter.field} {rule_filter.op} {rule_filter.value}" for rule_filter in item.filters)
                if item.filters else
                "без бизнес-фильтров"
            )
            keyword_text = ", ".join(item.keywords) if item.keywords else "без keywords"
            action_text = item.target_tag if item.type == "assign_tag" else item.target_topic
            lines.append(
                f"- {item.code} [{item.type}]: поля={', '.join(item.source_fields) or 'не заданы'}; "
                f"фильтры={filter_text}; keywords={keyword_text}; действие={action_text or 'не задано'}."
            )
            if item.description:
                lines.append(f"  Описание: {item.description}")
        return "\n".join(lines)

    @staticmethod
    def _format_row_rule_context(evaluation: GigaChatRuleEvaluationRow) -> str:
        lines: list[str] = []
        if not evaluation.hits:
            lines.append("Rule-based precheck для этой строки не нашел совпадений.")
        else:
            lines.append("Rule-based precheck для этой строки нашел следующие совпадения:")
            for hit in evaluation.hits:
                action = f"tag={hit.target_tag}" if hit.type == "assign_tag" else f"topic={hit.target_topic}"
                lines.append(
                    f"- {hit.code} [{hit.type}] -> {action}; keywords={', '.join(hit.matched_keywords)}; "
                    f"fields={', '.join(hit.matched_fields)}."
                )
            if evaluation.suggested_tags:
                lines.append(f"Предложенные теги по локальным правилам: {', '.join(evaluation.suggested_tags)}")
            if evaluation.suggested_topics:
                lines.append(f"Предложенные переклассификации по локальным правилам: {', '.join(evaluation.suggested_topics)}")
        lines.append(
            "Подтверди или отвергни каждое срабатывание. В confirmed_rule_hits верни подтвержденные коды правил, "
            "в rejected_rule_hits — отклоненные. Если rule-pack предлагает reclass_topic, при подтверждении верни его в reclassified_topic."
        )
        lines.append(
            "Отдельно выполни semantic tag review: локальные совпадения — это подсказка, а не полный результат. "
            "Самостоятельно проверь все разрешенные теги по смыслу в разрешенных source_fields rule-pack'ов: "
            "точные ключи, однокоренные формы, опечатки, искаженные написания, синонимы и косвенное описание ситуации. "
            "Если разрешенный тег подходит без local rule hit, добавь его в model_added_tags. "
            "Если local rule hit ошибочный по контексту или отрицанию, добавь тег в model_rejected_tags и код правила в rejected_rule_hits."
        )
        return "\n".join(lines)

    @staticmethod
    def _placeholder_row(columns: list[str]) -> dict[str, str]:
        if not columns:
            return {"complaint_text": "{{complaint_text}}"}
        return {column: f"{{{{{column}}}}}" for column in columns}

    @classmethod
    def _render_payload_for_row(cls, payload: Any, row: dict[str, Any]) -> Any:
        if isinstance(payload, dict):
            return {key: cls._render_payload_for_row(value, row) for key, value in payload.items()}
        if isinstance(payload, list):
            return [cls._render_payload_for_row(item, row) for item in payload]
        if isinstance(payload, str):
            rendered = payload
            for column, value in row.items():
                rendered = rendered.replace(f"{{{{{column}}}}}", "" if value is None else str(value))
            return rendered
        return payload

    def _compose_final_payload(self, values: dict[str, Any], columns: list[str]) -> dict[str, Any]:
        model = str(values.get("model", self.cfg.llm.model) or self.cfg.llm.model)
        temperature = float(values.get("temperature", getattr(self.cfg.llm, "temperature", 0.2)) or 0.2)
        top_p = float(values.get("top_p", getattr(self.cfg.llm, "top_p", 0.95)) or 0.95)
        max_tokens = int(values.get("max_output_tokens", getattr(self.cfg.llm, "max_output_tokens", 2048)) or 2048)
        system_prompt = str(values.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT) or "").strip() or self.DEFAULT_SYSTEM_PROMPT
        user_prompt_prefix = str(values.get("user_prompt_prefix", getattr(self.cfg.llm, "user_prompt_prefix", "")) or "").strip()
        context_notes = str(values.get("context_notes", getattr(self.cfg.llm, "context_notes", "")) or "").strip()

        class_rules = self._format_rule_lines(values.get("classification_prompt_notes", ""), label="Классификации")
        tag_rules = self._format_rule_lines(values.get("tagging_prompt_notes", ""), label="Теги")
        rule_pack_rules = self._format_rule_pack_lines(values.get("rule_pack_prompt_notes", ""))
        parsed_class_rules = self._parse_rule_items(values.get("classification_prompt_notes", ""))
        parsed_tag_rules = self._parse_rule_items(values.get("tagging_prompt_notes", ""))
        placeholder_row = self._placeholder_row(columns)

        prompt_parts: list[str] = []
        if context_notes:
            prompt_parts.append(context_notes)
        if user_prompt_prefix:
            prompt_parts.append(user_prompt_prefix)
        prompt_parts.append(class_rules)
        prompt_parts.append(tag_rules)
        prompt_parts.append(rule_pack_rules)
        prompt_parts.append(
            "Semantic tag review:\n"
            "- Можно назначать только теги из раздела `Теги` и из действий активных assign_tag rule packs.\n"
            "- Local rule hits являются подсказками, но не ограничивают результат: проверь разрешенные теги независимо.\n"
            "- Для каждого итогового тега укажи `tag_decisions`: tag, decision (`confirmed`, `added`, `rejected`, `not_applicable`), "
            "source (`local_rule`, `llm_semantic`), match_type (`exact`, `stem`, `typo`, `synonym`, `semantic`, `rejected`, `none`), evidence, reason.\n"
            "- `local_tags` = теги, предложенные локальными правилами. `model_added_tags` = разрешенные теги, добавленные моделью без local hit. "
            "`model_rejected_tags` = локальные теги, отвергнутые моделью. `assigned_tags` = финальные теги после подтверждения/добавления/отклонения.\n"
            "- Если тег добавлен по смыслу, evidence должен быть короткой фразой из строки или точным указанием поля, иначе тег не ставь."
        )
        prompt_parts.append(f"Активные колонки для анализа: {', '.join(columns) if columns else 'не выбраны'}")
        prompt_parts.append(
            "Ниже шаблон одной записи. В реальном запросе на место плейсхолдеров будут подставлены значения выбранных колонок:"
        )
        prompt_parts.append(json.dumps(placeholder_row, ensure_ascii=False, indent=2))
        prompt_parts.append(
            "Перед отправкой каждой реальной строки локальный rule-based precheck вычислит совпадения по rule packs и "
            "добавит их в user message. Используй эти совпадения как сильную подсказку, но подтверждай их только если они реально согласуются с текстом строки."
        )

        return {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n\n".join(prompt_parts)},
            ],
        }

    @staticmethod
    def _payload_to_count_input(payload: dict[str, Any]) -> str:
        messages = payload.get("messages")
        if isinstance(messages, list):
            parts: list[str] = []
            for item in messages:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if content in (None, ""):
                    continue
                parts.append(str(content))
            if parts:
                return "\n\n".join(parts)
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def _inject_row_rule_context(cls, payload: dict[str, Any], evaluation: GigaChatRuleEvaluationRow) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        next_payload = dict(payload)
        next_messages: list[dict[str, Any]] = []
        injected = False
        rule_context = cls._format_row_rule_context(evaluation)

        for message in messages:
            if not isinstance(message, dict):
                next_messages.append(message)
                continue
            next_message = dict(message)
            if next_message.get("role") == "user":
                content = str(next_message.get("content", "") or "").strip()
                next_message["content"] = f"{content}\n\n{rule_context}".strip()
                injected = True
            next_messages.append(next_message)

        if not injected:
            next_messages.append({"role": "user", "content": rule_context})

        next_payload["messages"] = next_messages
        return next_payload

    def _build_sheet_preview(self, sheet_name: str, df: pd.DataFrame) -> GigaChatWorkbookSheetPreview:
        columns = [str(c) for c in df.columns]
        preview_columns = columns[: min(len(columns), 6)]
        preview_df = df[preview_columns].head(5).copy() if preview_columns else df.head(5).copy()
        return GigaChatWorkbookSheetPreview(
            name=sheet_name,
            rows_total=int(len(df)),
            column_count=int(len(columns)),
            columns=columns,
            preview_rows=self._frame_to_rows(preview_df),
        )

    @classmethod
    def _frame_to_rows(cls, df: pd.DataFrame) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        work = df.copy()
        work.columns = [str(c) for c in work.columns]
        for _, row in work.iterrows():
            out.append({col: cls._normalize_cell(row[col]) for col in work.columns})
        return out

    @staticmethod
    def _normalize_cell(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    @classmethod
    def _normalize_export_cell(cls, value: Any) -> Any:
        normalized = cls._normalize_cell(value)
        if isinstance(normalized, str):
            text = normalized.strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                try:
                    return datetime.fromisoformat(text).date()
                except Exception:
                    return normalized
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", text):
                try:
                    return datetime.fromisoformat(text.replace(" ", "T"))
                except Exception:
                    return normalized
        return normalized

    @staticmethod
    def _excel_safe_sheet_name(value: str) -> str:
        cleaned = re.sub(r"[:\\\\/?*\\[\\]]", "_", str(value or "Разметка")).strip()
        return (cleaned or "Разметка")[:31]

    @staticmethod
    def _sanitize_for_excel(df: pd.DataFrame, max_len: int = 32767) -> pd.DataFrame:
        out = df.copy()
        cols = out.select_dtypes(include=["object", "string"]).columns
        for col in cols:
            series = out[col]
            mask = series.map(lambda v: isinstance(v, str))
            if not bool(mask.any()):
                continue
            sanitized = series.loc[mask].astype(str).str.replace(_OPENPYXL_ILLEGAL_RE, "", regex=True).str.slice(0, max_len)
            out.loc[mask, col] = sanitized
        return out
