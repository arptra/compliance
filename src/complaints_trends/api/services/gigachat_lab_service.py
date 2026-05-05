from __future__ import annotations

import json
import math
import re
import uuid
from io import BytesIO
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd

from ...config import LLMConfig, ProjectConfig
from ...gigachat_api import build_gigachat_transport_client
from ...gigachat_mtls import SYSTEM_PROMPT
from ...taxonomy import load_taxonomy
from ..schemas import (
    GigaChatAnnotatedExportRequest,
    GigaChatFinalPromptRequest,
    GigaChatFinalPromptResponse,
    GigaChatLabRowRunRequest,
    GigaChatLabRowRunResponse,
    GigaChatLabSettingField,
    GigaChatLabSettingOption,
    GigaChatLabSettingsResponse,
    GigaChatLabSettingsUpdateRequest,
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
        "product_area, loan_product, severity, keywords, assigned_tags, confidence, notes, evidence_columns. "
        "Сначала определи, является ли запись жалобой. Затем выбери основную категорию, при необходимости подкатегорию, "
        "назначь только разрешенные теги, оцени severity и confidence, а в notes кратко объясни решение. "
        "Если данных недостаточно, оставляй поле пустым и коротко объясняй неоднозначность в notes."
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
    ]

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self.base_dir = Path(cfg.analysis.pattern_monitoring.interim_dir) / "gigachat_lab"
        self.uploads_dir = self.base_dir / "uploads"
        self.settings_path = self.base_dir / "settings.json"
        self.final_prompt_path = self.base_dir / "final_prompt_snapshot.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _serialize_rule_items(items: list[dict[str, str]]) -> str:
        return json.dumps(items, ensure_ascii=False, indent=2)

    def _default_classification_rules(self) -> str:
        taxonomy = load_taxonomy(self.cfg.files.categories_seed_path)
        raw_categories = taxonomy.get("raw", {}).get("categories", {}) or {}
        items: list[dict[str, str]] = []
        for code, body in raw_categories.items():
            label = code
            sub_labels: list[str] = []
            if isinstance(body, dict):
                label = str(body.get("label_ru", code) or code)
                subcategories = body.get("subcategories", {}) or {}
                if isinstance(subcategories, dict):
                    for subcode, meta in subcategories.items():
                        if isinstance(meta, dict):
                            sub_label = str(meta.get("label_ru", subcode) or subcode)
                        else:
                            sub_label = str(meta)
                        sub_labels.append(f"{subcode} ({sub_label})")
            description_parts = [label]
            if sub_labels:
                description_parts.append(f"Подкатегории: {', '.join(sub_labels)}.")
            if code == "OTHER":
                description_parts.append("Используй только если ни одна другая категория явно не подходит.")
            else:
                description_parts.append("Выбирай эту категорию, если основная суть жалобы относится именно сюда.")
            items.append({"name": code, "description": " ".join(description_parts)})
        return self._serialize_rule_items(items)

    def _default_tag_rules(self) -> str:
        items = [
            {"name": "mobile_app", "description": "Назначай, когда проблема проявляется в мобильном приложении или app store версии клиента."},
            {"name": "web_channel", "description": "Назначай, когда проблема относится к веб-версии, браузеру или личному кабинету в вебе."},
            {"name": "login", "description": "Назначай, когда обращение связано со входом, авторизацией или недоступностью аккаунта."},
            {"name": "otp", "description": "Назначай, когда упоминаются SMS, push или одноразовые коды подтверждения."},
            {"name": "payment", "description": "Назначай, когда проблема относится к оплате, списанию, комиссии или платежной операции."},
            {"name": "transfer", "description": "Назначай, когда речь идет о переводе между счетами, по реквизитам или по номеру телефона."},
            {"name": "refund", "description": "Назначай, когда клиент ждет возврат денег, отмену операции или откат платежа."},
            {"name": "delay", "description": "Назначай, когда ключевая проблема — задержка, зависание, отсутствие статуса или долгое ожидание."},
            {"name": "fee", "description": "Назначай, когда клиент жалуется на комиссию, тариф, проценты, штрафы или некорректный расчет."},
            {"name": "ui_bug", "description": "Назначай, когда проблема связана с кнопками, экраном, формой, интерфейсом или визуальной ошибкой."},
            {"name": "support_quality", "description": "Назначай, когда клиент жалуется на работу поддержки, сроки ответа или качество консультации."},
            {"name": "security", "description": "Назначай, когда есть признаки взлома, мошенничества, блокировки по безопасности или подозрительных операций."},
            {"name": "kyc", "description": "Назначай, когда проблема связана с документами, идентификацией, анкетой или персональными данными."},
            {"name": "loan", "description": "Назначай, когда обращение относится к кредиту, кредитной карте, графику, ставке, страховке или погашению."},
            {"name": "notifications", "description": "Назначай, когда проблема касается SMS, push, email или других уведомлений."},
            {"name": "delivery", "description": "Назначай, когда жалоба связана с доставкой карты, документов, товара или курьером."},
            {"name": "integration", "description": "Назначай, когда сбой вызван сторонним сервисом, интеграцией, платежным шлюзом или внешним контуром."},
            {"name": "after_update", "description": "Назначай, когда проблема появилась после обновления приложения, версии клиента или релиза."},
        ]
        return self._serialize_rule_items(items)

    def _default_values(self) -> dict[str, Any]:
        llm = self.cfg.llm.model_dump()
        default_system_prompt = str(llm.get("system_prompt", "") or "").strip()
        if not default_system_prompt or default_system_prompt == LLMConfig.model_fields["system_prompt"].default:
            default_system_prompt = self.DEFAULT_SYSTEM_PROMPT

        default_user_prompt_prefix = str(llm.get("user_prompt_prefix", "") or "").strip() or self.DEFAULT_USER_PROMPT_PREFIX
        default_context_notes = str(llm.get("context_notes", "") or "").strip() or self.DEFAULT_CONTEXT_NOTES
        default_classification_notes = str(llm.get("classification_prompt_notes", "") or "").strip() or self._default_classification_rules()
        default_tagging_notes = str(llm.get("tagging_prompt_notes", "") or "").strip() or self._default_tag_rules()
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
        saved_at = None
        if isinstance(saved_at_raw, str) and saved_at_raw.strip():
            try:
                saved_at = datetime.fromisoformat(saved_at_raw)
            except Exception:
                saved_at = None
        return GigaChatLabSettingsResponse(
            title="GigaChat Lab Settings",
            fields=fields,
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
        rendered_payload = self._render_payload_for_row(base_payload, req.row)

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
        )

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
            }
            for column in source_columns:
                export_row[column] = self._normalize_export_cell(row.source_row.get(column))
            export_rows.append(export_row)

        ordered_columns = ["Класс", "Теги", *source_columns]
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
            df = pd.read_csv(stored_path, dtype=object)
            sheet_name = "data"
        else:
            sheet_name = req.sheet_name
            available_sheets = {str(sheet.get("name", "")) for sheet in meta.get("sheets", [])}
            if sheet_name not in available_sheets:
                raise ValueError(f"Sheet not found in workbook: {sheet_name}")
            df = pd.read_excel(stored_path, sheet_name=sheet_name, dtype=object)

        columns = [str(c) for c in df.columns]
        rows = self._frame_to_rows(df.head(min(max(int(req.row_limit), 1), 1000)))
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
            df = pd.read_csv(stored_path, dtype=object)
            sheet = self._build_sheet_preview("data", df)
            return {
                "upload_id": stored_path.parent.name,
                "filename": filename,
                "file_format": "csv",
                "sheet_count": 1,
                "sheets": [sheet.model_dump()],
            }

        excel = pd.ExcelFile(stored_path)
        sheets = [self._build_sheet_preview(sheet_name, pd.read_excel(excel, sheet_name=sheet_name, dtype=object)).model_dump() for sheet_name in excel.sheet_names]
        return {
            "upload_id": stored_path.parent.name,
            "filename": filename,
            "file_format": "excel",
            "sheet_count": len(sheets),
            "sheets": sheets,
        }

    @staticmethod
    def _extract_supported_file_from_zip(filename: str, content: bytes) -> tuple[str, bytes]:
        supported_suffixes = (".xlsx", ".xls", ".xlsm", ".csv")
        with ZipFile(BytesIO(content)) as archive:
            candidates = [
                info for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(supported_suffixes)
            ]
            if not candidates:
                raise ValueError(
                    f"В архиве {filename} не найдено поддерживаемых файлов Excel/CSV. "
                    "Ожидается .xlsx, .xls, .xlsm или .csv."
                )
            selected = candidates[0]
            return selected.filename, archive.read(selected)

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
        prompt_parts.append(f"Активные колонки для анализа: {', '.join(columns) if columns else 'не выбраны'}")
        prompt_parts.append(
            "Ниже шаблон одной записи. В реальном запросе на место плейсхолдеров будут подставлены значения выбранных колонок:"
        )
        prompt_parts.append(json.dumps(placeholder_row, ensure_ascii=False, indent=2))

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
