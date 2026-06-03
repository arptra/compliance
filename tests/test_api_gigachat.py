from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from complaints_trends.api import create_app


def _setup(tmp_path: Path):
    data = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    data["prepare"]["output_parquet"] = str(tmp_path / "all_prepared.parquet")
    data.setdefault("analysis", {}).setdefault("pattern_monitoring", {})["interim_dir"] = str(tmp_path)

    ca = tmp_path / "ca.pem"
    cert = tmp_path / "client.pem"
    key = tmp_path / "client.key"
    auth_key = tmp_path / "key"
    for path in (ca, cert, key, auth_key):
        path.write_text("stub", encoding="utf-8")

    data["llm"]["ca_bundle_file"] = str(ca)
    data["llm"]["cert_file"] = str(cert)
    data["llm"]["key_file"] = str(key)
    data["llm"]["authorization_key_file"] = str(auth_key)

    cfg = tmp_path / "project.yaml"
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return cfg


def test_gigachat_status_exposes_mtls_and_token(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    response = client.get("/api/gigachat/status")
    assert response.status_code == 200
    payload = response.json()
    names = {item["name"] for item in payload["transports"]}
    assert names == {"mtls", "token"}
    assert any(item["name"] == "mtls" and item["ready"] is True for item in payload["transports"])
    assert any(item["name"] == "token" and item["ready"] is True for item in payload["transports"])


def test_gigachat_probe_uses_selected_transport(monkeypatch, tmp_path: Path):
    class FakeTransportClient:
        def list_models(self):
            return ["GigaChat", "GigaChat-Pro"]

    monkeypatch.setattr(
        "complaints_trends.api.services.gigachat_connection_service.build_gigachat_transport_client",
        lambda cfg, transport=None: FakeTransportClient(),
    )

    client = TestClient(create_app(str(_setup(tmp_path))))
    response = client.post("/api/gigachat/probe", json={"transport": "token"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "token"
    assert payload["ok"] is True
    assert payload["models"] == ["GigaChat", "GigaChat-Pro"]


def test_gigachat_status_keeps_token_ready_when_ca_bundle_missing(tmp_path: Path):
    cfg = _setup(tmp_path)
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    Path(data["llm"]["ca_bundle_file"]).unlink()
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    client = TestClient(create_app(str(cfg)))
    response = client.get("/api/gigachat/status")
    assert response.status_code == 200
    payload = response.json()
    token_status = next(item for item in payload["transports"] if item["name"] == "token")

    assert token_status["ready"] is True
    assert [artifact["label"] for artifact in token_status["artifacts"]] == ["Authorization key"]
    assert token_status["message"] == "Готово к использованию."


def test_gigachat_lab_settings_expose_only_request_fields(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    response = client.get("/api/gigachat/lab/settings")
    assert response.status_code == 200
    payload = response.json()
    keys = {field["key"] for field in payload["fields"]}
    fields = {field["key"]: field for field in payload["fields"]}

    assert {"model", "temperature", "top_p", "max_output_tokens", "system_prompt", "user_prompt_prefix", "context_notes"} <= keys
    assert {"classification_prompt_notes", "tagging_prompt_notes"} <= keys
    assert "cert_file" not in keys
    assert "authorization_key_file" not in keys
    assert "mode" not in keys
    assert "валидный JSON" in str(fields["system_prompt"]["value"])
    assert "Одна строка таблицы = одно обращение" in str(fields["context_notes"]["value"])
    assert "TECHNICAL" in str(fields["classification_prompt_notes"]["value"])
    assert "mobile_app" in str(fields["tagging_prompt_notes"]["value"])


def test_gigachat_final_prompt_uses_rules_and_columns(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    response = client.post(
        "/api/gigachat/lab/final-prompt",
        json={
            "values": {
                "model": "GigaChat-2-Max",
                "temperature": 0.15,
                "top_p": 0.9,
                "max_output_tokens": 1500,
                "system_prompt": "SYSTEM CUSTOM",
                "context_notes": "Размечаем обращения розничного банка.",
                "user_prompt_prefix": "Смотри на все поля, а не только на текст звонка.",
                "classification_prompt_notes": '[{"name":"LOGIN_ISSUE","description":"Проблемы со входом и подтверждением."}]',
                "tagging_prompt_notes": '[{"name":"otp","description":"Проблема связана с кодом подтверждения."}]',
            },
            "columns": ["Во. ID вопроса", "Обр. Текст чата", "Во. Продукт"],
        },
    )
    assert response.status_code == 200

    payload = response.json()["payload"]
    assert payload["model"] == "GigaChat-2-Max"
    assert payload["temperature"] == 0.15
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 1500

    system_message = payload["messages"][0]["content"]
    user_message = payload["messages"][1]["content"]

    assert system_message == "SYSTEM CUSTOM"
    assert "Размечаем обращения розничного банка." in user_message
    assert "Смотри на все поля, а не только на текст звонка." in user_message
    assert "LOGIN_ISSUE" in user_message
    assert "otp" in user_message
    assert "{{Во. ID вопроса}}" in user_message
    assert "{{Обр. Текст чата}}" in user_message
    assert '"assigned_tags"' in user_message


def test_gigachat_final_prompt_save_writes_snapshot(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    response = client.post(
        "/api/gigachat/lab/final-prompt/save",
        json={
            "values": {
                "classification_prompt_notes": '[{"name":"PAYMENT","description":"Ошибки платежей."}]',
                "tagging_prompt_notes": '[{"name":"mobile","description":"Проблема только в мобильном канале."}]',
            },
            "columns": ["Обр. Текст чата"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["saved"] is True
    assert payload["saved_path"]
    assert Path(payload["saved_path"]).exists()


def test_gigachat_rule_pack_keyword_keeps_trailing_space(tmp_path: Path):
    client = TestClient(create_app(str(_setup(tmp_path))))
    rules = [
        {
            "code": "SVO_SPACE",
            "enabled": True,
            "type": "assign_tag",
            "source_fields": ["text"],
            "keywords": ["СВО "],
            "filters": [],
            "target_tag": "SVO_SPACE",
            "target_topic": None,
        }
    ]

    response = client.post(
        "/api/gigachat/lab/rule-packs/evaluate",
        json={
            "values": {"rule_pack_prompt_notes": json.dumps(rules, ensure_ascii=False)},
            "rows": [
                {"text": "Очень своеобразный кейс."},
                {"text": "Участник СВО получил отсрочку."},
                {"text": "Участник СВО, получил отсрочку."},
            ],
        },
    )

    assert response.status_code == 200
    rows = response.json()["evaluations"]
    codes_by_row = [{hit["code"] for hit in row["hits"]} for row in rows]
    assert codes_by_row == [set(), {"SVO_SPACE"}, set()]


def test_gigachat_run_row_renders_payload_and_returns_model_output(monkeypatch, tmp_path: Path):
    class FakeTransportClient:
        def chat(self, payload):
            class _Msg:
                def __init__(self, content):
                    self.content = content

            class _Choice:
                def __init__(self, content):
                    self.message = _Msg(content)

            class _Resp:
                def __init__(self, content):
                    self.choices = [_Choice(content)]

            assert "Текст клиента: не приходит смс" in payload["messages"][1]["content"]
            return _Resp('{"ok": true, "category": "LOGIN_ISSUE"}')

    monkeypatch.setattr(
        "complaints_trends.api.services.gigachat_lab_service.build_gigachat_transport_client",
        lambda cfg, transport=None: FakeTransportClient(),
    )

    client = TestClient(create_app(str(_setup(tmp_path))))
    response = client.post(
        "/api/gigachat/lab/run-row",
        json={
            "transport": "token",
            "values": {},
            "columns": ["text"],
            "row": {"text": "Текст клиента: не приходит смс"},
            "payload_override": {
                "model": "GigaChat-2-Max",
                "messages": [
                    {"role": "system", "content": "SYSTEM"},
                    {"role": "user", "content": "Проверь строку: {{text}}"},
                ],
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["transport"] == "token"
    assert payload["parse_ok"] is True
    assert payload["response_json"]["category"] == "LOGIN_ISSUE"
    assert "Текст клиента: не приходит смс" in payload["request_payload"]["messages"][1]["content"]
