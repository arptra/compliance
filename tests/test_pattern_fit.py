from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from complaints_trends.config import ProjectConfig
from complaints_trends.pattern_fit import run_pattern_fit


def _cfg(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "input": {"input_dir": str(tmp_path), "signal_columns": ["subject"]},
            "client_first_extraction": {
                "client_markers": ["CLIENT:"],
                "operator_markers": ["OPERATOR:"],
                "chatbot_markers": ["CHATBOT:"],
                "stop_on_markers": ["OPERATOR:", "CHATBOT:"],
            },
            "pii": {},
            "llm": {"base_url": "http://localhost", "enabled": False},
            "prepare": {
                "output_parquet": str(tmp_path / "prepared.parquet"),
                "pilot_parquet": str(tmp_path / "pilot.parquet"),
                "pilot_review_xlsx": str(tmp_path / "pilot.xlsx"),
            },
            "training": {"model_dir": str(tmp_path / "models")},
            "analysis": {
                "pattern_monitoring": {
                    "vectorizer_source": "fit_normal",
                    "normal_period": "2025-01..2025-08",
                    "event_period": "2025-11..2025-12",
                    "interim_dir": str(tmp_path / "interim"),
                    "exports_dir": str(tmp_path / "exports"),
                    "reports_dir": str(tmp_path / "reports"),
                    "min_normal_rows_per_category": 10,
                    "min_event_rows_per_category": 6,
                    "min_total_seeds_per_category": 4,
                    "min_cluster_size": 2,
                    "anomaly_min_excess_total": 1,
                    "anomaly_min_event_days": 1,
                }
            },
            "files": {
                "deny_tokens_path": str(tmp_path / "deny.txt"),
                "extra_stopwords_path": str(tmp_path / "stop.txt"),
                "categories_seed_path": str(tmp_path / "cats.yaml"),
            },
        }
    )


def test_pattern_fit_builds_growth_and_clusters(tmp_path: Path):
    cfg = _cfg(tmp_path)
    rows = []
    normal_text = {
        "login": ["не работает кнопка входа", "не проходит подтверждение"],
        "payment": ["ошибка оплаты", "не проходит оплата"],
        "transfer": ["кнопка перевода недоступна", "не отправляется перевод"],
    }
    for m in ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08"]:
        for cat, texts in normal_text.items():
            for i in range(20):
                rows.append({
                    "row_id": f"n-{m}-{cat}-{i}",
                    "month": m,
                    "event_time": f"{m}-10 10:00:00",
                    "client_first_message": texts[i % len(texts)],
                    "is_complaint_llm": True,
                    "complaint_category_llm": cat,
                })
    event_login = [
        "после ввода кода подтверждения кнопка входа не активна",
        "на шаге подтверждения входа кнопка не нажимается",
        "после смс нельзя завершить вход",
    ]
    for m in ["2025-11", "2025-12"]:
        for i in range(50):
            txt = event_login[i % len(event_login)] if i < 30 else "не работает кнопка входа"
            rows.append({
                "row_id": f"e-{m}-login-{i}",
                "month": m,
                "event_time": f"{m}-11 10:00:00",
                "client_first_message": txt,
                "is_complaint_llm": True,
                "complaint_category_llm": "login",
            })
    for cat in ["payment", "transfer"]:
        for m in ["2025-11", "2025-12"]:
            for i in range(20):
                rows.append({
                    "row_id": f"e-{m}-{cat}-{i}",
                    "month": m,
                    "event_time": f"{m}-12 10:00:00",
                    "client_first_message": normal_text[cat][i % 2],
                    "is_complaint_llm": True,
                    "complaint_category_llm": cat,
                })
    pd.DataFrame(rows).to_parquet(cfg.prepare.output_parquet, index=False)

    report, fit_bundle, growth_path = run_pattern_fit(cfg, tag="t1", normal_period="2025-01..2025-08", event_period="2025-11..2025-12", label_source="llm")

    assert report.exists()
    assert fit_bundle.exists()
    growth = pd.read_parquet(growth_path)
    assert not growth.empty
    assert "login" in set(growth[growth["is_candidate"] == True]["category"])

    seed_pool = pd.read_parquet(Path(cfg.analysis.pattern_monitoring.interim_dir) / "pattern_fit_t1" / "seed_pool.parquet")
    assert (seed_pool["category"] == "login").any()

    clusters = json.loads((Path(cfg.analysis.pattern_monitoring.interim_dir) / "pattern_fit_t1" / "cluster_profiles.json").read_text(encoding="utf-8"))
    login_entry = next(x for x in clusters if x["category"] == "login")
    assert len(login_entry["clusters"]) >= 1


def test_pattern_fit_falls_back_to_pred_labels_when_llm_labels_missing(tmp_path: Path):
    cfg = _cfg(tmp_path)
    rows = []
    for m in ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08"]:
        for i in range(20):
            rows.append(
                {
                    "row_id": f"n-{m}-{i}",
                    "month": m,
                    "event_time": f"{m}-10 10:00:00",
                    "client_first_message": "не работает кнопка входа",
                    "is_complaint_pred": True,
                    "category_pred": "login",
                }
            )
    for m in ["2025-11", "2025-12"]:
        for i in range(45):
            rows.append(
                {
                    "row_id": f"e-{m}-{i}",
                    "month": m,
                    "event_time": f"{m}-11 10:00:00",
                    "client_first_message": "после смс нельзя завершить вход",
                    "is_complaint_pred": True,
                    "category_pred": "login",
                }
            )
    pd.DataFrame(rows).to_parquet(cfg.prepare.output_parquet, index=False)

    _, fit_bundle_path, _ = run_pattern_fit(cfg, tag="t-fallback", normal_period="2025-01..2025-08", event_period="2025-11..2025-12", label_source="llm")

    fit_bundle = joblib.load(fit_bundle_path)
    assert fit_bundle["requested_label_source"] == "llm"
    assert fit_bundle["label_source"] == "pred"
