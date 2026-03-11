from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from complaints_trends.config import ProjectConfig
from complaints_trends.novelty_hunt import novelty_hunt


def _cfg(tmp_path: Path) -> ProjectConfig:
    prepared = tmp_path / "prepared.parquet"
    return ProjectConfig.model_validate(
        {
            "input": {
                "input_dir": str(tmp_path),
                "signal_columns": ["subject"],
            },
            "client_first_extraction": {
                "client_markers": ["CLIENT:"],
                "operator_markers": ["OPERATOR:"],
                "chatbot_markers": ["CHATBOT:"],
                "stop_on_markers": ["OPERATOR:", "CHATBOT:"],
            },
            "pii": {},
            "llm": {
                "base_url": "http://localhost:9999",
                "enabled": False,
            },
            "prepare": {
                "output_parquet": str(prepared),
                "pilot_parquet": str(tmp_path / "pilot.parquet"),
                "pilot_review_xlsx": str(tmp_path / "pilot.xlsx"),
            },
            "training": {
                "model_dir": str(tmp_path / "models"),
            },
            "analysis": {
                "novelty_hunt": {
                    "vectorizer_source": "fit_baseline",
                    "select_mode": "top_k",
                    "top_k": 12,
                    "min_cluster_size": 2,
                    "reports_dir": str(tmp_path / "reports"),
                    "exports_dir": str(tmp_path / "exports"),
                    "interim_dir": str(tmp_path / "interim"),
                    "novelty_scope": "both",
                    "candidate_pool": "all",
                }
            },
            "files": {
                "deny_tokens_path": str(tmp_path / "deny.txt"),
                "extra_stopwords_path": str(tmp_path / "stop.txt"),
                "categories_seed_path": str(tmp_path / "cats.yaml"),
            },
        }
    )


def test_novelty_hunt_detects_new_subpattern(tmp_path: Path):
    cfg = _cfg(tmp_path)

    baseline_rows = []
    base_texts = [
        "кнопка не работает в приложении",
        "не открывается экран оплаты",
        "ошибка входа при запуске",
    ]
    for i in range(240):
        txt = base_texts[i % len(base_texts)]
        baseline_rows.append(
            {
                "row_id": f"b{i}",
                "month": "2025-10" if i < 120 else "2025-11",
                "event_time": "2025-10-10 10:00:00",
                "client_first_message": txt,
                "is_complaint_llm": True,
                "complaint_category_llm": "TECHNICAL",
            }
        )
    pd.DataFrame(baseline_rows).to_parquet(cfg.prepare.output_parquet, index=False)

    new_rows = []
    for i in range(40):
        txt = "кнопка не работает в приложении"
        if i % 2 == 0:
            txt = "кнопка не работает после смены способа входа при подтверждении в safari ios 17"
        new_rows.append(
            {
                "row_id": f"n{i}",
                "month": "2025-12",
                "event_time": "2025-12-10 10:00:00",
                "client_first_message": txt,
                "complaint_score": 0.8,
                "is_complaint_pred": True,
                "category_pred": "TECHNICAL",
                "subcategory_pred": "login_issue",
            }
        )
    month_path = Path(cfg.analysis.novelty_hunt.interim_dir) / "month_2025-12.parquet"
    month_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(new_rows).to_parquet(month_path, index=False)

    report_path, state_path, export_path = novelty_hunt(cfg, "2025-12", "2025-10..2025-11", "test")

    assert report_path.exists()
    assert state_path.exists()
    assert export_path.exists()

    state = pd.read_parquet(state_path)
    assert int(state["is_novel"].sum()) > 0
    assert int((state["cluster_id"] != -1).sum()) > 0

    clusters_json = Path(cfg.analysis.novelty_hunt.interim_dir) / "novelty_hunt_clusters_test.json"
    payload = json.loads(clusters_json.read_text(encoding="utf-8"))
    assert payload
    assert payload[0]["top_terms"]
    assert payload[0]["examples"]
