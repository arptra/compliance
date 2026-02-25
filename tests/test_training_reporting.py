from pathlib import Path

import pandas as pd

from complaints_trends.config import load_config
import complaints_trends.train_models as train_models
from complaints_trends.train_models import _build_time_scatter_frame, _normalize_subcategory_by_taxonomy, train


def test_train_generates_human_report_and_charts(tmp_path):
    cfg = load_config("configs/project.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.prepare.output_parquet = str(tmp_path / "prepared.parquet")
    cfg.prepare.pilot_review_xlsx = str(tmp_path / "missing_review.xlsx")
    cfg.training.model_dir = str(tmp_path / "models")
    cfg.training.vectorizer.min_df = 1
    cfg.training.vectorizer.max_df = 1.0

    df = pd.DataFrame(
        [
            {"row_id": "r1", "event_time": "2025-01-01", "client_first_message": "не работает приложение", "is_complaint_llm": True, "complaint_category_llm": "TECHNICAL", "complaint_subcategory_llm": "app_crash"},
            {"row_id": "r2", "event_time": "2025-01-02", "client_first_message": "вопрос по тарифу", "is_complaint_llm": False, "complaint_category_llm": "OTHER", "complaint_subcategory_llm": "UNKNOWN"},
            {"row_id": "r3", "event_time": "2025-01-03", "client_first_message": "ошибка оплаты", "is_complaint_llm": True, "complaint_category_llm": "PAYMENTS_TRANSFERS", "complaint_subcategory_llm": "payment_error"},
            {"row_id": "r4", "event_time": "2025-01-04", "client_first_message": "спасибо", "is_complaint_llm": False, "complaint_category_llm": "OTHER", "complaint_subcategory_llm": "UNKNOWN"},
            {"row_id": "r5", "event_time": "2025-01-05", "client_first_message": "не зачислили платеж", "is_complaint_llm": True, "complaint_category_llm": "PAYMENTS_TRANSFERS", "complaint_subcategory_llm": "transfer_delay"},
            {"row_id": "r6", "event_time": "2025-01-06", "client_first_message": "не могу войти", "is_complaint_llm": True, "complaint_category_llm": "TECHNICAL", "complaint_subcategory_llm": "login_issue"},
            {"row_id": "r7", "event_time": "2025-01-07", "client_first_message": "какой статус заявки", "is_complaint_llm": False, "complaint_category_llm": "OTHER", "complaint_subcategory_llm": "UNKNOWN"},
            {"row_id": "r8", "event_time": "2025-01-08", "client_first_message": "пропал перевод", "is_complaint_llm": True, "complaint_category_llm": "PAYMENTS_TRANSFERS", "complaint_subcategory_llm": "transfer_reversed"},
        ]
    )
    df["event_time"] = pd.to_datetime(df["event_time"])
    df.to_parquet(cfg.prepare.output_parquet, index=False)

    metrics = train(cfg)
    assert "complaint_f1" in metrics
    assert "subcategory_macro_f1" in metrics

    assert Path("reports/training_report.html").exists()
    assert Path("reports/training_report.md").exists()
    assert Path("reports/training_predicted_complaint_distribution.png").exists()
    assert Path("reports/training_category_hist_ru.png").exists()
    assert Path("reports/training_subcategory_hist_ru.png").exists()
    assert Path("reports/training_category_subcategory_scatter_ru.png").exists()

    md = Path("reports/training_report.md").read_text(encoding="utf-8")
    assert "Кратко" in md
    assert "Основные метрики" in md
    assert Path(cfg.training.model_dir, "subcategory_model.joblib").exists()
    assert Path(cfg.training.model_dir, "subcategory_label_encoder.joblib").exists()


def test_build_time_scatter_frame_respects_validation_window():
    df = pd.DataFrame(
        [
            {"event_time": "2025-01-01 10:00:00", "complaint_category_llm": "TECH", "complaint_subcategory_llm": "LOGIN"},
            {"event_time": "2025-01-02 10:00:00", "complaint_category_llm": "TECH", "complaint_subcategory_llm": "APP"},
            {"event_time": "2025-01-03 10:00:00", "complaint_category_llm": "PAY", "complaint_subcategory_llm": "CARD"},
        ]
    )
    out = _build_time_scatter_frame(df, val_from="2025-01-02 00:00:00", val_to="2025-01-02 23:59:59")
    assert len(out) == 1
    assert out.iloc[0]["day"].strftime("%Y-%m-%d") == "2025-01-02"
    assert int(out.iloc[0]["category_count"]) == 1
    assert int(out.iloc[0]["subcategory_count"]) == 1



def test_normalize_subcategory_by_taxonomy_replaces_unknown_values():
    df = pd.DataFrame([
        {"complaint_category_llm": "TECHNICAL", "complaint_subcategory_llm": "login_issue"},
        {"complaint_category_llm": "TECHNICAL", "complaint_subcategory_llm": "inheritance transfer"},
    ])
    taxonomy = {"subcategories_by_category": {"TECHNICAL": ["login_issue", "app_crash"]}}
    out = _normalize_subcategory_by_taxonomy(df, taxonomy)
    assert out.iloc[0] == "login_issue"
    assert out.iloc[1] == "UNKNOWN"


def test_train_uses_full_period_for_charts(monkeypatch, tmp_path):
    cfg = load_config("configs/project.yaml").model_copy(deep=True)
    cfg.prepare.output_parquet = str(tmp_path / "prepared.parquet")
    cfg.prepare.pilot_review_xlsx = str(tmp_path / "missing_review.xlsx")
    cfg.training.model_dir = str(tmp_path / "models")
    cfg.training.vectorizer.min_df = 1
    cfg.training.vectorizer.max_df = 1.0
    cfg.training.validation.split_mode = "time"
    cfg.training.validation.val_from = "2025-03-25 00:00:00"
    cfg.training.validation.val_to = "2025-03-31 23:59:59"

    rows = []
    for i in range(1, 31):
        rows.append(
            {
                "row_id": f"r{i}",
                "event_time": f"2025-03-{i:02d}",
                "client_first_message": "ошибка платежа" if i % 2 == 0 else "просто вопрос",
                "is_complaint_llm": i % 2 == 0,
                "complaint_category_llm": "PAYMENTS_TRANSFERS" if i % 2 == 0 else "OTHER",
                "complaint_subcategory_llm": "payment_error" if i % 2 == 0 else "UNKNOWN",
            }
        )
    df = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df.to_parquet(cfg.prepare.output_parquet, index=False)

    captured = {}

    def _fake_save_training_charts(ycat_all, subcat_all, pred_bin_all, out_dir, complaints_df):
        captured["ycat_all_len"] = len(ycat_all)
        captured["subcat_all_len"] = len(subcat_all)
        captured["pred_bin_all_len"] = len(pred_bin_all)
        captured["complaints_df_len"] = len(complaints_df)
        captured["days"] = complaints_df["event_time"].dt.day.nunique()
        return {
            "binary_distribution": "reports/a.png",
            "category_distribution": "reports/b.png",
            "subcategory_distribution": "reports/c.png",
            "category_subcategory_scatter": "reports/d.png",
        }

    monkeypatch.setattr(train_models, "_save_training_charts", _fake_save_training_charts)
    train(cfg)

    assert captured["pred_bin_all_len"] == len(df)
    assert captured["complaints_df_len"] == int(df["is_complaint_llm"].sum())
    assert captured["ycat_all_len"] == int(df["is_complaint_llm"].sum())
    assert captured["subcat_all_len"] == int(df["is_complaint_llm"].sum())
    assert captured["days"] == int(df[df["is_complaint_llm"]]["event_time"].dt.day.nunique())
