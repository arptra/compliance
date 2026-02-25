from pathlib import Path

import pandas as pd

from complaints_trends.config import load_config
from complaints_trends.train_models import train


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
            {"row_id": "r1", "event_time": "2025-01-01", "client_first_message": "не работает приложение", "is_complaint_llm": True, "complaint_category_llm": "TECHNICAL"},
            {"row_id": "r2", "event_time": "2025-01-02", "client_first_message": "вопрос по тарифу", "is_complaint_llm": False, "complaint_category_llm": "OTHER"},
            {"row_id": "r3", "event_time": "2025-01-03", "client_first_message": "ошибка оплаты", "is_complaint_llm": True, "complaint_category_llm": "PAYMENTS"},
            {"row_id": "r4", "event_time": "2025-01-04", "client_first_message": "спасибо", "is_complaint_llm": False, "complaint_category_llm": "OTHER"},
            {"row_id": "r5", "event_time": "2025-01-05", "client_first_message": "не зачислили платеж", "is_complaint_llm": True, "complaint_category_llm": "PAYMENTS"},
            {"row_id": "r6", "event_time": "2025-01-06", "client_first_message": "не могу войти", "is_complaint_llm": True, "complaint_category_llm": "TECHNICAL"},
            {"row_id": "r7", "event_time": "2025-01-07", "client_first_message": "какой статус заявки", "is_complaint_llm": False, "complaint_category_llm": "OTHER"},
            {"row_id": "r8", "event_time": "2025-01-08", "client_first_message": "пропал перевод", "is_complaint_llm": True, "complaint_category_llm": "PAYMENTS"},
        ]
    )
    df["event_time"] = pd.to_datetime(df["event_time"])
    df.to_parquet(cfg.prepare.output_parquet, index=False)

    metrics = train(cfg)
    assert "complaint_f1" in metrics

    assert Path("reports/training_report.html").exists()
    assert Path("reports/training_report.md").exists()
    assert Path("reports/training_predicted_complaint_distribution.png").exists()
    assert Path("reports/training_predicted_category_distribution.png").exists()

    md = Path("reports/training_report.md").read_text(encoding="utf-8")
    assert "Кратко" in md
    assert "Основные метрики" in md
