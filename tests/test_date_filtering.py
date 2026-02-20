import pandas as pd

from complaints_trends.prepare_dataset import prepare_dataset
from complaints_trends.config import load_config


def test_prepare_filters_by_event_time_range(tmp_path):
    cfg = load_config("configs/project.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.input.input_dir = str(tmp_path / "raw")
    cfg.prepare.output_parquet = str(tmp_path / "all.parquet")
    cfg.prepare.pilot_parquet = str(tmp_path / "pilot.parquet")
    cfg.prepare.pilot_review_xlsx = str(tmp_path / "review.xlsx")
    cfg.llm.enabled = False
    cfg.input.signal_columns = ["dialog_text", "subject", "channel", "product", "status"]
    cfg.input.dialog_columns = ["dialog_text"]
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {"created_at": "2025-01-09 12:55:29", "dialog_text": "CLIENT: проблема", "subject": "s", "channel": "chat", "product": "app", "status": "x"},
            {"created_at": "2025-02-09 12:55:29", "dialog_text": "CLIENT: вопрос", "subject": "s", "channel": "chat", "product": "app", "status": "x"},
        ]
    )
    df.to_excel(tmp_path / "raw" / "sample.xlsx", index=False)

    out = prepare_dataset(cfg, pilot=False, llm_mock=True, date_from="2025-02-01 00:00:00", date_to="2025-02-28 23:59:59")
    assert len(out) == 1
    assert out.iloc[0]["event_time"].month == 2


def test_prepare_pilot_empty_range_does_not_crash(tmp_path):
    cfg = load_config("configs/project.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.input.input_dir = str(tmp_path / "raw")
    cfg.prepare.output_parquet = str(tmp_path / "all.parquet")
    cfg.prepare.pilot_parquet = str(tmp_path / "pilot.parquet")
    cfg.prepare.pilot_review_xlsx = str(tmp_path / "review.xlsx")
    cfg.llm.enabled = False
    cfg.input.signal_columns = ["dialog_text", "subject", "channel", "product", "status"]
    cfg.input.dialog_columns = ["dialog_text"]
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([
        {"created_at": "2025-01-09 12:55:29", "dialog_text": "CLIENT: проблема", "subject": "s", "channel": "chat", "product": "app", "status": "x"},
    ])
    df.to_excel(tmp_path / "raw" / "sample.xlsx", index=False)

    out = prepare_dataset(
        cfg,
        pilot=True,
        llm_mock=True,
        date_from="2025-02-01 00:00:00",
        date_to="2025-02-28 23:59:59",
        limit=100,
    )
    assert len(out) == 0
