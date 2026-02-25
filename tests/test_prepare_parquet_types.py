import pandas as pd

from complaints_trends.config import load_config
from complaints_trends.prepare_dataset import prepare_dataset


def test_prepare_handles_mixed_object_column_for_parquet(tmp_path):
    cfg = load_config("configs/project.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.input.input_dir = str(tmp_path / "raw")
    cfg.prepare.output_parquet = str(tmp_path / "all.parquet")
    cfg.prepare.pilot_parquet = str(tmp_path / "pilot.parquet")
    cfg.prepare.pilot_review_xlsx = str(tmp_path / "review.xlsx")
    cfg.llm.enabled = False
    cfg.input.signal_columns = ["dialog_text", "Во Описание", "subject", "channel", "product", "status"]
    cfg.input.dialog_columns = ["dialog_text", "Во Описание"]
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "created_at": "2025-01-09 12:55:29",
                "dialog_text": "CLIENT: проблема",
                "Во Описание": 123,
                "subject": "s",
                "channel": "chat",
                "product": "app",
                "status": "x",
            },
            {
                "created_at": "2025-01-10 12:55:29",
                "dialog_text": "CLIENT: вопрос",
                "Во Описание": "текст",
                "subject": "s",
                "channel": "chat",
                "product": "app",
                "status": "x",
            },
        ]
    )
    df.to_excel(tmp_path / "raw" / "sample.xlsx", index=False)

    out = prepare_dataset(cfg, pilot=False, llm_mock=True)

    assert len(out) == 2
    assert (tmp_path / "all.parquet").exists()
    loaded = pd.read_parquet(tmp_path / "all.parquet")
    assert loaded["Во Описание"].map(type).eq(str).all()
