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


def test_prepare_skips_llm_errors_and_collects_them(tmp_path, monkeypatch):
    from complaints_trends import prepare_dataset as pd_mod

    cfg = load_config("configs/project.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.input.input_dir = str(tmp_path / "raw")
    cfg.prepare.output_parquet = str(tmp_path / "all.parquet")
    cfg.prepare.pilot_parquet = str(tmp_path / "pilot.parquet")
    cfg.prepare.pilot_review_xlsx = str(tmp_path / "review.xlsx")
    cfg.analysis.reports_dir = str(tmp_path / "reports")
    cfg.llm.enabled = True
    cfg.input.signal_columns = ["dialog_text", "subject", "channel", "product", "status"]
    cfg.input.dialog_columns = ["dialog_text"]
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {"created_at": "2025-02-09 12:55:29", "dialog_text": "CLIENT: проблема A", "subject": "s", "channel": "chat", "product": "app", "status": "x"},
            {"created_at": "2025-02-10 12:55:29", "dialog_text": "CLIENT: проблема B", "subject": "s", "channel": "chat", "product": "app", "status": "x"},
        ]
    )
    df.to_excel(tmp_path / "raw" / "sample.xlsx", index=False)

    class _OK:
        def model_dump(self):
            return {
                "client_first_message": "ok",
                "short_summary": "ok",
                "is_complaint": False,
                "complaint_category": "OTHER",
                "complaint_subcategory": None,
                "product_area": None,
                "loan_product": "NONE",
                "severity": "low",
                "keywords": ["вопрос", "инфо", "уточнение"],
                "confidence": 0.9,
                "notes": None,
            }

    class _FakeNorm:
        def __init__(self, *args, **kwargs):
            self._n = 0

        def normalize(self, payload):
            self._n += 1
            if self._n == 2:
                raise ValueError("broken row")
            return _OK()

    monkeypatch.setattr(pd_mod, "GigaChatNormalizer", _FakeNorm)

    out = prepare_dataset(cfg, pilot=False, llm_mock=False)
    assert len(out) == 2
    assert out["llm_error"].astype(str).str.contains("LLM_ERROR").sum() == 1
    assert (tmp_path / "reports" / "llm_errors.json").exists()
