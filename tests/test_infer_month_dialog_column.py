from pathlib import Path

import numpy as np
import pandas as pd

from complaints_trends.config import load_config
from complaints_trends.infer_month import _enforce_subcategory_taxonomy, _predict_subcategories_by_category, _select_infer_text_field, infer_month


class _FakeVec:
    def transform(self, texts):
        return np.zeros((len(texts), 1))


class _FakeComplaintModel:
    def predict_proba(self, x):
        n = x.shape[0]
        return np.tile(np.array([[0.2, 0.8]]), (n, 1))


class _FakeCatModel:
    def predict(self, x):
        return np.zeros(x.shape[0], dtype=int)


class _FakeEnc:
    def inverse_transform(self, x):
        return np.array(["OTHER"] * len(x), dtype=object)


def test_infer_month_works_when_dialog_column_is_none(tmp_path, monkeypatch):
    cfg = load_config("configs/project.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.input.dialog_column = None
    cfg.input.dialog_columns = None
    cfg.training.model_dir = str(tmp_path / "models")

    excel = tmp_path / "month.xlsx"
    pd.DataFrame(
        [
            {"created_at": "2025-02-01 10:00:00", "dialog_text": "CLIENT: не работает оплата", "subject": "s"},
            {"created_at": "2025-02-02 11:00:00", "dialog_text": "CLIENT: вопрос", "subject": "s"},
        ]
    ).to_excel(excel, index=False)

    objs = [_FakeVec(), _FakeComplaintModel(), _FakeCatModel(), _FakeEnc()]

    def _fake_load(_):
        return objs.pop(0)

    monkeypatch.setattr("complaints_trends.infer_month.joblib.load", _fake_load)

    out = infer_month(cfg, str(excel), "2025-02")
    assert len(out) == 2
    assert "raw_dialog" in out.columns
    assert "subcategory_pred" in out.columns
    assert out["category_pred"].eq("OTHER").all()
    assert out["subcategory_pred"].eq("UNKNOWN").all()
    assert Path("reports/month_2025-02_category_hist.png").exists()
    assert Path("reports/month_2025-02_subcategory_hist.png").exists()
    html = Path("reports/month_report_2025-02.html").read_text(encoding="utf-8")
    assert "Top subcategories" in html


def test_infer_month_parquet_sanitizes_mixed_object_columns(tmp_path, monkeypatch):
    cfg = load_config("configs/project.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.training.model_dir = str(tmp_path / "models")

    excel = tmp_path / "month_mixed.xlsx"
    pd.DataFrame(
        [
            {
                "created_at": "2025-02-01 10:00:00",
                "dialog_text": "CLIENT: не работает оплата",
                "колонка": pd.Timestamp("2025-02-01 10:00:00"),
            },
            {
                "created_at": "2025-02-02 11:00:00",
                "dialog_text": "CLIENT: вопрос",
                "колонка": "текст",
            },
        ]
    ).to_excel(excel, index=False)

    objs = [_FakeVec(), _FakeComplaintModel(), _FakeCatModel(), _FakeEnc()]

    def _fake_load(_):
        return objs.pop(0)

    monkeypatch.setattr("complaints_trends.infer_month.joblib.load", _fake_load)

    out = infer_month(cfg, str(excel), "2025-03")
    assert len(out) == 2

    parq = Path("data/interim/month_2025-03.parquet")
    assert parq.exists()
    loaded = pd.read_parquet(parq)
    assert loaded["колонка"].map(type).eq(str).all()
    parq.unlink(missing_ok=True)


def test_infer_month_does_not_convert_empty_dialogs_to_nan_string(tmp_path, monkeypatch):
    cfg = load_config("configs/project.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.training.model_dir = str(tmp_path / "models")

    excel = tmp_path / "month_nan.xlsx"
    pd.DataFrame(
        [
            {"created_at": "2025-02-01 10:00:00", "dialog_text": None},
            {"created_at": "2025-02-02 11:00:00", "dialog_text": np.nan},
        ]
    ).to_excel(excel, index=False)

    objs = [_FakeVec(), _FakeComplaintModel(), _FakeCatModel(), _FakeEnc()]

    def _fake_load(_):
        return objs.pop(0)

    monkeypatch.setattr("complaints_trends.infer_month.joblib.load", _fake_load)

    out = infer_month(cfg, str(excel), "2025-04")
    assert len(out) == 0


def test_infer_month_skips_rows_when_all_dialog_columns_empty(tmp_path, monkeypatch):
    cfg = load_config("configs/project.yaml").model_copy(deep=True)
    cfg.training.model_dir = str(tmp_path / "models")
    cfg.input.dialog_columns = ["dialog_text", "call_text", "comment_text", "summary_text"]

    excel = tmp_path / "month_skip_empty.xlsx"
    pd.DataFrame(
        [
            {
                "created_at": "2025-02-01 10:00:00",
                "dialog_text": "",
                "call_text": None,
                "comment_text": " ",
                "summary_text": None,
            },
            {
                "created_at": "2025-02-02 11:00:00",
                "dialog_text": "",
                "call_text": "CLIENT: не проходит платеж",
                "comment_text": "",
                "summary_text": "",
            },
        ]
    ).to_excel(excel, index=False)

    objs = [_FakeVec(), _FakeComplaintModel(), _FakeCatModel(), _FakeEnc()]

    def _fake_load(_):
        return objs.pop(0)

    monkeypatch.setattr("complaints_trends.infer_month.joblib.load", _fake_load)

    out = infer_month(cfg, str(excel), "2025-05")
    assert len(out) == 1



def test_enforce_subcategory_taxonomy_rejects_unknown_subcategories():
    cats = np.array(["TECHNICAL", "TECHNICAL", "OTHER"], dtype=object)
    subs = np.array(["login_issue", "inheritance transfer", "NOT_COMPLAINT"], dtype=object)
    taxonomy = {"subcategories_by_category": {"TECHNICAL": ["login_issue"]}}
    out = _enforce_subcategory_taxonomy(cats, subs, taxonomy)
    assert out.tolist() == ["login_issue", "UNKNOWN", "NOT_COMPLAINT"]


def test_select_infer_text_field_prefers_training_text_field():
    cfg = load_config("configs/project.yaml").model_copy(deep=True)
    cfg.training.text_field = "raw_dialog"
    df = pd.DataFrame(
        [
            {"raw_dialog": "D1", "client_first_message": "C1"},
            {"raw_dialog": "D2", "client_first_message": "C2"},
        ]
    )

    series, used = _select_infer_text_field(df, cfg)
    assert used == "raw_dialog"
    assert series.tolist() == ["D1", "D2"]


def test_select_infer_text_field_falls_back_to_client_message_when_column_missing():
    cfg = load_config("configs/project.yaml").model_copy(deep=True)
    cfg.training.text_field = "dialog_text"
    df = pd.DataFrame(
        [
            {"raw_dialog": "D1", "client_first_message": "C1"},
            {"raw_dialog": "D2", "client_first_message": "C2"},
        ]
    )

    series, used = _select_infer_text_field(df, cfg)
    assert used == "client_first_message (fallback from dialog_text)"
    assert series.tolist() == ["C1", "C2"]


class _OneRowModel:
    def __init__(self, val):
        self.val = val

    def predict(self, x):
        return np.array([self.val], dtype=int)


class _OneRowEnc:
    def __init__(self, val):
        self.val = val

    def inverse_transform(self, x):
        return np.array([self.val], dtype=object)


def test_predict_subcategories_by_category_prefers_category_specific_model():
    x = np.zeros((2, 1))
    idx = np.array([True, True])
    categories = np.array(["TECHNICAL", "PAYMENTS_TRANSFERS"], dtype=object)

    models_by_cat = {
        "TECHNICAL": {"mode": "constant", "value": "login_issue"},
        "PAYMENTS_TRANSFERS": {"mode": "constant", "value": "payment_error"},
    }

    out = _predict_subcategories_by_category(
        x=x,
        idx_mask=idx,
        categories=categories,
        fallback_subcat_model=None,
        fallback_subcat_enc=None,
        models_by_category=models_by_cat,
    )
    assert out.tolist() == ["login_issue", "payment_error"]
