from pathlib import Path

import numpy as np
import pandas as pd

from complaints_trends.config import load_config
from complaints_trends.infer_month import infer_month


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
    assert out["category_pred"].eq("OTHER").all()
