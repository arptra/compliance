from pathlib import Path

import pandas as pd

from complaints_trends.prepare_dataset import _build_subcategory_examples, _build_subcategory_histogram


def test_build_subcategory_examples_groups_and_truncates():
    df = pd.DataFrame(
        [
            {"complaint_subcategory_llm": "payment_error", "short_summary_llm": "Очень длинный текст " + "x" * 300},
            {"complaint_subcategory_llm": "payment_error", "short_summary_llm": "Повторный кейс"},
            {"complaint_subcategory_llm": "login_issue", "short_summary_llm": "Не могу войти"},
            {"complaint_subcategory_llm": "", "short_summary_llm": "Пустая подкатегория"},
        ]
    )

    out = _build_subcategory_examples(df, max_examples_per_subcategory=2, example_char_limit=40)

    assert list(out.columns) == ["subcategory", "gigachat_examples"]
    assert out.iloc[0]["subcategory"] == "payment_error"
    assert "…" in out.iloc[0]["gigachat_examples"]
    assert "- Повторный кейс" in out.iloc[0]["gigachat_examples"]
    assert "login_issue" in out["subcategory"].tolist()


def test_build_subcategory_histogram_creates_file(tmp_path):
    complaints = pd.DataFrame(
        [
            {"complaint_subcategory_llm": "payment_error"},
            {"complaint_subcategory_llm": "payment_error"},
            {"complaint_subcategory_llm": "login_issue"},
        ]
    )

    out_file = tmp_path / "subcategories.png"
    saved = _build_subcategory_histogram(complaints, out_file)

    assert saved == out_file
    assert Path(saved).exists()
    assert Path(saved).stat().st_size > 0
