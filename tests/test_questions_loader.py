import json

import pytest

from complaints_trends.questions_loader import load_questions


def test_load_questions_valid_and_stable_codes(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "categories": [
                    {"question_ru": "Есть ли жалоба на переводы?"},
                    {"question_ru": "Есть ли жалоба на вход в приложение?"},
                ],
                },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out1 = load_questions(p)
    out2 = load_questions(p)
    assert out1["version"] == 1
    assert len(out1["items"]) == 2
    assert out1["items"][0]["code"] == out2["items"][0]["code"]
    assert out1["fallback_code"] == "OTHER"


def test_load_questions_deduplicates_same_questions(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "categories": [
                    {"question_ru": "Есть ли жалоба на переводы?"},
                    {"question_ru": "Есть ли жалоба на переводы?"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = load_questions(p)
    assert len(out["items"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 1, "categories": []},
        {"version": 1, "categories": [{"question_ru": ""}]},
    ],
)
def test_load_questions_raises_on_invalid_schema(tmp_path, payload):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(p)
