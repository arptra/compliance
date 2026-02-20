from complaints_trends.config import InputConfig


def base():
    return {
        "input_dir": "data/raw",
        "file_glob": "*.xlsx",
        "month_source": "filename",
        "month_regex": r"(\\d{4})-(\\d{2})",
        "month_column": None,
        "id_column": None,
        "signal_columns": ["dialog_text"],
        "dialog_column": "dialog_text",
        "encoding": "utf-8",
    }


def test_month_source_accepts_file_like_value():
    d = base()
    d["month_source"] = "filename.xlsx"
    cfg = InputConfig.model_validate(d)
    assert cfg.month_source == "filename"


def test_month_source_column_name_autofix():
    d = base()
    d["month_source"] = "created_at"
    cfg = InputConfig.model_validate(d)
    assert cfg.month_source == "column"
    assert cfg.month_column == "created_at"
