from __future__ import annotations

from pathlib import Path

import pandas as pd

from complaints_trends.prepare_dataset import _sanitize_excel_cell, _to_excel_safe


def test_sanitize_excel_cell_removes_illegal_chars_and_truncates():
    value = "ok" + "\x00" + "bad" + ("x" * 40000)
    out = _sanitize_excel_cell(value)
    assert "\x00" not in out
    assert len(out) == 32767


def test_to_excel_safe_handles_long_and_illegal_content(tmp_path: Path):
    df = pd.DataFrame(
        {
            "a": ["start\x01end", "y" * 50000],
            "b": ["normal", None],
        }
    )
    out = tmp_path / "safe.xlsx"
    _to_excel_safe(df, out, index=False)
    assert out.exists()

    loaded = pd.read_excel(out)
    assert "\x01" not in loaded.loc[0, "a"]
    assert len(loaded.loc[1, "a"]) == 32767
