from __future__ import annotations

import json
from datetime import datetime, timezone

from complaints_trends.api.schemas import GigaChatBackgroundTaskSummary
from complaints_trends.api.services.gigachat_lab_service import GigaChatLabService


def test_compact_background_result_restores_source_row(tmp_path) -> None:
    service = GigaChatLabService.__new__(GigaChatLabService)
    service.background_dir = tmp_path
    task_id = "fast-task"
    task_dir = tmp_path / task_id
    task_dir.mkdir()

    summary = GigaChatBackgroundTaskSummary(
        task_id=task_id,
        status="completed",
        created_at=datetime.now(timezone.utc),
        total_rows=2,
        completed_rows=1,
        progress=0.5,
    )
    (task_dir / "task.json").write_text(summary.model_dump_json(), encoding="utf-8")
    (task_dir / "result.json").write_text(
        json.dumps(
            {
                "workbook": {
                    "upload_id": "background-fast-task",
                    "filename": "batch.csv",
                    "file_format": "csv",
                    "sheet_name": "data",
                    "total_rows": 2,
                    "rendered_rows": 2,
                    "columns": ["text"],
                    "rows": [{"text": "first"}, {"text": "second"}],
                },
                "row_runs": [{
                    "row_index": 99,
                    "_source_position": 1,
                    "result": {
                        "transport": "token",
                        "response_raw": "{}",
                        "response_json": {},
                        "parse_ok": True,
                    },
                    "reclassification_result": None,
                    "error": None,
                }],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    result = service.load_background_result(task_id)

    assert result.row_runs[0].row_index == 99
    assert result.row_runs[0].source_row == {"text": "second"}
