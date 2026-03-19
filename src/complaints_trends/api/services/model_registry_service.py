from __future__ import annotations

import json
from typing import Any

from .feedback_db import FeedbackDB


class ModelRegistryService:
    def __init__(self, db: FeedbackDB) -> None:
        self.db = db

    def create_version(self, payload: dict[str, Any]) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO reranker_model_versions
                (version_id, created_at, status, algorithm, metrics_json, artifact_path, train_rows, active, notes)
                VALUES (:version_id, :created_at, :status, :algorithm, :metrics_json, :artifact_path, :train_rows, :active, :notes)
                """,
                {
                    **payload,
                    "metrics_json": json.dumps(payload.get("metrics_json")) if payload.get("metrics_json") is not None else None,
                },
            )

    def list_versions(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM reranker_model_versions ORDER BY created_at DESC").fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metrics_json"] = json.loads(item["metrics_json"]) if item.get("metrics_json") else None
            out.append(item)
        return out

    def get_active(self) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM reranker_model_versions WHERE active=1 LIMIT 1").fetchone()
        if not row:
            return None
        item = dict(row)
        item["metrics_json"] = json.loads(item["metrics_json"]) if item.get("metrics_json") else None
        return item

    def set_active(self, version_id: str, active: bool = True) -> None:
        with self.db.connect() as conn:
            if active:
                conn.execute("UPDATE reranker_model_versions SET active=0, status=CASE WHEN status='active' THEN 'ready' ELSE status END")
                conn.execute("UPDATE reranker_model_versions SET active=1, status='active' WHERE version_id=?", (version_id,))
            else:
                conn.execute("UPDATE reranker_model_versions SET active=0, status='ready' WHERE version_id=?", (version_id,))
