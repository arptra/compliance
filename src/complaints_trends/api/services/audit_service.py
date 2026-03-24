from __future__ import annotations

import json
import random
import uuid
from datetime import datetime
from typing import Any

import pandas as pd

from .data_loader import DataLoader
from .feedback_db import FeedbackDB
from .feedback_service import FeedbackService


class AuditService:
    def __init__(self, db: FeedbackDB, loader: DataLoader, feedback_service: FeedbackService) -> None:
        self.db = db
        self.loader = loader
        self.feedback_service = feedback_service

    def _pool_row_ids(self, pattern_tag: str, date_from: str | None, date_to: str | None) -> tuple[list[str], int]:
        prepared = self.loader.load_prepare()
        if prepared.empty:
            return [], 0
        df = prepared.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "event_time" in df.columns:
            df["date"] = pd.to_datetime(df["event_time"], errors="coerce")

        if date_from and "date" in df.columns:
            df = df[df["date"] >= pd.to_datetime(date_from, errors="coerce")]
        if date_to and "date" in df.columns:
            df = df[df["date"] <= pd.to_datetime(date_to, errors="coerce")]

        all_row_ids = [self.feedback_service.build_row_id(r) for r in df.to_dict(orient="records")]

        scored = self.loader.load_pattern_monitor_scored(self.loader.resolve_tag("pattern_monitor", pattern_tag))
        flagged_ids: set[str] = set()
        if not scored.empty:
            alerts_mask = (scored.get("is_pattern_alert") == True) if "is_pattern_alert" in scored.columns else (scored.get("is_alert") == True) if "is_alert" in scored.columns else pd.Series([False] * len(scored))
            for r in scored[alerts_mask].to_dict(orient="records"):
                flagged_ids.add(self.feedback_service.build_row_id(r))

        pool = [row_id for row_id in all_row_ids if row_id not in flagged_ids]
        return pool, len(pool)

    def create_unflagged_audit_sample(self, params: dict[str, Any]) -> dict[str, Any]:
        pattern_tag = str(params.get("pattern_tag") or "latest")
        sample_size = max(1, int(params.get("sample_size") or 100))
        pool, pool_size = self._pool_row_ids(pattern_tag, params.get("date_from"), params.get("date_to"))
        seed = params.get("random_seed")
        rnd = random.Random(seed)
        actual_size = min(sample_size, len(pool))
        sample_ids = rnd.sample(pool, actual_size) if actual_size else []

        sample_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO unflagged_audit_samples
                (sample_id, created_at, pattern_tag, date_from, date_to, sample_size, source_pool_size, query_meta_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    now,
                    pattern_tag,
                    params.get("date_from"),
                    params.get("date_to"),
                    actual_size,
                    pool_size,
                    json.dumps({"random_seed": seed}, ensure_ascii=False),
                    "created",
                ),
            )
            conn.executemany(
                "INSERT INTO unflagged_audit_rows (sample_id, row_id) VALUES (?, ?)",
                [(sample_id, row_id) for row_id in sample_ids],
            )
        return self.get_unflagged_sample(sample_id)

    def list_unflagged_audit_samples(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM unflagged_audit_samples ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_unflagged_sample(self, sample_id: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            sample = conn.execute("SELECT * FROM unflagged_audit_samples WHERE sample_id = ?", (sample_id,)).fetchone()
            rows = conn.execute("SELECT * FROM unflagged_audit_rows WHERE sample_id = ? ORDER BY id", (sample_id,)).fetchall()
        if not sample:
            raise ValueError("sample not found")
        sample_dict = dict(sample)
        sample_dict["rows"] = [dict(r) for r in rows]
        sample_dict["estimate"] = self.estimate_hidden_positives(sample_dict)
        return sample_dict

    def review_unflagged_sample(self, sample_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        with self.db.connect() as conn:
            for row in rows:
                conn.execute(
                    """
                    UPDATE unflagged_audit_rows
                    SET review_verdict = ?, reviewer = ?, reviewed_at = ?, comment = ?
                    WHERE sample_id = ? AND row_id = ?
                    """,
                    (row.get("review_verdict"), row.get("reviewer"), now, row.get("comment"), sample_id, row.get("row_id")),
                )
            reviewed_count = conn.execute(
                "SELECT COUNT(*) FROM unflagged_audit_rows WHERE sample_id = ? AND review_verdict IS NOT NULL",
                (sample_id,),
            ).fetchone()[0]
            sample_size = conn.execute("SELECT sample_size FROM unflagged_audit_samples WHERE sample_id = ?", (sample_id,)).fetchone()[0]
            if reviewed_count >= sample_size:
                conn.execute("UPDATE unflagged_audit_samples SET status = 'reviewed' WHERE sample_id = ?", (sample_id,))
        return self.get_unflagged_sample(sample_id)

    def estimate_hidden_positives(self, sample: dict[str, Any]) -> dict[str, Any]:
        rows = sample.get("rows") or []
        reviewed = [r for r in rows if r.get("review_verdict") in {"true", "false", "uncertain"}]
        reviewed_in_sample = len(reviewed)
        true_in_sample = sum(1 for r in reviewed if r.get("review_verdict") == "true")
        p_hat = (true_in_sample / reviewed_in_sample) if reviewed_in_sample else None
        source_pool = int(sample.get("source_pool_size") or 0)
        estimated_hidden = (p_hat * source_pool) if p_hat is not None else None
        return {
            "reviewed_in_sample": reviewed_in_sample,
            "true_in_sample": true_in_sample,
            "estimated_hidden_positive_rate": p_hat,
            "estimated_hidden_positives_in_unflagged_pool": estimated_hidden,
            "note": "Оценка основана на случайной выборке и несет статистическую неопределенность.",
        }
