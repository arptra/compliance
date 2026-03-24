from __future__ import annotations

from datetime import datetime
from hashlib import sha1
from typing import Any

from .feedback_db import FeedbackDB
from .model_quality_service import compute_precision_at_k, compute_precision_by_bucket, compute_precision_by_group


class FeedbackService:
    def __init__(self, db: FeedbackDB) -> None:
        self.db = db

    @staticmethod
    def build_row_id(row: dict[str, Any]) -> str:
        if row.get("row_id"):
            return str(row["row_id"])
        seed = "|".join(
            [
                str(row.get("date") or ""),
                str(row.get("category") or ""),
                str(row.get("subcategory") or ""),
                str(row.get("row_dialog") or row.get("dialog_text") or ""),
            ]
        )
        return sha1(seed.encode("utf-8")).hexdigest()

    def upsert_feedback(self, item: dict[str, Any]) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        payload = {
            "row_id": item["row_id"],
            "pattern_tag": item.get("pattern_tag"),
            "review_date": item.get("review_date") or now[:10],
            "reviewer": item.get("reviewer"),
            "date_from": item.get("date_from"),
            "date_to": item.get("date_to"),
            "category": item.get("category"),
            "subcategory": item.get("subcategory"),
            "base_score": item.get("base_score"),
            "rerank_score": item.get("rerank_score"),
            "verdict": item["verdict"],
            "reason_code": item.get("reason_code"),
            "comment": item.get("comment"),
            "model_version": item.get("model_version"),
            "created_at": now,
            "updated_at": now,
        }
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO analyst_feedback (
                    row_id, pattern_tag, review_date, reviewer, date_from, date_to, category, subcategory,
                    base_score, rerank_score, verdict, reason_code, comment, model_version, created_at, updated_at
                ) VALUES (
                    :row_id, :pattern_tag, :review_date, :reviewer, :date_from, :date_to, :category, :subcategory,
                    :base_score, :rerank_score, :verdict, :reason_code, :comment, :model_version, :created_at, :updated_at
                )
                ON CONFLICT(row_id, pattern_tag)
                DO UPDATE SET
                    review_date=excluded.review_date,
                    reviewer=excluded.reviewer,
                    date_from=excluded.date_from,
                    date_to=excluded.date_to,
                    category=excluded.category,
                    subcategory=excluded.subcategory,
                    base_score=excluded.base_score,
                    rerank_score=excluded.rerank_score,
                    verdict=excluded.verdict,
                    reason_code=excluded.reason_code,
                    comment=excluded.comment,
                    model_version=excluded.model_version,
                    updated_at=excluded.updated_at
                """,
                payload,
            )
        return payload

    def bulk_upsert(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        for row in rows:
            self.upsert_feedback(row)
        return {"saved": len(rows)}

    def list_feedback(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        where = []
        args: list[Any] = []
        for key in ("pattern_tag", "reviewer", "verdict", "category"):
            if params.get(key):
                where.append(f"{key} = ?")
                args.append(params[key])
        if params.get("date_from"):
            where.append("review_date >= ?")
            args.append(params["date_from"])
        if params.get("date_to"):
            where.append("review_date <= ?")
            args.append(params["date_to"])
        query = "SELECT * FROM analyst_feedback"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY updated_at DESC"
        query += f" LIMIT {int(params.get('limit') or 1000)}"
        with self.db.connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [dict(r) for r in rows]

    def summary(self, params: dict[str, Any]) -> dict[str, Any]:
        rows = self.list_feedback({**params, "limit": params.get("limit") or 100000})
        reviewed = [r for r in rows if r["verdict"] in {"true", "false", "uncertain"}]
        true_count = sum(1 for r in reviewed if r["verdict"] == "true")
        false_count = sum(1 for r in reviewed if r["verdict"] == "false")
        uncertain_count = sum(1 for r in reviewed if r["verdict"] == "uncertain")
        reviewed_rows = len(reviewed)
        precision = true_count / reviewed_rows if reviewed_rows else None
        scoring_mode = str(params.get("scoring_mode") or "base")
        score_column = "base_score" if scoring_mode == "base" else "rerank_score"
        by_category = [
            {
                "category": item["name"],
                "reviewed": item["reviewed_count"],
                "true": item["true_count"],
                "false": item["false_count"],
                "precision": item["precision"],
            }
            for item in compute_precision_by_group(reviewed, key="category")
        ]
        by_cluster = [
            {
                "cluster": item["name"],
                "reviewed": item["reviewed_count"],
                "true": item["true_count"],
                "false": item["false_count"],
                "precision": item["precision"],
            }
            for item in compute_precision_by_group(reviewed, key="subcategory")
        ]
        by_score_bucket = compute_precision_by_bucket(reviewed, score_column=score_column)

        return {
            "reviewed_rows": reviewed_rows,
            "true_count": true_count,
            "false_count": false_count,
            "uncertain_count": uncertain_count,
            "precision_reviewed": precision,
            "precision_at_50": compute_precision_at_k(reviewed, 50, score_column=score_column),
            "precision_at_100": compute_precision_at_k(reviewed, 100, score_column=score_column),
            "by_category": by_category,
            "by_cluster": by_cluster,
            "by_score_bucket": by_score_bucket,
        }

    def delete_feedback(self, row_id: str, pattern_tag: str | None = None) -> dict[str, Any]:
        with self.db.connect() as conn:
            if pattern_tag:
                cur = conn.execute("DELETE FROM analyst_feedback WHERE row_id = ? AND pattern_tag = ?", (row_id, pattern_tag))
            else:
                cur = conn.execute("DELETE FROM analyst_feedback WHERE row_id = ?", (row_id,))
        return {"deleted": int(cur.rowcount)}

    def delete_feedback_for_scope(self, pattern_tag: str | None = None, reviewer: str | None = None) -> dict[str, Any]:
        where = []
        args: list[Any] = []
        if pattern_tag:
            where.append("pattern_tag = ?")
            args.append(pattern_tag)
        if reviewer:
            where.append("reviewer = ?")
            args.append(reviewer)
        query = "DELETE FROM analyst_feedback"
        if where:
            query += " WHERE " + " AND ".join(where)
        with self.db.connect() as conn:
            cur = conn.execute(query, args)
        return {"deleted": int(cur.rowcount)}
