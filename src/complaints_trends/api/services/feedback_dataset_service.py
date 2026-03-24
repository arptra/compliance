from __future__ import annotations

import csv
import io
import json
from typing import Any

from .feedback_db import FeedbackDB
from .taxonomy_label_service import TaxonomyLabelService


ALLOWED_SORT = {
    "review_date",
    "row_id",
    "pattern_tag",
    "category",
    "subcategory",
    "base_score",
    "rerank_score",
    "verdict",
    "reason_code",
    "reviewer",
    "model_version",
    "updated_at",
}


class FeedbackDatasetService:
    def __init__(self, db: FeedbackDB, labels: TaxonomyLabelService | None = None) -> None:
        self.db = db
        self.labels = labels

    def _build_where(self, params: dict[str, Any]) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        args: list[Any] = []
        for key in ("pattern_tag", "reviewer", "verdict", "category", "subcategory", "reason_code", "model_version"):
            if params.get(key):
                where.append(f"{key} = ?")
                args.append(params[key])
        if params.get("date_from"):
            where.append("review_date >= ?")
            args.append(params["date_from"])
        if params.get("date_to"):
            where.append("review_date <= ?")
            args.append(params["date_to"])
        if params.get("q"):
            where.append("(row_id LIKE ? OR comment LIKE ? OR category_label_ru LIKE ? OR subcategory_label_ru LIKE ?)")
            q = f"%{params['q']}%"
            args.extend([q, q, q, q])
        return where, args

    def list_feedback_dataset(self, params: dict[str, Any]) -> dict[str, Any]:
        where, args = self._build_where(params)
        page = max(1, int(params.get("page") or 1))
        page_size = max(1, min(500, int(params.get("page_size") or 50)))
        offset = (page - 1) * page_size

        sort_by = str(params.get("sort_by") or "updated_at")
        if sort_by not in ALLOWED_SORT:
            sort_by = "updated_at"
        sort_order = "ASC" if str(params.get("sort_order") or "desc").lower() == "asc" else "DESC"

        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        query = f"SELECT * FROM analyst_feedback{where_sql} ORDER BY {sort_by} {sort_order} LIMIT ? OFFSET ?"

        with self.db.connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM analyst_feedback{where_sql}", args).fetchone()[0]
            rows = conn.execute(query, [*args, page_size, offset]).fetchall()

        items = [self.labels.enrich_row(dict(r)) if self.labels else dict(r) for r in rows]
        summary = self.get_feedback_dataset_summary(params)
        return {
            "items": items,
            "total": int(total),
            "page": page,
            "page_size": page_size,
            "summary": summary,
        }

    def get_feedback_dataset_summary(self, params: dict[str, Any]) -> dict[str, Any]:
        where, args = self._build_where(params)
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        with self.db.connect() as conn:
            rows = conn.execute(f"SELECT verdict, model_version FROM analyst_feedback{where_sql}", args).fetchall()
        reviewed = [dict(r) for r in rows]
        reviewed_rows = len(reviewed)
        true_count = sum(1 for r in reviewed if r.get("verdict") == "true")
        false_count = sum(1 for r in reviewed if r.get("verdict") == "false")
        uncertain_count = sum(1 for r in reviewed if r.get("verdict") == "uncertain")
        active_model_version = next((r.get("model_version") for r in reviewed if r.get("model_version")), None)
        return {
            "reviewed_rows": reviewed_rows,
            "true_count": true_count,
            "false_count": false_count,
            "uncertain_count": uncertain_count,
            "precision_reviewed": (true_count / reviewed_rows) if reviewed_rows else None,
            "active_model_version": active_model_version,
        }

    def get_feedback_item(self, row_id: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM analyst_feedback WHERE row_id = ? ORDER BY updated_at DESC LIMIT 1", (row_id,)).fetchone()
        return self.labels.enrich_row(dict(row)) if (row and self.labels) else (dict(row) if row else None)

    def export_feedback_dataset(self, params: dict[str, Any], output_format: str) -> str:
        data = self.list_feedback_dataset({**params, "page": 1, "page_size": 100000})["items"]
        if params.get("positives_only"):
            data = [r for r in data if r.get("verdict") == "true"]
        if output_format == "json":
            return json.dumps(data, ensure_ascii=False)

        fieldnames = [
            "review_date",
            "row_id",
            "pattern_tag",
            "category",
            "category_label_ru",
            "subcategory",
            "subcategory_label_ru",
            "base_score",
            "rerank_score",
            "verdict",
            "reason_code",
            "reviewer",
            "model_version",
            "comment",
            "updated_at",
        ]
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow({k: row.get(k) for k in fieldnames})
        return stream.getvalue()
