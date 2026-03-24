from __future__ import annotations

import sqlite3
from pathlib import Path


class FeedbackDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyst_feedback (
                    id INTEGER PRIMARY KEY,
                    row_id TEXT NOT NULL,
                    pattern_tag TEXT NULL,
                    review_date TEXT NOT NULL,
                    reviewer TEXT NULL,
                    date_from TEXT NULL,
                    date_to TEXT NULL,
                    category TEXT NULL,
                    subcategory TEXT NULL,
                    base_score REAL NULL,
                    rerank_score REAL NULL,
                    verdict TEXT NOT NULL CHECK (verdict IN ('true','false','uncertain')),
                    reason_code TEXT NULL,
                    comment TEXT NULL,
                    model_version TEXT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(row_id, pattern_tag)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reranker_model_versions (
                    version_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('training','ready','active','failed','archived')),
                    algorithm TEXT NOT NULL,
                    metrics_json TEXT NULL,
                    artifact_path TEXT NULL,
                    train_rows INTEGER NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    reviewer TEXT NULL,
                    date_from TEXT NULL,
                    date_to TEXT NULL,
                    pattern_tag TEXT NULL,
                    rows_reviewed INTEGER,
                    true_count INTEGER,
                    false_count INTEGER,
                    uncertain_count INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS unflagged_audit_samples (
                    sample_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    pattern_tag TEXT NULL,
                    date_from TEXT NULL,
                    date_to TEXT NULL,
                    sample_size INTEGER NOT NULL,
                    source_pool_size INTEGER NOT NULL,
                    query_meta_json TEXT NULL,
                    status TEXT NOT NULL CHECK (status IN ('created','reviewed','archived'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS unflagged_audit_rows (
                    id INTEGER PRIMARY KEY,
                    sample_id TEXT NOT NULL,
                    row_id TEXT NOT NULL,
                    review_verdict TEXT NULL CHECK (review_verdict IN ('true','false','uncertain')),
                    reviewer TEXT NULL,
                    reviewed_at TEXT NULL,
                    comment TEXT NULL,
                    FOREIGN KEY(sample_id) REFERENCES unflagged_audit_samples(sample_id)
                )
                """
            )
