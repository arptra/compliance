from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


class CatalogService:
    """Small SQLite catalog for auth, file metadata, artifacts, jobs, and dedup."""

    def __init__(self, db_path: Path, secret_path: Path | None = None):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_path = secret_path or (self.db_path.parent / "auth.secret")
        self._secret = self._load_or_create_secret()
        self._init_db()
        self.ensure_default_workspace()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _load_or_create_secret(self) -> bytes:
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        if self.secret_path.exists():
            return self.secret_path.read_bytes()
        secret = secrets.token_bytes(32)
        self.secret_path.write_bytes(secret)
        return secret

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id TEXT PRIMARY KEY,
                  email TEXT NOT NULL UNIQUE,
                  password_hash TEXT NOT NULL,
                  display_name TEXT NOT NULL,
                  role TEXT NOT NULL DEFAULT 'user',
                  is_active INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL,
                  last_login_at TEXT
                );

                CREATE TABLE IF NOT EXISTS workspaces (
                  id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workspace_members (
                  workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                  role TEXT NOT NULL DEFAULT 'user',
                  created_at TEXT NOT NULL,
                  PRIMARY KEY (workspace_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                  id TEXT PRIMARY KEY,
                  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                  token_hash TEXT NOT NULL UNIQUE,
                  expires_at TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  revoked_at TEXT
                );

                CREATE TABLE IF NOT EXISTS files (
                  id TEXT PRIMARY KEY,
                  workspace_id TEXT NOT NULL,
                  uploaded_by_user_id TEXT NOT NULL,
                  original_filename TEXT NOT NULL,
                  display_filename TEXT NOT NULL,
                  storage_path TEXT NOT NULL,
                  file_format TEXT NOT NULL,
                  sheet_count INTEGER NOT NULL DEFAULT 0,
                  row_count INTEGER NOT NULL DEFAULT 0,
                  visibility TEXT NOT NULL DEFAULT 'workspace',
                  status TEXT NOT NULL DEFAULT 'uploaded',
                  created_at TEXT NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                  id TEXT PRIMARY KEY,
                  file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                  workspace_id TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  artifact_type TEXT NOT NULL,
                  storage_path TEXT NOT NULL,
                  rows_count INTEGER NOT NULL DEFAULT 0,
                  columns_json TEXT NOT NULL DEFAULT '[]',
                  metadata_json TEXT NOT NULL DEFAULT '{}',
                  source_artifact_id TEXT,
                  created_by_user_id TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_file_stage ON artifacts(file_id, stage);
                CREATE INDEX IF NOT EXISTS idx_artifacts_workspace_stage ON artifacts(workspace_id, stage);

                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY,
                  workspace_id TEXT NOT NULL,
                  file_id TEXT,
                  input_artifact_id TEXT,
                  output_artifact_id TEXT,
                  type TEXT NOT NULL,
                  status TEXT NOT NULL,
                  progress_current INTEGER NOT NULL DEFAULT 0,
                  progress_total INTEGER NOT NULL DEFAULT 0,
                  error TEXT,
                  created_by_user_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS record_index (
                  workspace_id TEXT NOT NULL,
                  row_hash TEXT NOT NULL,
                  record_id TEXT NOT NULL,
                  first_file_id TEXT NOT NULL,
                  first_artifact_id TEXT NOT NULL,
                  first_seen_at TEXT NOT NULL,
                  PRIMARY KEY (workspace_id, row_hash)
                );

                CREATE TABLE IF NOT EXISTS lab_settings_versions (
                  version_id TEXT PRIMARY KEY,
                  workspace_id TEXT NOT NULL,
                  owner_user_id TEXT NOT NULL,
                  visibility TEXT NOT NULL DEFAULT 'private',
                  title TEXT NOT NULL,
                  description TEXT NOT NULL DEFAULT '',
                  status TEXT NOT NULL DEFAULT 'draft',
                  created_by TEXT NOT NULL DEFAULT '',
                  updated_by TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  updated_at TEXT,
                  base_version_id TEXT,
                  values_json TEXT NOT NULL DEFAULT '{}',
                  metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_lab_settings_versions_workspace_visibility
                  ON lab_settings_versions(workspace_id, visibility);
                CREATE INDEX IF NOT EXISTS idx_lab_settings_versions_owner
                  ON lab_settings_versions(owner_user_id);
                """
            )
            user_columns = {str(row["name"]) for row in con.execute("PRAGMA table_info(users)").fetchall()}
            if "first_name" not in user_columns:
                con.execute("ALTER TABLE users ADD COLUMN first_name TEXT NOT NULL DEFAULT ''")
            if "last_name" not in user_columns:
                con.execute("ALTER TABLE users ADD COLUMN last_name TEXT NOT NULL DEFAULT ''")

    def ensure_default_workspace(self) -> str:
        workspace_id = "default"
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO workspaces (id, name, created_at) VALUES (?, ?, ?)",
                (workspace_id, "Default workspace", iso_now()),
            )
        return workspace_id

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_bytes(16)
        iterations = 260_000
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return f"pbkdf2_sha256${iterations}${_b64url_encode(salt)}${_b64url_encode(digest)}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        try:
            algo, iterations_text, salt_text, digest_text = password_hash.split("$", 3)
            if algo != "pbkdf2_sha256":
                return False
            iterations = int(iterations_text)
            salt = _b64url_decode(salt_text)
            expected = _b64url_decode(digest_text)
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
            return hmac.compare_digest(actual, expected)
        except Exception:
            return False

    def create_user(self, email: str, password: str, display_name: str, role: str | None = None) -> dict[str, Any]:
        normalized_email = email.strip().lower()
        if not normalized_email:
            raise ValueError("Email is required.")
        if len(password) < 6:
            raise ValueError("Password must contain at least 6 characters.")
        clean_display_name = display_name.strip() or normalized_email
        name_parts = clean_display_name.split(maxsplit=1)
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        workspace_id = self.ensure_default_workspace()
        with self._connect() as con:
            existing_count = int(con.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            user_role = role or ("admin" if existing_count == 0 else "user")
            user_id = str(uuid.uuid4())
            now = iso_now()
            con.execute(
                """
                INSERT INTO users (id, email, password_hash, display_name, first_name, last_name, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (user_id, normalized_email, self._hash_password(password), clean_display_name, first_name, last_name, user_role, now),
            )
            con.execute(
                """
                INSERT OR IGNORE INTO workspace_members (workspace_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (workspace_id, user_id, user_role, now),
            )
        return self.get_user(user_id) or {}

    def authenticate_user(self, email: str, password: str) -> dict[str, Any] | None:
        normalized_email = email.strip().lower()
        with self._connect() as con:
            row = con.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (normalized_email,)).fetchone()
            if not row or not self._verify_password(password, str(row["password_hash"])):
                return None
            con.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (iso_now(), row["id"]))
        return self.get_user(str(row["id"]))

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT u.id, u.email, u.display_name, u.first_name, u.last_name, u.role, u.is_active, u.created_at, u.last_login_at,
                       COALESCE(wm.workspace_id, 'default') AS workspace_id,
                       COALESCE(wm.role, u.role) AS workspace_role
                FROM users u
                LEFT JOIN workspace_members wm ON wm.user_id = u.id
                WHERE u.id = ?
                ORDER BY wm.created_at ASC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def users_by_ids(self, user_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = sorted({str(user_id).strip() for user_id in user_ids if str(user_id).strip()})
        if not ids:
            return {}
        found: dict[str, dict[str, Any]] = {}
        with self._connect() as con:
            for idx in range(0, len(ids), 800):
                chunk = ids[idx:idx + 800]
                placeholders = ",".join("?" for _ in chunk)
                rows = con.execute(
                    f"""
                    SELECT u.id, u.email, u.display_name, u.role, u.is_active, u.created_at, u.last_login_at,
                           u.first_name, u.last_name,
                           COALESCE(wm.workspace_id, 'default') AS workspace_id,
                           COALESCE(wm.role, u.role) AS workspace_role
                    FROM users u
                    LEFT JOIN workspace_members wm ON wm.user_id = u.id
                    WHERE u.id IN ({placeholders})
                    """,
                    chunk,
                ).fetchall()
                for row in rows:
                    data = dict(row)
                    found[str(data["id"])] = data
        return found

    def update_profile(self, user_id: str, *, first_name: str, last_name: str, display_name: str | None = None) -> dict[str, Any]:
        clean_first_name = first_name.strip()
        clean_last_name = last_name.strip()
        clean_display_name = (display_name or "").strip() or " ".join(part for part in [clean_first_name, clean_last_name] if part).strip()
        if not clean_display_name:
            user = self.get_user(user_id)
            clean_display_name = str(user.get("email") or user_id) if user else user_id
        with self._connect() as con:
            con.execute(
                "UPDATE users SET first_name = ?, last_name = ?, display_name = ? WHERE id = ?",
                (clean_first_name, clean_last_name, clean_display_name, user_id),
            )
        user = self.get_user(user_id)
        if not user:
            raise ValueError("User not found.")
        return user

    def change_password(self, user_id: str, current_password: str, new_password: str) -> None:
        if len(new_password) < 6:
            raise ValueError("New password must contain at least 6 characters.")
        with self._connect() as con:
            row = con.execute("SELECT password_hash FROM users WHERE id = ? AND is_active = 1", (user_id,)).fetchone()
            if not row or not self._verify_password(current_password, str(row["password_hash"])):
                raise ValueError("Current password is incorrect.")
            con.execute("UPDATE users SET password_hash = ? WHERE id = ?", (self._hash_password(new_password), user_id))

    def lab_settings_version_exists(self, version_id: str) -> bool:
        with self._connect() as con:
            row = con.execute("SELECT 1 FROM lab_settings_versions WHERE version_id = ?", (version_id,)).fetchone()
        return row is not None

    def save_lab_settings_version(self, payload: dict[str, Any]) -> dict[str, Any]:
        version_id = str(payload.get("version_id") or "").strip()
        if not version_id:
            raise ValueError("version_id is required.")
        now = iso_now()
        created_at = str(payload.get("created_at") or now)
        updated_at = payload.get("updated_at") or now
        visibility = str(payload.get("visibility") or "private")
        if visibility not in {"public", "private"}:
            visibility = "private"
        values = payload.get("values") if isinstance(payload.get("values"), dict) else {}
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO lab_settings_versions (
                  version_id, workspace_id, owner_user_id, visibility, title, description, status,
                  created_by, updated_by, created_at, updated_at, base_version_id, values_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version_id) DO UPDATE SET
                  workspace_id = excluded.workspace_id,
                  owner_user_id = excluded.owner_user_id,
                  visibility = excluded.visibility,
                  title = excluded.title,
                  description = excluded.description,
                  status = excluded.status,
                  created_by = excluded.created_by,
                  updated_by = excluded.updated_by,
                  updated_at = excluded.updated_at,
                  base_version_id = excluded.base_version_id,
                  values_json = excluded.values_json,
                  metadata_json = excluded.metadata_json
                """,
                (
                    version_id,
                    str(payload.get("workspace_id") or "default"),
                    str(payload.get("owner_user_id") or ""),
                    visibility,
                    str(payload.get("title") or version_id),
                    str(payload.get("description") or ""),
                    str(payload.get("status") or "draft"),
                    str(payload.get("created_by") or ""),
                    str(payload.get("updated_by") or ""),
                    created_at,
                    str(updated_at) if updated_at else None,
                    payload.get("base_version_id"),
                    _json_dumps(values),
                    _json_dumps(metadata),
                ),
            )
        return self.get_lab_settings_version(version_id, user_id=str(payload.get("owner_user_id") or ""), workspace_id=str(payload.get("workspace_id") or "default"), include_private=True) or {}

    def list_lab_settings_versions(self, *, user_id: str, workspace_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT * FROM lab_settings_versions
                WHERE workspace_id = ?
                  AND (visibility = 'public' OR owner_user_id = ?)
                ORDER BY CASE WHEN version_id = 'default' THEN 0 ELSE 1 END, lower(title), created_at
                """,
                (workspace_id, user_id),
            ).fetchall()
        return [self._lab_settings_version_from_row(row, user_id=user_id) for row in rows]

    def get_lab_settings_version(
        self,
        version_id: str,
        *,
        user_id: str,
        workspace_id: str,
        include_private: bool = False,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM lab_settings_versions WHERE version_id = ? AND workspace_id = ?"
        params: list[Any] = [version_id, workspace_id]
        if not include_private:
            query += " AND (visibility = 'public' OR owner_user_id = ?)"
            params.append(user_id)
        with self._connect() as con:
            row = con.execute(query, params).fetchone()
        return self._lab_settings_version_from_row(row, user_id=user_id) if row else None

    @staticmethod
    def _lab_settings_version_from_row(row: sqlite3.Row, *, user_id: str) -> dict[str, Any]:
        data = dict(row)
        data["values"] = json.loads(data.pop("values_json") or "{}")
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        data["is_default"] = data.get("version_id") == "default"
        data["can_edit"] = bool(data.get("owner_user_id")) and str(data.get("owner_user_id")) == str(user_id) and not data["is_default"]
        data["path"] = "sqlite:data/app.sqlite"
        return data

    def _sign_token_payload(self, payload: dict[str, Any]) -> str:
        body = _b64url_encode(_json_dumps(payload).encode("utf-8"))
        signature = hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest()
        return f"{body}.{_b64url_encode(signature)}"

    def _decode_token_payload(self, token: str) -> dict[str, Any]:
        body, signature_text = token.split(".", 1)
        expected = hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_decode(signature_text), expected):
            raise ValueError("Invalid token signature.")
        payload = json.loads(_b64url_decode(body).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(utc_now().timestamp()):
            raise ValueError("Token expired.")
        return payload

    def issue_access_token(self, user_id: str, expires_in_seconds: int = 7 * 24 * 3600) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        expires_at = utc_now() + timedelta(seconds=expires_in_seconds)
        payload = {
            "sub": user_id,
            "sid": session_id,
            "exp": int(expires_at.timestamp()),
            "iat": int(utc_now().timestamp()),
        }
        token = self._sign_token_payload(payload)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO sessions (id, user_id, token_hash, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, token_hash, expires_at.isoformat(), iso_now()),
            )
        return {"access_token": token, "token_type": "bearer", "expires_at": expires_at.isoformat()}

    def user_from_authorization(self, authorization: str | None) -> dict[str, Any] | None:
        if not authorization:
            return None
        parts = authorization.strip().split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        token = parts[1].strip()
        try:
            payload = self._decode_token_payload(token)
        except Exception:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connect() as con:
            session = con.execute(
                "SELECT * FROM sessions WHERE id = ? AND token_hash = ? AND revoked_at IS NULL",
                (payload.get("sid"), token_hash),
            ).fetchone()
        if not session:
            return None
        return self.get_user(str(payload.get("sub")))

    def revoke_authorization(self, authorization: str | None) -> None:
        if not authorization:
            return
        parts = authorization.strip().split(" ", 1)
        if len(parts) != 2:
            return
        token_hash = hashlib.sha256(parts[1].strip().encode("utf-8")).hexdigest()
        with self._connect() as con:
            con.execute("UPDATE sessions SET revoked_at = ? WHERE token_hash = ?", (iso_now(), token_hash))

    def get_or_create_dev_user(self) -> dict[str, Any]:
        """Fallback for legacy endpoints/tests that do not send Authorization."""
        with self._connect() as con:
            row = con.execute("SELECT id FROM users WHERE email = ?", ("dev@local",)).fetchone()
        if row:
            return self.get_user(str(row["id"])) or {}
        return self.create_user("dev@local", "dev-password", "Local Dev", role="admin")

    def create_file(
        self,
        *,
        file_id: str,
        workspace_id: str,
        uploaded_by_user_id: str,
        original_filename: str,
        display_filename: str,
        storage_path: str,
        file_format: str,
        sheet_count: int,
        metadata: dict[str, Any],
    ) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO files (
                  id, workspace_id, uploaded_by_user_id, original_filename, display_filename,
                  storage_path, file_format, sheet_count, row_count, visibility, status, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT row_count FROM files WHERE id = ?), 0), 'workspace', 'uploaded', ?, ?)
                """,
                (
                    file_id,
                    workspace_id,
                    uploaded_by_user_id,
                    original_filename,
                    display_filename,
                    storage_path,
                    file_format,
                    int(sheet_count),
                    file_id,
                    iso_now(),
                    _json_dumps(metadata),
                ),
            )

    def update_file_status(self, file_id: str, status: str, row_count: int | None = None) -> None:
        with self._connect() as con:
            if row_count is None:
                con.execute("UPDATE files SET status = ? WHERE id = ?", (status, file_id))
            else:
                con.execute("UPDATE files SET status = ?, row_count = ? WHERE id = ?", (status, int(row_count), file_id))

    def delete_file(self, file_id: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def create_artifact(
        self,
        *,
        artifact_id: str | None = None,
        file_id: str,
        workspace_id: str,
        stage: str,
        artifact_type: str,
        storage_path: str,
        rows_count: int,
        columns: list[str],
        metadata: dict[str, Any],
        created_by_user_id: str,
        source_artifact_id: str | None = None,
    ) -> str:
        artifact_id = artifact_id or str(uuid.uuid4())
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO artifacts (
                  id, file_id, workspace_id, stage, artifact_type, storage_path, rows_count,
                  columns_json, metadata_json, source_artifact_id, created_by_user_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    file_id,
                    workspace_id,
                    stage,
                    artifact_type,
                    storage_path,
                    int(rows_count),
                    _json_dumps(columns),
                    _json_dumps(metadata),
                    source_artifact_id,
                    created_by_user_id,
                    iso_now(),
                ),
            )
        return artifact_id

    def list_artifacts(self, file_id: str, stage: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM artifacts WHERE file_id = ?"
        params: list[Any] = [file_id]
        if stage:
            query += " AND stage = ?"
            params.append(stage)
        query += " ORDER BY created_at ASC"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def get_raw_artifact_for_sheet(self, file_id: str, sheet_name: str) -> dict[str, Any] | None:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM artifacts WHERE file_id = ? AND stage = 'raw' ORDER BY created_at ASC",
                (file_id,),
            ).fetchall()
        for row in rows:
            artifact = self._artifact_from_row(row)
            if str(artifact["metadata"].get("sheet_name", "")) == sheet_name:
                return artifact
        return None

    def update_artifact_rows_count(self, artifact_id: str, rows_count: int) -> None:
        with self._connect() as con:
            con.execute("UPDATE artifacts SET rows_count = ? WHERE id = ?", (int(rows_count), artifact_id))

    def delete_artifacts(self, artifact_ids: Iterable[str]) -> None:
        ids = sorted({str(artifact_id).strip() for artifact_id in artifact_ids if str(artifact_id).strip()})
        if not ids:
            return
        with self._connect() as con:
            for idx in range(0, len(ids), 800):
                chunk = ids[idx:idx + 800]
                placeholders = ",".join("?" for _ in chunk)
                con.execute(f"DELETE FROM artifacts WHERE id IN ({placeholders})", chunk)

    def delete_record_index_by_record_ids(self, workspace_id: str, record_ids: Iterable[str]) -> None:
        ids = sorted({str(record_id).strip() for record_id in record_ids if str(record_id).strip()})
        if not ids:
            return
        with self._connect() as con:
            for idx in range(0, len(ids), 800):
                chunk = ids[idx:idx + 800]
                placeholders = ",".join("?" for _ in chunk)
                con.execute(
                    f"DELETE FROM record_index WHERE workspace_id = ? AND record_id IN ({placeholders})",
                    [workspace_id, *chunk],
                )

    def delete_record_index_by_artifact_ids(self, workspace_id: str, artifact_ids: Iterable[str]) -> None:
        ids = sorted({str(artifact_id).strip() for artifact_id in artifact_ids if str(artifact_id).strip()})
        if not ids:
            return
        with self._connect() as con:
            for idx in range(0, len(ids), 800):
                chunk = ids[idx:idx + 800]
                placeholders = ",".join("?" for _ in chunk)
                con.execute(
                    f"DELETE FROM record_index WHERE workspace_id = ? AND first_artifact_id IN ({placeholders})",
                    [workspace_id, *chunk],
                )

    def _artifact_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["columns"] = json.loads(data.pop("columns_json") or "[]")
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        return data

    def existing_hashes(self, workspace_id: str, row_hashes: list[str]) -> set[str]:
        if not row_hashes:
            return set()
        found: set[str] = set()
        with self._connect() as con:
            for idx in range(0, len(row_hashes), 800):
                chunk = row_hashes[idx:idx + 800]
                placeholders = ",".join("?" for _ in chunk)
                rows = con.execute(
                    f"SELECT row_hash FROM record_index WHERE workspace_id = ? AND row_hash IN ({placeholders})",
                    [workspace_id, *chunk],
                ).fetchall()
                found.update(str(row["row_hash"]) for row in rows)
        return found

    def insert_record_hashes(
        self,
        *,
        workspace_id: str,
        records: list[tuple[str, str]],
        file_id: str,
        artifact_id: str,
    ) -> None:
        if not records:
            return
        now = iso_now()
        with self._connect() as con:
            con.executemany(
                """
                INSERT OR IGNORE INTO record_index (
                  workspace_id, row_hash, record_id, first_file_id, first_artifact_id, first_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(workspace_id, row_hash, record_id, file_id, artifact_id, now) for row_hash, record_id in records],
            )
