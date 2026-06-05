from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile
from io import BytesIO

import pandas as pd

from .catalog_service import CatalogService


SYSTEM_COLUMNS = {
    "__record_id",
    "__workspace_id",
    "__file_id",
    "__artifact_id",
    "__uploaded_by_user_id",
    "__source_filename",
    "__sheet_name",
    "__source_row_number",
    "__row_hash",
    "__ingested_at",
    "__stage",
    "__period_year",
    "__period_month",
}

RECORD_METADATA_COLUMNS = [
    "__record_id",
    "__workspace_id",
    "__file_id",
    "__artifact_id",
    "__uploaded_by_user_id",
    "__source_filename",
    "__sheet_name",
    "__source_row_number",
    "__ingested_at",
    "__stage",
    "__period_year",
    "__period_month",
]


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("._")[:80] or "data"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_cell(value: Any) -> Any:
    if _is_missing_cell(value):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, str):
        return value.strip()
    return _json_safe_cell(value)


def _is_missing_cell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _json_safe_cell(value: Any) -> Any:
    if _is_missing_cell(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe_cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_cell(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            item = value.item()
        except (TypeError, ValueError):
            item = value
        if item is not value:
            return _json_safe_cell(item)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _canonical_row(row: dict[str, Any]) -> str:
    payload = {str(key): _canonical_cell(value) for key, value in sorted(row.items(), key=lambda item: str(item[0]))}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _row_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_row(row).encode("utf-8")).hexdigest()


class ParquetLakeService:
    def __init__(self, base_dir: Path, catalog: CatalogService):
        self.base_dir = base_dir
        self.catalog = catalog
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def ingest_workbook(
        self,
        *,
        file_id: str,
        workspace_id: str,
        user_id: str,
        stored_path: Path,
        display_filename: str,
        file_format: str,
        sheets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        total_new_rows = 0
        artifacts: list[dict[str, Any]] = []
        for sheet in sheets:
            sheet_name = str(sheet.get("name") or "data")
            df = self._read_sheet(stored_path, file_format, sheet_name)
            df = self._drop_empty_unnamed_columns(df)
            artifact = self.write_stage_frame(
                stage="raw",
                file_id=file_id,
                workspace_id=workspace_id,
                user_id=user_id,
                source_filename=display_filename,
                sheet_name=sheet_name,
                df=df,
                metadata={"file_format": file_format, "sheet_name": sheet_name},
            )
            total_new_rows += int(artifact["rows_count"])
            artifacts.append(artifact)
        self.catalog.update_file_status(file_id, "ready", total_new_rows)
        return {"rows_count": total_new_rows, "artifacts": artifacts}

    def write_stage_frame(
        self,
        *,
        stage: str,
        file_id: str,
        workspace_id: str,
        user_id: str,
        source_filename: str,
        sheet_name: str,
        df: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
        source_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        clean_df = df.copy()
        clean_df.columns = [str(column) for column in clean_df.columns]
        source_columns = [column for column in clean_df.columns if column not in SYSTEM_COLUMNS]
        row_hashes = [_row_hash(row) for row in clean_df[source_columns].to_dict(orient="records")]
        existing = self.catalog.existing_hashes(workspace_id, row_hashes) if stage == "raw" else set()
        seen_new: set[str] = set()
        keep_indexes: list[int] = []
        for idx, value in enumerate(row_hashes):
            if value in existing or value in seen_new:
                continue
            keep_indexes.append(idx)
            seen_new.add(value)
        duplicate_rows = len(row_hashes) - len(keep_indexes)
        if keep_indexes:
            out = clean_df.iloc[keep_indexes].reset_index(drop=True)
            kept_hashes = [row_hashes[idx] for idx in keep_indexes]
            source_row_numbers = [int(idx) + 1 for idx in keep_indexes]
        else:
            out = clean_df.iloc[0:0].copy()
            kept_hashes = []
            source_row_numbers = []

        record_ids = [str(uuid.uuid4()) for _ in kept_hashes]
        now = _now()
        year = now.year
        month = now.month
        artifact_id = str(uuid.uuid4())
        out.insert(0, "__record_id", record_ids)
        out.insert(1, "__workspace_id", workspace_id)
        out.insert(2, "__file_id", file_id)
        out.insert(3, "__artifact_id", artifact_id)
        out.insert(4, "__uploaded_by_user_id", user_id)
        out.insert(5, "__source_filename", source_filename)
        out.insert(6, "__sheet_name", sheet_name)
        out.insert(7, "__source_row_number", source_row_numbers)
        out.insert(8, "__row_hash", kept_hashes)
        out.insert(9, "__ingested_at", now.isoformat())
        out.insert(10, "__stage", stage)
        out.insert(11, "__period_year", year)
        out.insert(12, "__period_month", month)

        partition_dir = (
            self.base_dir
            / stage
            / f"workspace_id={_slug(workspace_id)}"
            / f"year={year:04d}"
            / f"month={month:02d}"
            / f"file_id={_slug(file_id)}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / f"{_slug(sheet_name)}-{artifact_id[:8]}.parquet"
        out.to_parquet(parquet_path, index=False)

        artifact_metadata = {
            **(metadata or {}),
            "sheet_name": sheet_name,
            "source_filename": source_filename,
            "duplicate_rows": duplicate_rows,
            "input_rows": len(clean_df),
            "system_columns": sorted(SYSTEM_COLUMNS),
        }
        created_artifact_id = self.catalog.create_artifact(
            artifact_id=artifact_id,
            file_id=file_id,
            workspace_id=workspace_id,
            stage=stage,
            artifact_type=f"{stage}_parquet",
            storage_path=str(parquet_path),
            rows_count=len(out),
            columns=source_columns,
            metadata=artifact_metadata,
            created_by_user_id=user_id,
            source_artifact_id=source_artifact_id,
        )
        if stage == "raw":
            self.catalog.insert_record_hashes(
                workspace_id=workspace_id,
                records=list(zip(kept_hashes, record_ids)),
                file_id=file_id,
                artifact_id=created_artifact_id,
            )
        return {
            "id": created_artifact_id,
            "file_id": file_id,
            "workspace_id": workspace_id,
            "stage": stage,
            "storage_path": str(parquet_path),
            "rows_count": len(out),
            "columns": source_columns,
            "metadata": artifact_metadata,
        }

    def load_raw_sheet(self, file_id: str, sheet_name: str, row_limit: int) -> dict[str, Any] | None:
        artifact = self.catalog.get_raw_artifact_for_sheet(file_id, sheet_name)
        if not artifact:
            return None
        df = pd.read_parquet(artifact["storage_path"])
        columns = list(artifact["columns"])
        rows = self._frame_to_rows(df[columns].head(max(1, int(row_limit))))
        return {
            "total_rows": int(artifact["rows_count"]),
            "rendered_rows": len(rows),
            "columns": columns,
            "rows": rows,
        }

    def search_records(
        self,
        *,
        stage: str,
        workspace_id: str,
        filters: list[dict[str, Any]],
        columns: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        paths = self._stage_paths(stage, workspace_id)
        if not paths:
            return {"columns": columns or [], "rows": [], "total": 0, "engine": "empty"}
        try:
            return self._search_duckdb(paths, filters, columns, limit, offset)
        except Exception:
            return self._search_pandas(paths, filters, columns, limit, offset)

    def delete_records(
        self,
        *,
        stage: str,
        workspace_id: str,
        record_ids: Iterable[str],
    ) -> dict[str, Any]:
        target_ids = {str(record_id).strip() for record_id in record_ids if str(record_id).strip()}
        if not target_ids:
            return {"deleted_rows": 0, "affected_files": 0, "message": "Нет выбранных строк для удаления."}

        deleted_rows = 0
        affected_files = 0
        deleted_artifact_ids: set[str] = set()
        updated_artifact_counts: dict[str, int] = {}
        paths = [Path(path) for path in self._stage_paths(stage, workspace_id)]

        for path in paths:
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if "__record_id" not in df.columns:
                continue
            match = df["__record_id"].astype(str).isin(target_ids)
            remove_count = int(match.sum())
            if remove_count <= 0:
                continue

            affected_files += 1
            deleted_rows += remove_count
            removed_artifact_ids = self._artifact_ids_from_frame(df.loc[match])
            remaining = df.loc[~match].reset_index(drop=True)
            if remaining.empty:
                deleted_artifact_ids.update(removed_artifact_ids)
                path.unlink(missing_ok=True)
                self._remove_empty_parent_dirs(path)
                continue

            remaining.to_parquet(path, index=False)
            remaining_artifact_ids = self._artifact_ids_from_frame(remaining)
            deleted_artifact_ids.update(removed_artifact_ids - remaining_artifact_ids)
            for artifact_id, rows_count in remaining["__artifact_id"].astype(str).value_counts().items():
                if artifact_id:
                    updated_artifact_counts[str(artifact_id)] = int(rows_count)

        for artifact_id, rows_count in updated_artifact_counts.items():
            self.catalog.update_artifact_rows_count(artifact_id, rows_count)
        if deleted_artifact_ids:
            self.catalog.delete_artifacts(deleted_artifact_ids)
        if stage == "raw" and deleted_rows:
            self.catalog.delete_record_index_by_record_ids(workspace_id, target_ids)

        return {
            "deleted_rows": deleted_rows,
            "affected_files": affected_files,
            "message": f"Удалено строк: {deleted_rows}.",
        }

    def clear_stage(self, *, stage: str, workspace_id: str) -> dict[str, Any]:
        deleted_rows = 0
        affected_files = 0
        artifact_ids: set[str] = set()
        paths = [Path(path) for path in self._stage_paths(stage, workspace_id)]

        for path in paths:
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["__artifact_id"])
            deleted_rows += int(len(df))
            artifact_ids.update(str(value) for value in df["__artifact_id"].dropna().unique())
            affected_files += 1
            path.unlink(missing_ok=True)
            self._remove_empty_parent_dirs(path)

        if stage == "raw":
            self.catalog.delete_record_index_by_artifact_ids(workspace_id, artifact_ids)
        self.catalog.delete_artifacts(artifact_ids)
        return {
            "deleted_rows": deleted_rows,
            "affected_files": affected_files,
            "message": f"Слой очищен. Удалено строк: {deleted_rows}.",
        }

    def delete_file_artifacts(self, *, file_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        artifacts = [
            artifact for artifact in self.catalog.list_artifacts(file_id)
            if workspace_id is None or str(artifact.get("workspace_id")) == workspace_id
        ]
        if not artifacts:
            self.catalog.delete_file(file_id)
            return {"deleted_rows": 0, "affected_files": 0}

        deleted_rows = 0
        affected_files = 0
        artifact_ids: set[str] = set()
        artifact_ids_by_workspace: dict[str, set[str]] = {}
        for artifact in artifacts:
            artifact_id = str(artifact.get("id") or "")
            artifact_workspace_id = str(artifact.get("workspace_id") or workspace_id or "default")
            if artifact_id:
                artifact_ids.add(artifact_id)
                artifact_ids_by_workspace.setdefault(artifact_workspace_id, set()).add(artifact_id)
            deleted_rows += int(artifact.get("rows_count") or 0)
            path = Path(str(artifact.get("storage_path") or ""))
            if path.exists():
                path.unlink()
                affected_files += 1
                self._remove_empty_parent_dirs(path)

        for artifact_workspace_id, ids in artifact_ids_by_workspace.items():
            self.catalog.delete_record_index_by_artifact_ids(artifact_workspace_id, ids)
        self.catalog.delete_artifacts(artifact_ids)
        self.catalog.delete_file(file_id)
        return {"deleted_rows": deleted_rows, "affected_files": affected_files}

    def _stage_paths(self, stage: str, workspace_id: str) -> list[str]:
        root = self.base_dir / stage / f"workspace_id={_slug(workspace_id)}"
        if not root.exists():
            return []
        return [str(path) for path in root.rglob("*.parquet")]

    @staticmethod
    def _artifact_ids_from_frame(df: pd.DataFrame) -> set[str]:
        if "__artifact_id" not in df.columns:
            return set()
        return {str(value) for value in df["__artifact_id"].dropna().unique() if str(value)}

    def _remove_empty_parent_dirs(self, path: Path) -> None:
        root = self.base_dir.resolve()
        current = path.parent
        while current.exists() and current.resolve() != root:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def _search_duckdb(
        self,
        paths: list[str],
        filters: list[dict[str, Any]],
        columns: list[str] | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        import duckdb  # type: ignore

        projection_columns = self._projection_columns(columns)
        projection = "*" if not projection_columns else ", ".join(self._quote_identifier(column) for column in projection_columns)
        con = duckdb.connect()
        con.execute("PRAGMA threads=4")
        table_expr = "read_parquet(?)"
        available_columns = self._duckdb_columns(con, table_expr, paths)
        where_sql, params = self._duckdb_where(filters, available_columns)
        rows_df = con.execute(
            f"SELECT {projection} FROM {table_expr} {where_sql} LIMIT ? OFFSET ?",
            [paths, *params, int(limit), int(offset)],
        ).fetchdf()
        total = con.execute(f"SELECT COUNT(*) AS total FROM {table_expr} {where_sql}", [paths, *params]).fetchone()[0]
        visible_columns = [str(column) for column in rows_df.columns if str(column) not in SYSTEM_COLUMNS]
        rows = self._enrich_rows(self._frame_to_rows(rows_df))
        return {
            "columns": visible_columns,
            "rows": rows,
            "total": int(total),
            "engine": "duckdb",
        }

    @staticmethod
    def _duckdb_columns(con: Any, table_expr: str, paths: list[str]) -> set[str]:
        rows = con.execute(f"DESCRIBE SELECT * FROM {table_expr}", [paths]).fetchall()
        return {str(row[0]) for row in rows}

    def _duckdb_where(self, filters: list[dict[str, Any]], available_columns: set[str] | None = None) -> tuple[str, list[Any]]:
        parts: list[str] = []
        params: list[Any] = []
        for item in filters:
            column = str(item.get("column") or "").strip()
            op = str(item.get("op") or "eq")
            value = item.get("value")
            if op == "exists_any":
                fields = self._filter_field_names(item)
                existing_fields = [field for field in fields if available_columns is None or field in available_columns]
                if not existing_fields:
                    parts.append("FALSE")
                    continue
                field_checks = []
                for field in existing_fields:
                    ident = self._quote_identifier(field)
                    field_checks.append(
                        f"({ident} IS NOT NULL AND LOWER(TRIM(CAST({ident} AS VARCHAR))) NOT IN ('', 'nan', 'inf', '-inf'))"
                    )
                parts.append(f"({' OR '.join(field_checks)})")
                continue
            if not column:
                continue
            ident = self._quote_identifier(column)
            if op == "contains":
                parts.append(f"CAST({ident} AS VARCHAR) ILIKE ?")
                params.append(f"%{value}%")
            elif op == "ne":
                parts.append(f"{ident} <> ?")
                params.append(value)
            elif op == "in" and isinstance(value, list):
                placeholders = ", ".join("?" for _ in value)
                parts.append(f"{ident} IN ({placeholders})")
                params.extend(value)
            elif op == "between" and isinstance(value, list) and len(value) >= 2:
                parts.append(f"{ident} BETWEEN ? AND ?")
                params.extend(value[:2])
            elif op == "gte":
                parts.append(f"{ident} >= ?")
                params.append(value)
            elif op == "lte":
                parts.append(f"{ident} <= ?")
                params.append(value)
            else:
                parts.append(f"{ident} = ?")
                params.append(value)
        return (f"WHERE {' AND '.join(parts)}" if parts else ""), params

    @staticmethod
    def _quote_identifier(column: str) -> str:
        return '"' + column.replace('"', '""') + '"'

    def _search_pandas(
        self,
        paths: list[str],
        filters: list[dict[str, Any]],
        columns: list[str] | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        frames = [pd.read_parquet(path) for path in paths]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for item in filters:
            column = str(item.get("column") or "").strip()
            op = str(item.get("op") or "eq")
            value = item.get("value")
            if op == "exists_any":
                df = df[self._field_exists_mask(df, self._filter_field_names(item))]
                continue
            if not column or column not in df.columns:
                continue
            series = df[column]
            if op == "contains":
                df = df[series.astype(str).str.contains(str(value), case=False, na=False)]
            elif op == "ne":
                df = df[series != value]
            elif op == "in" and isinstance(value, list):
                df = df[series.isin(value)]
            elif op == "between" and isinstance(value, list) and len(value) >= 2:
                df = df[(series >= value[0]) & (series <= value[1])]
            elif op == "gte":
                df = df[series >= value]
            elif op == "lte":
                df = df[series <= value]
            else:
                df = df[series == value]
        visible_columns = [column for column in (columns or list(df.columns)) if column in df.columns and column not in SYSTEM_COLUMNS]
        page_columns = [column for column in [*visible_columns, *RECORD_METADATA_COLUMNS] if column in df.columns]
        page = df.iloc[int(offset):int(offset) + int(limit)]
        return {
            "columns": visible_columns,
            "rows": self._enrich_rows(self._frame_to_rows(page[page_columns])),
            "total": int(len(df)),
            "engine": "pandas",
        }

    @staticmethod
    def _filter_field_names(item: dict[str, Any]) -> list[str]:
        value = item.get("value")
        if isinstance(value, list):
            raw_values = value
        elif value not in (None, ""):
            raw_values = [value]
        else:
            raw_values = [item.get("column")]

        fields: list[str] = []
        for raw_value in raw_values:
            for field in re.split(r"[,;|\n]+", str(raw_value or "")):
                field = field.strip()
                if field and field not in fields:
                    fields.append(field)
        return fields

    @classmethod
    def _field_exists_mask(cls, df: pd.DataFrame, fields: list[str]) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        for field in fields:
            if field not in df.columns:
                continue
            mask = mask | df[field].map(cls._field_value_exists)
        return mask

    @staticmethod
    def _field_value_exists(value: Any) -> bool:
        if _is_missing_cell(value):
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    @staticmethod
    def _projection_columns(columns: list[str] | None) -> list[str]:
        if not columns:
            return []
        return [
            *[str(column) for column in columns if str(column).strip()],
            *[column for column in RECORD_METADATA_COLUMNS if column not in columns],
        ]

    def _enrich_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        users = self.catalog.users_by_ids(str(row.get("__uploaded_by_user_id") or "") for row in rows)
        for row in rows:
            user = users.get(str(row.get("__uploaded_by_user_id") or ""))
            if user:
                row["__uploaded_by_display_name"] = user.get("display_name") or user.get("email") or ""
                row["__uploaded_by_email"] = user.get("email") or ""
                row["__uploaded_by_role"] = user.get("workspace_role") or user.get("role") or ""
            else:
                row["__uploaded_by_display_name"] = ""
                row["__uploaded_by_email"] = ""
                row["__uploaded_by_role"] = ""
        return rows

    def _read_sheet(self, stored_path: Path, file_format: str, sheet_name: str) -> pd.DataFrame:
        if file_format == "csv" or stored_path.suffix.lower() == ".csv":
            return self._read_csv_frame(stored_path)
        return pd.read_excel(stored_path, sheet_name=sheet_name, dtype=object)

    @staticmethod
    def _read_csv_frame(path: Path) -> pd.DataFrame:
        errors: list[str] = []
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                df = pd.read_csv(path, dtype=object, sep=None, engine="python", encoding=encoding)
                df.columns = [str(column).strip() for column in df.columns]
                return df
            except Exception as exc:
                errors.append(f"{encoding}/auto: {exc}")
            for separator in (";", ",", "\t", "|"):
                try:
                    df = pd.read_csv(path, dtype=object, sep=separator, encoding=encoding)
                    if len(df.columns) > 1:
                        df.columns = [str(column).strip() for column in df.columns]
                        return df
                except Exception as exc:
                    errors.append(f"{encoding}/{separator}: {exc}")
        raise ValueError("; ".join(errors[:4]) or "Could not read CSV.")

    @staticmethod
    def _drop_empty_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for column in df.columns:
            name = str(column)
            if name.lower().startswith("unnamed") and df[column].isna().all():
                continue
            keep.append(column)
        return df.loc[:, keep]

    @staticmethod
    def _frame_to_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
        rows = df.to_dict(orient="records")
        return [{str(key): _json_safe_cell(value) for key, value in row.items()} for row in rows]
