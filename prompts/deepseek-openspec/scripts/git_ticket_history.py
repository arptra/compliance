#!/usr/bin/env python3
"""Index ticket IDs from local Git history and prepare bounded ticket context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


class GitTicketError(RuntimeError):
    pass


def run_git(repo: Path, args: list[str], *, text: bool = True) -> str | bytes:
    command = ["git", "-C", str(repo), *args]
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise GitTicketError(f"Git command failed: {' '.join(command)}\n{message}") from exc
    if text:
        return result.stdout.decode("utf-8", errors="replace")
    return result.stdout


def repo_root(repo: str) -> Path:
    candidate = Path(repo).expanduser().resolve()
    root = run_git(candidate, ["rev-parse", "--show-toplevel"])
    return Path(str(root).strip()).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_prefix(prefix: str) -> str:
    if not prefix or len(prefix) > 64:
        raise GitTicketError("Ticket prefix must contain 1-64 characters")
    if "\x00" in prefix or "\n" in prefix or "\r" in prefix:
        raise GitTicketError("Ticket prefix cannot contain NUL or newline characters")
    if not any(char.isalnum() for char in prefix):
        raise GitTicketError("Ticket prefix must contain at least one letter or digit")
    return prefix


def prefix_key(prefix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("._").lower()
    if not normalized:
        normalized = "tickets"
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"


def ticket_file_name(ticket_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ticket_id)
    if safe != ticket_id:
        safe = f"{safe}-{hashlib.sha256(ticket_id.encode('utf-8')).hexdigest()[:8]}"
    return f"{safe}.json"


def ticket_pattern(prefix: str, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(prefix)}(?P<number>[0-9]+)(?![A-Za-z0-9])",
        flags,
    )


def canonical_ticket(prefix: str, match: re.Match[str]) -> str:
    return f"{prefix}{match.group('number')}"


def exact_ticket_pattern(ticket_id: str, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(ticket_id)}(?![A-Za-z0-9])",
        flags,
    )


def parse_git_log(repo: Path, include_reflog: bool) -> tuple[int, list[dict[str, Any]]]:
    args = [
        "log",
        "--all",
        "--show-notes=*",
        "-z",
        "--date=iso-strict",
        "--format=%H%x00%P%x00%aI%x00%B%x00%N",
    ]
    if include_reflog:
        args.insert(2, "--reflog")
    raw = run_git(repo, args, text=False)
    fields = bytes(raw).split(b"\x00")
    # `git log -z` contributes one final record terminator. Empty `%N` notes are
    # real fields and must not be trimmed.
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 5 != 0:
        raise GitTicketError(
            f"Unexpected git log record shape: {len(fields)} fields is not divisible by 5"
        )

    commits: list[dict[str, Any]] = []
    for index in range(0, len(fields), 5):
        oid, parents, authored_at, message, notes = (
            field.decode("utf-8", errors="replace") for field in fields[index : index + 5]
        )
        commits.append(
            {
                "oid": oid.strip(),
                "parents": [value for value in parents.strip().split() if value],
                "authored_at": authored_at.strip() or None,
                "message": message.rstrip(),
                "notes": notes.rstrip(),
                "_scan_order": len(commits),
            }
        )
    unique: dict[str, dict[str, Any]] = {}
    for commit in commits:
        if commit["oid"] and commit["oid"] not in unique:
            unique[commit["oid"]] = commit
    return len(unique), list(unique.values())


def matching_line_numbers(text: str, pattern: re.Pattern[str], limit: int = 20) -> list[int]:
    result: list[int] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            result.append(number)
            if len(result) >= limit:
                break
    return result


def refs_with_tickets(
    repo: Path, prefix: str, pattern: re.Pattern[str]
) -> dict[str, list[dict[str, str]]]:
    raw = run_git(repo, ["for-each-ref", "--format=%(refname)%00%(objectname)%00"])
    fields = str(raw).split("\x00")
    result: dict[str, list[dict[str, str]]] = {}
    for index in range(0, len(fields) - 1, 2):
        ref = fields[index].strip()
        oid = fields[index + 1].strip()
        if not ref or not oid:
            continue
        for match in pattern.finditer(ref):
            ticket_id = canonical_ticket(prefix, match)
            result.setdefault(ticket_id, []).append({"ref": ref, "object": oid})
    return result


def refs_fingerprint(repo: Path, include_reflog: bool) -> str:
    try:
        refs_text = str(run_git(repo, ["show-ref", "--head"]))
    except GitTicketError:
        refs_text = ""
    refs = refs_text.encode("utf-8")
    digest = hashlib.sha256(refs)
    digest.update(b"\x00reflog=")
    digest.update(str(include_reflog).lower().encode("ascii"))
    if include_reflog:
        try:
            reflog = str(run_git(repo, ["reflog", "--all", "--format=%H%x00%gD"]))
        except GitTicketError:
            reflog = ""
        digest.update(b"\x00")
        digest.update(reflog.encode("utf-8"))
    return digest.hexdigest()


def resolve_scan_store(root: Path, prefix: str, output: str | None) -> tuple[Path, Path]:
    history_root = root / "openspec" / "history" / "git-tickets"
    if output:
        store = Path(output).expanduser()
        if not store.is_absolute():
            store = root / store
        return history_root, store.resolve()
    return history_root, history_root / prefix_key(prefix)


def update_registry(history_root: Path, prefix: str, store: Path) -> None:
    registry_path = history_root / "registry.json"
    registry = load_json(registry_path, {"schema_version": SCHEMA_VERSION, "indexes": []})
    indexes = [item for item in registry.get("indexes", []) if item.get("prefix") != prefix]
    try:
        relative_store = str(store.relative_to(history_root))
    except ValueError:
        relative_store = str(store)
    indexes.append(
        {
            "prefix": prefix,
            "store": relative_store,
            "updated_at": utc_now(),
        }
    )
    registry["schema_version"] = SCHEMA_VERSION
    registry["indexes"] = sorted(indexes, key=lambda item: item["prefix"])
    atomic_write_json(registry_path, registry)


def ticket_signature(record: dict[str, Any]) -> str:
    stable = {
        "ticket_id": record["ticket_id"],
        "commits": [
            item["oid"]
            for item in sorted(
                record.get("commits", []),
                key=lambda value: (value.get("authored_at") or "", value["oid"]),
            )
        ],
        "ref_matches": record.get("ref_matches", []),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scan(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    prefix = validate_prefix(args.prefix)
    history_root, store = resolve_scan_store(root, prefix, args.output)
    pattern = ticket_pattern(prefix, args.case_sensitive)
    scanned_count, commits = parse_git_log(root, args.include_reflog)
    ref_matches = refs_with_tickets(root, prefix, pattern) if args.include_ref_names else {}

    tickets: dict[str, dict[str, Any]] = {}
    matched_commits = 0
    for commit in commits:
        message_matches = list(pattern.finditer(commit["message"]))
        note_matches = list(pattern.finditer(commit["notes"]))
        ticket_ids = {
            canonical_ticket(prefix, match) for match in [*message_matches, *note_matches]
        }
        if not ticket_ids:
            continue
        matched_commits += 1
        for ticket_id in ticket_ids:
            specific_pattern = exact_ticket_pattern(ticket_id, args.case_sensitive)
            record = tickets.setdefault(
                ticket_id,
                {
                    "schema_version": SCHEMA_VERSION,
                    "ticket_id": ticket_id,
                    "prefix": prefix,
                    "commits": [],
                    "ref_matches": [],
                },
            )
            sources: list[str] = []
            if any(canonical_ticket(prefix, match) == ticket_id for match in message_matches):
                sources.append("commit_message")
            if any(canonical_ticket(prefix, match) == ticket_id for match in note_matches):
                sources.append("git_note")
            record["commits"].append(
                {
                    "oid": commit["oid"],
                    "parents": commit["parents"],
                    "authored_at": commit["authored_at"],
                    "sources": sources,
                    "matching_message_line_numbers": matching_line_numbers(
                        commit["message"], specific_pattern
                    ),
                    "matching_note_line_numbers": matching_line_numbers(
                        commit["notes"], specific_pattern
                    ),
                    "is_merge": len(commit["parents"]) > 1,
                    "_scan_order": commit["_scan_order"],
                }
            )

    for ticket_id, matches in ref_matches.items():
        record = tickets.setdefault(
            ticket_id,
            {
                "schema_version": SCHEMA_VERSION,
                "ticket_id": ticket_id,
                "prefix": prefix,
                "commits": [],
                "ref_matches": [],
            },
        )
        record["ref_matches"] = sorted(matches, key=lambda item: item["ref"])

    existing_queue = load_json(store / "queue.json", {"items": []})
    previous_items = {item.get("ticket_id"): item for item in existing_queue.get("items", [])}
    scheduler = normalized_scheduler(existing_queue.get("scheduler"))
    index_items: list[dict[str, Any]] = []
    queue_items: list[dict[str, Any]] = []
    tickets_dir = store / "tickets"
    for ticket_id in sorted(tickets):
        record = tickets[ticket_id]
        record["commits"] = sorted(
            record["commits"],
            key=lambda item: (
                item.get("authored_at") or "",
                -int(item.get("_scan_order", 0)),
                item["oid"],
            ),
        )
        for commit in record["commits"]:
            commit.pop("_scan_order", None)
        record["commit_count"] = len(record["commits"])
        record["first_commit_at"] = (
            record["commits"][0].get("authored_at") if record["commits"] else None
        )
        record["last_commit_at"] = (
            record["commits"][-1].get("authored_at") if record["commits"] else None
        )
        record["signature"] = ticket_signature(record)
        file_name = ticket_file_name(ticket_id)
        atomic_write_json(tickets_dir / file_name, record)

        previous = previous_items.get(ticket_id, {})
        unchanged = previous.get("signature") == record["signature"]
        status = previous.get("status", "pending") if unchanged else "stale"
        if status == "running":
            status = "pending"
        if not previous:
            status = "pending"
        analysis_file = f"analyses/{Path(file_name).stem}.md"
        queue_items.append(
            {
                "ticket_id": ticket_id,
                "ticket_file": f"tickets/{file_name}",
                "analysis_file": analysis_file,
                "signature": record["signature"],
                "status": status,
                "enqueued_at": previous.get("enqueued_at") or utc_now(),
                "attempts": previous.get("attempts", 0) if unchanged else 0,
                "last_error": previous.get("last_error") if unchanged else None,
                "last_http_status": (
                    previous.get("last_http_status") if unchanged else None
                ),
                "cost": {
                    "commits": record["commit_count"],
                    "ref_matches": len(record["ref_matches"]),
                },
            }
        )
        index_items.append(
            {
                "ticket_id": ticket_id,
                "ticket_file": f"tickets/{file_name}",
                "analysis_file": analysis_file,
                "analysis_status": status,
                "commit_count": record["commit_count"],
                "first_commit_at": record["first_commit_at"],
                "last_commit_at": record["last_commit_at"],
                "signature": record["signature"],
            }
        )

    index = {
        "schema_version": SCHEMA_VERSION,
        "prefix": prefix,
        "ticket_pattern": pattern.pattern,
        "ticket_count": len(index_items),
        "tickets": index_items,
    }
    queue = {
        "schema_version": SCHEMA_VERSION,
        "prefix": prefix,
        "updated_at": utc_now(),
        "scheduler": scheduler,
        "items": queue_items,
        "summary": queue_summary(queue_items),
    }
    try:
        head = str(run_git(root, ["rev-parse", "HEAD"])).strip()
    except GitTicketError:
        head = None
    meta = {
        "schema_version": SCHEMA_VERSION,
        "repository_root": ".",
        "prefix": prefix,
        "case_sensitive": args.case_sensitive,
        "include_reflog": args.include_reflog,
        "include_ref_names": args.include_ref_names,
        "refs_fingerprint": refs_fingerprint(root, args.include_reflog),
        "head": head,
        "scanned_commit_count": scanned_count,
        "matched_commit_count": matched_commits,
        "ticket_count": len(index_items),
        "generated_at": utc_now(),
    }
    atomic_write_json(store / "meta.json", meta)
    atomic_write_json(store / "index.json", index)
    atomic_write_json(store / "queue.json", queue)
    update_registry(history_root, prefix, store)
    print(
        json.dumps(
            {
                "store": str(store),
                "prefix": prefix,
                "scanned_commits": scanned_count,
                "matched_commits": matched_commits,
                "tickets": len(index_items),
                "queue": queue["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def queue_summary(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {"total": 0}
    for item in items:
        summary["total"] += 1
        status = str(item.get("status", "unknown"))
        summary[status] = summary.get(status, 0) + 1
    return summary


def default_scheduler() -> dict[str, Any]:
    return {
        "dispatch_mode": "PARALLEL",
        "current_concurrency": None,
        "rate_limit_latched": False,
        "rate_limited_at": None,
        "retry_after": None,
        "retry_not_before": None,
        "rate_limit_event_count": 0,
        "last_429": None,
    }


def normalized_scheduler(value: Any) -> dict[str, Any]:
    scheduler = default_scheduler()
    if isinstance(value, dict):
        scheduler.update(value)
    if scheduler.get("rate_limit_latched"):
        scheduler["dispatch_mode"] = "GLOBAL_SERIAL_QUEUE"
        scheduler["current_concurrency"] = 1
    return scheduler


def bounded_log_value(value: str | None, limit: int = 500) -> str | None:
    if value is None:
        return None
    return " ".join(value.split())[:limit]


def retry_deadline(value: str | None) -> str | None:
    if not value:
        return None
    now = datetime.now(timezone.utc)
    if value.isdigit():
        try:
            deadline = now + timedelta(seconds=int(value))
        except OverflowError:
            return None
        return deadline.replace(microsecond=0).isoformat()
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def history_root(root: Path, output: str | None) -> Path:
    if output:
        path = Path(output).expanduser()
        return (root / path).resolve() if not path.is_absolute() else path.resolve()
    return root / "openspec" / "history" / "git-tickets"


def registered_stores(root: Path, output: str | None) -> list[Path]:
    base = history_root(root, output)
    if (base / "index.json").exists():
        return [base]
    registry = load_json(base / "registry.json", {"indexes": []})
    stores: list[Path] = []
    for item in registry.get("indexes", []):
        value = Path(item["store"])
        stores.append(value if value.is_absolute() else base / value)
    return stores


def find_store(root: Path, output: str | None, prefix: str | None, ticket: str | None) -> Path:
    stores = registered_stores(root, output)
    matches: list[Path] = []
    for store in stores:
        index = load_json(store / "index.json", {})
        if prefix and index.get("prefix") != prefix:
            continue
        if ticket and ticket not in {item.get("ticket_id") for item in index.get("tickets", [])}:
            continue
        matches.append(store)
    if not matches:
        raise GitTicketError("No matching Git ticket index found")
    if len(matches) > 1:
        raise GitTicketError("Multiple indexes match; provide --prefix or --output")
    return matches[0]


def list_tickets(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    store = find_store(root, args.output, args.prefix, None)
    index = load_json(store / "index.json", {})
    tickets = index.get("tickets", [])
    if args.json:
        print(json.dumps(tickets, ensure_ascii=False, indent=2))
    else:
        for item in tickets:
            print(
                f"{item['ticket_id']}\t{item['analysis_status']}\t"
                f"{item['commit_count']} commits\t{item.get('last_commit_at') or '-'}"
            )
    return 0


def status(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    store = find_store(root, args.output, args.prefix, None)
    meta = load_json(store / "meta.json", {})
    queue = load_json(store / "queue.json", {"items": []})
    result = {
        "store": str(store),
        "prefix": meta.get("prefix"),
        "generated_at": meta.get("generated_at"),
        "refs_fingerprint": meta.get("refs_fingerprint"),
        "current_refs_fingerprint": refs_fingerprint(root, bool(meta.get("include_reflog"))),
        "scheduler": normalized_scheduler(queue.get("scheduler")),
        "queue": queue_summary(queue.get("items", [])),
    }
    result["fresh"] = result["refs_fingerprint"] == result["current_refs_fingerprint"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def commit_message_and_notes(root: Path, oid: str) -> tuple[str, str]:
    message = str(run_git(root, ["show", "-s", "--format=%B", oid])).rstrip()
    notes = str(
        run_git(root, ["show", "-s", "--show-notes=*", "--format=%N", oid])
    ).rstrip()
    return message, notes


def commit_file_changes(
    root: Path, oid: str, parents: list[str]
) -> list[dict[str, Any]]:
    revisions = [parents[0], oid] if parents else [oid]
    root_option = [] if parents else ["--root"]
    raw = str(
        run_git(
            root,
            [
                "diff-tree",
                *root_option,
                "--no-commit-id",
                "-r",
                "-M",
                "--numstat",
                *revisions,
            ],
        )
    )
    result: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        additions, deletions, path = parts
        result.append(
            {
                "path": path,
                "additions": None if additions == "-" else int(additions),
                "deletions": None if deletions == "-" else int(deletions),
                "binary": additions == "-" or deletions == "-",
            }
        )
    return result


def commit_patch(root: Path, oid: str, parents: list[str]) -> bytes:
    common = ["--find-renames", "--find-copies", "--no-ext-diff", "--patch"]
    if parents:
        return bytes(run_git(root, ["diff", *common, parents[0], oid], text=False))
    return bytes(run_git(root, ["show", "--format=", *common, oid], text=False))


def truncate_text(value: str, max_chars: int) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def context(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    store = find_store(root, args.output, args.prefix, args.ticket)
    index = load_json(store / "index.json", {})
    item = next(
        (value for value in index.get("tickets", []) if value.get("ticket_id") == args.ticket),
        None,
    )
    if not item:
        raise GitTicketError(f"Ticket not found: {args.ticket}")
    record = load_json(store / item["ticket_file"], {})
    all_commits = record.get("commits", [])
    offset = max(0, args.offset)
    selected_commits = all_commits[offset : offset + max(1, args.max_commits)]
    commits: list[dict[str, Any]] = []
    patch_budget = max(0, args.max_patch_bytes)
    for commit in selected_commits:
        oid = commit["oid"]
        parents = commit.get("parents", [])
        message, notes = commit_message_and_notes(root, oid)
        message, message_truncated = truncate_text(message, max(1, args.max_message_chars))
        notes, notes_truncated = truncate_text(notes, max(1, args.max_message_chars))
        file_changes = commit_file_changes(root, oid, parents)
        files_truncated = len(file_changes) > max(1, args.max_files_per_commit)
        value = {
            **commit,
            "message": message,
            "message_truncated": message_truncated,
            "notes": notes,
            "notes_truncated": notes_truncated,
            "subject": message.splitlines()[0][:500] if message else "",
            "file_changes": file_changes[: max(1, args.max_files_per_commit)],
            "file_changes_truncated": files_truncated,
        }
        if args.include_patch and patch_budget > 0:
            patch = commit_patch(root, oid, parents)
            truncated = len(patch) > patch_budget
            selected = patch[:patch_budget]
            value["patch"] = selected.decode("utf-8", errors="replace")
            value["patch_truncated"] = truncated
            patch_budget -= len(selected)
        commits.append(value)
    result = {
        "schema_version": SCHEMA_VERSION,
        "ticket_id": args.ticket,
        "signature": record.get("signature"),
        "store": str(store),
        "commits": commits,
        "commit_window": {
            "offset": offset,
            "limit": max(1, args.max_commits),
            "returned": len(commits),
            "total": len(all_commits),
            "has_more": offset + len(commits) < len(all_commits),
            "next_offset": offset + len(commits),
        },
        "ref_matches": record.get("ref_matches", []),
        "analysis_file": item.get("analysis_file"),
        "analysis_status": item.get("analysis_status"),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    return 0


def mark(args: argparse.Namespace) -> int:
    root = repo_root(args.repo)
    store = find_store(root, args.output, args.prefix, args.ticket)
    queue_path = store / "queue.json"
    index_path = store / "index.json"
    queue = load_json(queue_path, {"items": []})
    index = load_json(index_path, {"tickets": []})
    scheduler = normalized_scheduler(queue.get("scheduler"))
    queue_item = next(
        (item for item in queue.get("items", []) if item.get("ticket_id") == args.ticket),
        None,
    )
    index_item = next(
        (item for item in index.get("tickets", []) if item.get("ticket_id") == args.ticket),
        None,
    )
    if not queue_item or not index_item:
        raise GitTicketError(f"Ticket not found: {args.ticket}")
    if args.status == "rate_limited" and args.http_status != 429:
        raise GitTicketError("rate_limited status requires --http-status 429")
    if args.status != "rate_limited" and args.http_status == 429:
        raise GitTicketError("HTTP 429 must be recorded with --status rate_limited")
    if args.status == "running" and scheduler.get("rate_limit_latched"):
        running_others = [
            item
            for item in queue.get("items", [])
            if item.get("status") == "running" and item.get("ticket_id") != args.ticket
        ]
        if running_others:
            raise GitTicketError(
                "Global serial queue already has a running ticket: "
                f"{running_others[0].get('ticket_id')}"
            )
        eligible = sorted(
            (
                item
                for item in queue.get("items", [])
                if item.get("status") in {"pending", "stale", "failed", "rate_limited"}
            ),
            key=lambda item: (item.get("enqueued_at") or "", item.get("ticket_id") or ""),
        )
        if eligible and eligible[0].get("ticket_id") != args.ticket:
            raise GitTicketError(
                "Global serial queue must preserve FIFO order; next ticket is "
                f"{eligible[0].get('ticket_id')}"
            )
    if args.status == "completed":
        analysis_path = store / index_item["analysis_file"]
        if not analysis_path.exists():
            raise GitTicketError(f"Analysis file does not exist: {analysis_path}")
    now = utc_now()
    queue_item["status"] = args.status
    if args.status == "running":
        queue_item["attempts"] = int(queue_item.get("attempts", 0)) + 1
    queue_item["last_error"] = bounded_log_value(args.error)
    queue_item["last_http_status"] = args.http_status
    queue_item["enqueued_at"] = queue_item.get("enqueued_at") or now
    log_records: list[str] = []
    user_message: str | None = None
    if args.status == "rate_limited":
        worker_id = bounded_log_value(args.worker_id, 128) or "unknown"
        request_id = bounded_log_value(args.request_id, 128) or "unknown"
        retry_after = bounded_log_value(args.retry_after, 128) or "unknown"
        scheduler.update(
            {
                "dispatch_mode": "GLOBAL_SERIAL_QUEUE",
                "current_concurrency": 1,
                "rate_limit_latched": True,
                "rate_limited_at": now,
                "retry_after": None if retry_after == "unknown" else retry_after,
                "retry_not_before": retry_deadline(
                    None if retry_after == "unknown" else retry_after
                ),
                "rate_limit_event_count": int(
                    scheduler.get("rate_limit_event_count", 0)
                )
                + 1,
                "last_429": {
                    "worker_id": worker_id,
                    "request_id": request_id,
                    "retry_after": retry_after,
                    "ticket_id": args.ticket,
                    "at": now,
                },
            }
        )
        waiting_count = sum(
            item.get("status") in {"pending", "stale", "failed", "rate_limited"}
            for item in queue.get("items", [])
        )
        log_records = [
            "WARN [RATE_LIMIT] status=429 "
            f"worker={worker_id} request={request_id} retry_after={retry_after}",
            "INFO [SCHEDULER] parallel dispatch stopped; "
            f"{waiting_count} requests are in the global FIFO queue; concurrency=1",
        ]
        user_message = (
            "HTTP 429: parallel dispatch stopped; requests entered one global "
            f"FIFO queue; concurrency=1; waiting={waiting_count}"
        )
    index_item["analysis_status"] = args.status
    queue["scheduler"] = scheduler
    queue["summary"] = queue_summary(queue.get("items", []))
    queue["updated_at"] = utc_now()
    atomic_write_json(queue_path, queue)
    atomic_write_json(index_path, index)
    result = {
        "ticket_id": args.ticket,
        "status": args.status,
        "scheduler": scheduler,
    }
    if log_records:
        result["log_records"] = log_records
        result["user_message"] = user_message
        for record in log_records:
            print(record, file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan local Git history for ticket IDs")
    scan_parser.add_argument("--repo", default=".")
    scan_parser.add_argument("--prefix", required=True)
    scan_parser.add_argument("--output")
    scan_parser.add_argument("--case-sensitive", action="store_true")
    scan_parser.add_argument("--include-reflog", action="store_true")
    scan_parser.add_argument("--include-ref-names", action="store_true")
    scan_parser.set_defaults(func=scan)

    list_parser = subparsers.add_parser("list", help="List indexed tickets")
    list_parser.add_argument("--repo", default=".")
    list_parser.add_argument("--prefix")
    list_parser.add_argument("--output")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=list_tickets)

    status_parser = subparsers.add_parser("status", help="Show index and queue status")
    status_parser.add_argument("--repo", default=".")
    status_parser.add_argument("--prefix")
    status_parser.add_argument("--output")
    status_parser.set_defaults(func=status)

    context_parser = subparsers.add_parser("context", help="Build bounded context for one ticket")
    context_parser.add_argument("--repo", default=".")
    context_parser.add_argument("--ticket", required=True)
    context_parser.add_argument("--prefix")
    context_parser.add_argument("--output")
    context_parser.add_argument("--include-patch", action="store_true")
    context_parser.add_argument("--max-patch-bytes", type=int, default=200000)
    context_parser.add_argument("--offset", type=int, default=0)
    context_parser.add_argument("--max-commits", type=int, default=50)
    context_parser.add_argument("--max-message-chars", type=int, default=20000)
    context_parser.add_argument("--max-files-per-commit", type=int, default=500)
    context_parser.set_defaults(func=context)

    mark_parser = subparsers.add_parser("mark", help="Update analysis status for one ticket")
    mark_parser.add_argument("--repo", default=".")
    mark_parser.add_argument("--ticket", required=True)
    mark_parser.add_argument("--prefix")
    mark_parser.add_argument("--output")
    mark_parser.add_argument(
        "--status",
        choices=["pending", "running", "completed", "failed", "stale", "rate_limited"],
        required=True,
    )
    mark_parser.add_argument("--error")
    mark_parser.add_argument("--http-status", type=int)
    mark_parser.add_argument("--worker-id")
    mark_parser.add_argument("--request-id")
    mark_parser.add_argument("--retry-after")
    mark_parser.set_defaults(func=mark)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (GitTicketError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
