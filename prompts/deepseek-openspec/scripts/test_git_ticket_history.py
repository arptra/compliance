#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("git_ticket_history.py")


class GitTicketHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.git("init")
        self.git("config", "user.name", "Test User")
        self.git("config", "user.email", "test@example.com")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout

    def tool(self, *args: str) -> dict:
        result = subprocess.run(
            ["python3", str(SCRIPT), *args, "--repo", str(self.repo)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return json.loads(result.stdout)

    def scan(self) -> dict:
        return self.tool(
            "scan",
            "--prefix",
            "PROJ-",
            "--include-ref-names",
            "--include-reflog",
        )

    def test_empty_repository(self) -> None:
        result = self.scan()
        self.assertEqual(result["scanned_commits"], 0)
        self.assertEqual(result["tickets"], 0)

    def test_exact_prefix_body_and_ref_matches(self) -> None:
        self.git(
            "commit",
            "--allow-empty",
            "-m",
            "PROJ-101 add importer",
            "-m",
            "Also handles PROJ-102. PROJ_999 is a different format.",
        )
        noted_commit = self.git("rev-parse", "HEAD").strip()
        self.git("notes", "add", "-m", "PROJ-303 documented in a Git note", noted_commit)
        self.git("checkout", "-b", "feature/PROJ-202")
        self.git("commit", "--allow-empty", "-m", "maintenance")

        result = self.scan()
        store = Path(result["store"])
        index = json.loads((store / "index.json").read_text(encoding="utf-8"))
        ids = [item["ticket_id"] for item in index["tickets"]]
        self.assertEqual(ids, ["PROJ-101", "PROJ-102", "PROJ-202", "PROJ-303"])

        ticket_item = next(item for item in index["tickets"] if item["ticket_id"] == "PROJ-101")
        persisted = json.loads((store / ticket_item["ticket_file"]).read_text(encoding="utf-8"))
        self.assertNotIn("message", persisted["commits"][0])
        self.assertNotIn("subject", persisted["commits"][0])

        context = self.tool("context", "--ticket", "PROJ-101")
        commit = context["commits"][0]
        self.assertEqual(commit["matching_message_line_numbers"], [1])
        self.assertNotIn("PROJ_999", ids)

        note_context = self.tool("context", "--ticket", "PROJ-303")
        self.assertIn("PROJ-303", note_context["commits"][0]["notes"])

    def test_context_is_paginated_and_merge_diff_uses_first_parent(self) -> None:
        tracked = self.repo / "feature.txt"
        tracked.write_text("base\n", encoding="utf-8")
        self.git("add", "feature.txt")
        self.git("commit", "-m", "baseline")
        base_branch = self.git("branch", "--show-current").strip()

        self.git("checkout", "-b", "ticket-work")
        tracked.write_text("base\nfeature\n", encoding="utf-8")
        self.git("commit", "-am", "implement feature")
        self.git("checkout", base_branch)
        self.git("merge", "--no-ff", "ticket-work", "-m", "PROJ-500 merge feature")
        self.git("commit", "--allow-empty", "-m", "PROJ-500 follow-up")

        self.scan()
        first = self.tool("context", "--ticket", "PROJ-500", "--max-commits", "1")
        self.assertTrue(first["commit_window"]["has_more"])
        self.assertEqual(first["commit_window"]["next_offset"], 1)
        self.assertEqual(first["commits"][0]["file_changes"][0]["path"], "feature.txt")

        second = self.tool(
            "context",
            "--ticket",
            "PROJ-500",
            "--max-commits",
            "1",
            "--offset",
            "1",
        )
        self.assertFalse(second["commit_window"]["has_more"])
        self.assertEqual(len(second["commits"]), 1)

    def test_completed_analysis_is_reused_then_marked_stale(self) -> None:
        self.git("commit", "--allow-empty", "-m", "PROJ-101 initial change")
        result = self.scan()
        store = Path(result["store"])
        index = json.loads((store / "index.json").read_text(encoding="utf-8"))
        item = index["tickets"][0]
        analysis = store / item["analysis_file"]
        analysis.parent.mkdir(parents=True, exist_ok=True)
        analysis.write_text("# PROJ-101\n", encoding="utf-8")

        self.tool("mark", "--ticket", "PROJ-101", "--status", "completed")
        self.scan()
        stable = json.loads((store / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(stable["tickets"][0]["analysis_status"], "completed")

        self.git("commit", "--allow-empty", "-m", "PROJ-101 follow-up")
        self.scan()
        changed = json.loads((store / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(changed["tickets"][0]["analysis_status"], "stale")


if __name__ == "__main__":
    unittest.main()
