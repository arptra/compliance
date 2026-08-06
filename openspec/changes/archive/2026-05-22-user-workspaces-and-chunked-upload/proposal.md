# Proposal: User Workspaces And Chunked Upload

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Move Lab data/settings to authenticated user-backed storage and support reliable large-file upload on VM deployments.

## Historical Evidence

- `0188e7f`: users, tokens, profiles, workspace catalog, record lake, auth UI, and user-backed Lab storage.
- `0dba962`, `7807b3d`: VM setup and dependency installation.
- `21f536e`: disable unsafe local workbook fallback on VM.
- `9921d7e`: chunked workbook upload.

## Outcome

The current auth/workspace boundary and Parquet lake were established, while large workbooks gained resumable task-oriented ingestion.
