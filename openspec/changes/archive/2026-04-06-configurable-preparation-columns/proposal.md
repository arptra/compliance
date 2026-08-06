# Proposal: Configurable Preparation Columns

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Preserve configurable service columns across preparation and monitoring while making uploaded data easier to run without manual schema setup.

## Historical Evidence

- `cad0a3f`: configurable service columns.
- `bd19eef`, `05cfe07`: propagation and duplicate-name fixes.
- `cbe9b2e`: llm_mock fallback when certificates are unavailable.
- `84f6c9c`: datetime/dialog column auto-detection.

## Outcome

Uploaded datasets could carry selected service fields through monitoring, avoid duplicate-column failures, and start with detected source columns.
