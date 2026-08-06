# Proposal: Novelty And Pattern Pipelines

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Add isolated novelty hunting plus standalone pattern-fit and pattern-monitor analysis paths, including CSV input.

## Historical Evidence

- `33e124e`: novelty-hunt pipeline and CLI command.
- `c0f51ec`: pattern-fit and pattern-monitor pipelines.
- `f2bfd89`, `d7d7af9`: CSV parsing and configurable delimiter.
- `23c19a6`, `1d6e139`, `85e4393`: safe and faster Excel exports.

## Outcome

Analysts could run novelty and pattern workflows independently and export results without illegal-cell failures.
