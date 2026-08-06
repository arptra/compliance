# Proposal: Workbook Statistics And Lake Filters

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Show local rule-hit statistics for the current workbook and make the Parquet lake searchable by field existence with JSON-safe rows.

## Historical Evidence

- `8ffd32e`, `4afd27b`, `933e7e8`, `6665e48`: workbook statistics panel and chart refinements.
- `68400c7`: JSON-safe lake rows.
- `a8abdf6`: field-existence filters.

## Outcome

Users could inspect rule coverage before model calls and find lake rows where any selected source field existed.
