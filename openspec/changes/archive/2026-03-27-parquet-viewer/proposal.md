# Proposal: Parquet Viewer

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Let users inspect prepared Parquet data directly in the dashboard with safe previews.

## Historical Evidence

- `b4dc5e5`: viewer tab and metadata preview endpoint.
- `e89544f`: loading and default date-range fixes.
- `2730342`: JSON-safe complex cell encoding.

## Outcome

The dashboard could preview Parquet columns and rows, including complex values. The page/API registration is historical after legacy route retirement.
