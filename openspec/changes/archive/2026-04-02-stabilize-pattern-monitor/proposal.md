# Proposal: Stabilize Pattern Monitor

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Make manual Pattern Monitor runs deterministic: correct source, filters, loading state, concurrency guard, and alert-only table output.

## Historical Evidence

- `560906f`, `f5d1048`: HTTP failures, concurrent-run guard, and manual gated start.
- `d5ae85e` through `e644ce8`: loading overlay, row limits, output source selection, stale-state reset.
- `bda537c`, `fc4c160`, `cc84d0a`, `3af3d66`, `f8f4564`, `0078012`: visibility, refetch, render, and state fixes.

## Outcome

The monitor table followed the latest explicit run and its filters instead of mixing cached or unrelated artifacts.
