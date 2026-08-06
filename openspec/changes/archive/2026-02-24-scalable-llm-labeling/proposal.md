# Proposal: Scalable LLM Labeling

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Make data preparation observable and resilient for large LLM labeling batches.

## Historical Evidence

- `d05d2f5`, `f497584`: stage/progress, token, and latency logs.
- `957535d`, `966b3a8`: full taxonomy/dialog context.
- `be157cb`, `e09185a`, `f6449a9`: bad-row isolation, token-limited batches, pending-only retries.
- `0d898b9`, `66ff5e0`: async/parallel modes and SQLite thread safety.

## Outcome

Preparation could process batches concurrently, retain validation failures per row, and report measurable progress instead of aborting the whole run.
