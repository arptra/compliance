# Proposal: Analyst Feedback And Reranker

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Capture analyst verdicts on pattern alerts and optionally rerank candidates using reviewed evidence.

## Historical Evidence

- `1772224`: feedback loop and optional reranker.
- `b357498` through `c86b787`: filtering, serialization, fit selection, and dialog evidence fixes.
- `eb06ec6`: per-row and bulk reset actions.

## Outcome

Pattern Monitor could collect review verdicts, summarize feedback, reset reviews, and feed an optional second-stage scorer.
