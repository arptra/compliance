# Proposal: Async Labeling And Rate Limits

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Expose worker-controlled asynchronous labeling in GigaChat Lab and retry GigaChat HTTP 429 responses with visible rate-limit handling.

## Historical Evidence

- `3529ce1`: Lab rule updates and async labeling controls.
- `53a7c89`: initial GigaChat API rate-limit handling.

## Outcome

Users could choose worker concurrency for row batches, and transports recognized 429 as a retryable rate-limit response. Later commits replaced this with a shared adaptive queue.
