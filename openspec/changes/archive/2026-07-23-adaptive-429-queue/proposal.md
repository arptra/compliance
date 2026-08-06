# Proposal: Adaptive HTTP 429 Queue

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Coordinate all GigaChat workers through a shared observable limiter that serializes after HTTP 429, probes recovery, returns to parallel dispatch, and avoids unnecessary fixed delays.

## Historical Evidence

- `a4d8a76`: prompt-pack guidance for serial requests after 429.
- `d3ef9f6`: queue requests after rate limiting.
- `7d82a0d`: return from probe to parallel queue processing.
- `019bdd2`: remove throughput bottlenecks and improve batch speed/logging.

## Outcome

The final limiter logs dispatch/completion/rate/queue transitions, uses a global FIFO serial mode after 429, and releases waiting workers after a successful control request. Current behavior is normative in `gigachat/background-processing`.
