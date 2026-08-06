# Proposal: Direct GigaChat Transports

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Replace SDK-dependent GigaChat access with a diagnosable direct HTTPX client supporting TLS/mTLS and resilient structured responses.

## Historical Evidence

- `190f69c`, `bd7c30a`, `0a221ca`, `69e751d`: setup and mTLS diagnostics.
- `f7d91e5`: direct HTTPX chat completions.
- `e5dfade`, `d1882b1`: compact JSON normalization and repair payload compatibility.

## Outcome

The backend gained explicit transport construction, certificate diagnostics, and schema coercion. Current transport behavior is described in `openspec/specs/gigachat/transports/spec.md`.
