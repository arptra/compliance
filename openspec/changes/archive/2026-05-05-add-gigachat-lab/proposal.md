# Proposal: Add GigaChat Lab

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Introduce an interactive GigaChat Lab for transport selection, settings, workbook upload, row inspection, and model execution on local and VM runtimes.

## Historical Evidence

- `01dc8a0`: GigaChat API router/services, token transport, Lab page, settings/transport/table components, tests, and local scripts.
- `8b2bf39`: VM start script.
- `bfff4ff`: refined Lab workflow, table UX, runtime host resolution, and VM lifecycle.
- `6d26d5c`, `93cec5c`: follow-up corrections.

## Outcome

GigaChat Lab became the product's interactive center and later the only mounted dashboard area. Current behavior is split across the `gigachat/*` main specs.
