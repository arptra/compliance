# Proposal: Containerized Runtime

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Package API and dashboard for repeatable Docker deployment with external data mounts and configurable registries.

## Historical Evidence

- `0e9ffb6`, `b711e76`: compose, lifecycle scripts, and full data mount.
- `935f460`: Python 3.12 Alpine API image.
- `fe86000`: configurable Alpine, pip, and npm registries.

## Outcome

The stack could be rebuilt and run as API/dashboard services while keeping data and generated artifacts outside the image.
