# Proposal: Retire Legacy Web Routes

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Narrow the mounted web product to GigaChat Lab, its data/tasks, authentication, and profile while leaving analytical modules in the repository.

## Historical Evidence

- `b141692`: removed legacy dashboard route registrations, navigation entries, API router registration, and unused service wiring.

## Outcome

`create_app()` now mounts only health/auth/gigachat/records, and the React router exposes only the current Lab surface. This boundary is normative in current specs.
