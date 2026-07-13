# Final OpenSpec Bootstrap Review

Use after full bootstrap synthesis or a major refresh. Read
`00-global-rules.md`, then execute `14-audit-sdd-coverage.md` using an
independent auditor worker when available.

Do not change production code. Small repairs to indexes and specs may be made
only from existing evidence. Missing evidence creates new work-queue items.

## Required Reviews

1. Structural and traceability validation from
   `05-add-openspec-validation.md`.
2. Independent repository coverage audit from
   `14-audit-sdd-coverage.md`.
3. Secret and private-data hygiene review.
4. Check that production code was not changed during bootstrap.
5. Check that bootstrap state, queue, and coverage index agree.

## Final Status Rules

- `READY`: all completion gates pass and no declared gaps remain.
- `READY_WITH_DECLARED_GAPS`: every in-scope area was processed and audited,
  but explicit unknowns, contradictions, missing tests, or missing contracts
  remain.
- `IN_PROGRESS`: pending, runnable, failed, or unaudited work remains.
- `BLOCKED`: only when an external dependency prevents further progress and all
  non-blocked work has completed.

Never use `READY` or `READY_WITH_DECLARED_GAPS` when manifest coverage is
incomplete.

## Finalize

Update:

- `openspec/index/coverage.yml`
- `openspec/bootstrap/state.yml`
- final bootstrap report under `openspec/bootstrap/reports/`
- `openspec/project.md` status and freshness links

Preserve worker findings for auditability unless repository policy explicitly
requires their removal.

## Output

Use the simple final format defined in `START_HERE.md`. Include only summary
counts and the next numbered menu; keep detailed findings in repository files.
