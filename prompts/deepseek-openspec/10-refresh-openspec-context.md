# Refresh OpenSpec Context Incrementally

Use after repository changes, stale evidence, a major merge, or an explicit
user request. Read `00-global-rules.md` first. Do not change production code.

Before repository refresh, compare current `prompt-pack.yml` with
`openspec/meta.yml`. Run `16-upgrade-existing-openspec.md` first when required,
so refresh uses the correct artifact contracts.

## Detect Change Set

Prefer deterministic change evidence:

- Git diff and changed paths since the recorded repository revision
- content hashes for manifest records
- added, moved, renamed, and deleted paths
- build/workspace manifest changes
- contract/schema/migration changes
- command registration, wrapper, build-task, environment, and config changes
- changed tests and entry points

If no reliable baseline exists, rebuild the manifest and compare records. Do
not assume unchanged paths are stale merely because the session restarted.

## Calculate Impact

Use reverse traceability to map changed artifacts to:

- modules and deployable units
- capabilities and requirements
- contracts, data entities, events, and jobs
- tests
- commands, parameters, environment/config mappings, and help evidence
- reverse-dependent capabilities
- cross-cutting and architecture records

Create targeted refresh work items for changed and dependency-affected areas.
Use the same worker roles and write-isolation rules as full bootstrap.

## Parallel Refresh

Dispatch independent changed shards in parallel when native subagents exist.
Only the coordinator updates canonical specs/indexes. Preserve confirmed
requirements until new evidence proves a change; record contradictions rather
than silently deleting intent.

All refresh, synthesis, and audit dispatch obeys `Global HTTP 429 Queue` from
`00-global-rules.md`. One explicit 429 stops new parallel slots and moves all
waiting/retry requests into the same serial FIFO queue.

## Validate

After synthesis:

- refresh evidence hashes and repository revision
- refresh affected command-index records and invalidate stale exact-answer
  evidence
- validate specs and bidirectional traceability
- invalidate stale task context packets
- run `14-audit-sdd-coverage.md` when topology, public contracts, data model,
  build structure, or cross-cutting behavior changed materially

Small local diffs need a targeted orphan check, not necessarily a complete
audit. A major structural change requires the full audit.

If `openspec/history/git-tickets/registry.json` exists, compare each stored refs
fingerprint with current local Git refs using
`python3 <prompt-pack>/scripts/git_ticket_history.py status --repo .`. Re-run
deterministic scans only for stale prefixes. Preserve completed
ticket analyses whose signatures did not change and queue only new/stale
tickets; do not deep-analyze all historical tickets as part of normal SDD
refresh.

## Status

- Preserve `READY` only when all affected coverage gates pass.
- Use `READY_WITH_DECLARED_GAPS` for explicit analyzed gaps.
- Use `IN_PROGRESS` while refresh work remains.

## Output

```text
# OpenSpec Context Refreshed

Changed artifacts: <count>
Affected capabilities: <count>
Specs updated: <count>
Traceability records updated: <count>
Context packets invalidated: <count>
Validation: <result>
Coverage status: <status>
Declared gaps: <count>
```
