# Refresh OpenSpec Context Incrementally

Use after repository changes, stale evidence, a major merge, or an explicit
user request. Read `00-global-rules.md` first. Do not change production code.

## Detect Change Set

Prefer deterministic change evidence:

- Git diff and changed paths since the recorded repository revision
- content hashes for manifest records
- added, moved, renamed, and deleted paths
- build/workspace manifest changes
- contract/schema/migration changes
- changed tests and entry points

If no reliable baseline exists, rebuild the manifest and compare records. Do
not assume unchanged paths are stale merely because the session restarted.

## Calculate Impact

Use reverse traceability to map changed artifacts to:

- modules and deployable units
- capabilities and requirements
- contracts, data entities, events, and jobs
- tests
- reverse-dependent capabilities
- cross-cutting and architecture records

Create targeted refresh work items for changed and dependency-affected areas.
Use the same worker roles and write-isolation rules as full bootstrap.

## Parallel Refresh

Dispatch independent changed shards in parallel when native subagents exist.
Only the coordinator updates canonical specs/indexes. Preserve confirmed
requirements until new evidence proves a change; record contradictions rather
than silently deleting intent.

## Validate

After synthesis:

- refresh evidence hashes and repository revision
- validate specs and bidirectional traceability
- invalidate stale task context packets
- run `14-audit-sdd-coverage.md` when topology, public contracts, data model,
  build structure, or cross-cutting behavior changed materially

Small local diffs need a targeted orphan check, not necessarily a complete
audit. A major structural change requires the full audit.

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
