# Audit SDD Coverage Against The Repository

Use after full bootstrap synthesis, after a major refresh, or when coverage is
questioned. This is an adversarial, repository-to-SDD audit. It must not be
performed solely by the same reasoning pass that synthesized the specs.

Read `00-global-rules.md` and `agents/coverage-auditor.md` first. Do not change
production code.

## Audit Direction

Start from the repository manifest and runtime/build topology, then ask whether
each significant artifact is represented in OpenSpec. Do not start only from
existing specs; that cannot reveal omitted capabilities.

## Coverage Dimensions

Audit and report separate coverage for:

- repository path classification
- in-scope artifact analysis
- modules, services, applications, and entry points
- public APIs and commands
- exact command parameters, defaults, aliases, config/env mappings, and evidence
- events, consumers, producers, schedulers, and background jobs
- data schemas, persistence boundaries, and migrations
- external integrations
- test-derived observable behavior
- security, authorization, audit, and privacy behavior
- reliability, transactions, idempotency, concurrency, and error semantics
- runtime, configuration, observability, and deployment
- capability-to-code links
- capability-to-contract links where contracts exist
- command-to-capability, command-to-artifact, and command-to-test links
- capability-to-test links where tests exist
- reverse artifact-to-capability links

Do not collapse these dimensions into a single vanity percentage. A summary
percentage may be shown only alongside all component counts.

## Systematic Orphan Checks

Find:

- in-scope files with no completed analysis
- modules or deployable units absent from the system map
- routes/controllers/handlers/commands with no capability
- executable entry points, build tasks, flags, or dynamic commands missing from
  the command index
- command attributes whose exact evidence is missing or stale
- schemas, migrations, topics, jobs, or integrations with no capability
- significant test suites with no linked requirement
- capability requirements with missing or stale evidence
- traceability edges pointing to deleted artifacts
- duplicate capabilities with overlapping evidence
- cross-cutting behavior mentioned in code/tests but absent from architecture
- exclusions without a recorded reason or match count

Use deterministic searches and parsers where possible. Sampling may discover
issues but cannot prove complete coverage.

## Audit Findings

For each finding record:

- stable finding ID
- severity: `blocking`, `material`, or `declared-gap`
- coverage dimension
- manifest/artifact evidence
- missing or conflicting SDD record
- recommended worker role
- proposed work-queue item

Write the auditor's raw result to its unique bootstrap finding file. The
coordinator updates canonical indexes and queue.

## Pass Rules

Return:

- `PASS` only when every applicable completion gate passes with no gaps.
- `PASS_WITH_DECLARED_GAPS` only when all in-scope areas were processed and all
  remaining issues are explicit unknowns, contradictions, or absent assurance
  artifacts.
- `FAIL` when any inventory, analysis, synthesis, traceability, validation, or
  audit work remains incomplete.

Missing work is never a declared gap. A gap is allowed only after the relevant
area was actually examined.

## Repair Loop

Convert blocking and material findings into queue items. Resume the relevant
worker, synthesis, and validation phases. Re-run the audit after repairs.

Update `openspec/index/coverage.yml` only after validating its counts against
manifest and traceability records.

## Output

```text
# SDD Coverage Audit

Result: <PASS|PASS_WITH_DECLARED_GAPS|FAIL>
Paths classified: <count>/<count>
In-scope artifacts analyzed: <count>/<count>
Modules/deployables mapped: <count>/<count>
Capabilities documented: <count>
Orphan artifacts: <count>
Blocking/material findings: <count>
Declared gaps: <count>
Command surfaces verified: <count>/<count>
Repair work items created: <count>
```
