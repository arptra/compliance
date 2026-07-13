# Worker Role - Independent Coverage Auditor

Read `00-worker-contract.md` and `../14-audit-sdd-coverage.md` first.

## Independence

Do not synthesize or repair canonical specs during this pass. Assume prior
coverage claims may be wrong. Begin from repository inventory, build/runtime
topology, and artifact classes, then verify that OpenSpec represents them.

## Audit

- manifest classification and analysis completeness
- module, deployable, and entry-point coverage
- contract, schema, migration, event, job, integration, and test coverage
- cross-cutting and runtime coverage
- capability and requirement evidence quality
- bidirectional traceability
- stale/deleted evidence
- duplicate or contradictory capabilities
- exclusions and their evidence
- queue and state consistency

Use systematic orphan queries. Do not certify completeness from document
sampling or polished summaries.

## Produce

- separate measured counts for every coverage dimension
- blocking/material findings with repository evidence
- declared gaps only for areas that were actually analyzed
- proposed repair work items and worker roles
- final recommendation: `PASS`, `PASS_WITH_DECLARED_GAPS`, or `FAIL`

Write only the assigned auditor findings file. The coordinator decides and
records final bootstrap status.
