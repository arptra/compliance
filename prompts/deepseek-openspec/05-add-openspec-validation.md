# Validate OpenSpec Structure, Evidence, And Traceability

Use during bootstrap, refresh, change creation, and final review. Do not change
production business logic.

## OpenSpec CLI Validation

If available, use the relevant commands:

```bash
openspec validate <change-id> --strict
openspec validate --all --strict
```

If unavailable, report that CLI validation was not run. Never treat CLI absence
as evidence that the custom indexes are valid.

## Current-State Spec Checks

Every capability spec must have:

- unique capability ID
- domain and scope
- at least one requirement
- stable requirement IDs unique within the repository
- at least one scenario per requirement
- allowed evidence state per requirement
- at least one repository evidence reference per non-`UNKNOWN` requirement
- dependencies represented in the traceability index
- explicit unknowns and contradictions where applicable

## Index Integrity Checks

Validate:

- every referenced path exists or is marked deleted/stale
- every manifest shard is listed by `repository-manifest.yml`
- every in-scope file has a classification and analysis status
- every capability index entry has a matching spec
- every traceability node and edge references existing IDs
- every command token/parameter attribute has current evidence
- every command-to-capability and command-to-artifact link resolves
- reverse artifact-to-capability links can be resolved
- unknown and contradiction IDs are unique
- queue counts agree with actual queue items
- coverage counts agree with manifest and traceability data
- no secret values or private payloads were copied into artifacts
- `openspec/meta.yml` matches the applied pack version/fingerprint and contains
  no incomplete migration marked as applied

Use a structured parser when the environment provides one. Do not validate YAML
through ad hoc text matching alone.

## Change Checks

Every active change must contain:

- `proposal.md`
- `tasks.md`
- at least one `specs/<capability-id>/spec.md`
- `design.md` for broad, risky, cross-module, data-model, API, auth, migration,
  concurrency, reliability, or performance-sensitive changes
- task context packet reference or a documented reason it was unnecessary

Spec deltas must use OpenSpec delta headings and include requirement scenarios.

## Grounded-Answer Checks

For any persisted grounded-answer record verify:

- each material claim has evidence
- every cited path/symbol/excerpt exists and is fresh
- exact tokens occur literally in cited current evidence
- conflicts are not omitted
- verified status has a passing grounding-validator result
- `NOT_VERIFIED` contains no guessed answer disguised as a candidate

## Git Ticket History Checks

When `openspec/history/git-tickets/registry.json` exists, validate:

- registry, meta, index, queue, and ticket JSON parse structurally
- each indexed ticket file and analysis path resolves
- queue/index statuses and ticket signatures agree
- every completed analysis cites only commits associated with that ticket
- shared multi-ticket commits and ref-only records are explicitly labeled
- no full patches, secrets, personal data, or fabricated issue-tracker facts are
  persisted

An uninitialized Git ticket history is not an OpenSpec validation failure.

## Validation Result

Return one:

```text
PASS
PASS_WITH_DECLARED_GAPS
FAIL
```

Declared gaps must be explicit index records. Missing or silently skipped work
is a failure, not a declared gap.

## Output

```text
# OpenSpec Validation

Result: <result>
OpenSpec CLI: <available/unavailable>
Specs checked: <count>
Index records checked: <count>
Errors: <count>
Declared gaps: <count>
Required repairs: <short list>
```
