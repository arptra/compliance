# Add OpenSpec Validation Guidance

Use this prompt to add lightweight validation guidance for OpenSpec workflows.

Do not change production business logic.

## Preferred Validation

If OpenSpec CLI is available:

```bash
openspec validate <change-id> --strict
openspec validate --all --strict
```

If CLI is not available, document file-level checks in `openspec/changes/README.md`.

## File-Level Checks

Each change should contain:

- `proposal.md`
- `tasks.md`
- at least one `specs/<capability>/spec.md`
- `design.md` for broad, risky, cross-module, data model, API, auth, migration,
  or performance-sensitive changes

Each spec delta should include:

- `ADDED`, `MODIFIED`, `REMOVED`, or `RENAMED` requirement section
- at least one `### Requirement: ...`
- at least one `#### Scenario: ...`

## Output

```text
# OpenSpec Validation Guidance Updated

## Files Updated

## CLI Available

## Manual Checks Documented
```
