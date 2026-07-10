# Create A New OpenSpec Change

Use this prompt when the user describes a new feature, behavior change, removal,
or refactor.

Do not change production code in this step.

## Inputs

Ask for the feature/change in one or two sentences if not provided.

## Read

- `openspec/project.md`
- `openspec/specs/` for affected capabilities only
- active `openspec/changes/` names
- Gradle/build files only if needed for scope

Do not scan the whole repo.

## Create

Choose a unique verb-led change ID:

- `add-...`
- `change-...`
- `remove-...`
- `refactor-...`

Create:

```text
openspec/changes/<change-id>/
  proposal.md
  tasks.md
  specs/<capability>/spec.md
```

Create `design.md` when the change is broad, risky, cross-module, API-facing,
data-model-changing, migration-related, auth-related, performance-sensitive, or
ambiguous.

## `proposal.md`

Include:

- problem
- proposed change
- user-visible outcome
- non-goals
- risks
- open questions

## `tasks.md`

Use checkboxes:

```markdown
## Tasks

- [ ] 1. Update tests
- [ ] 2. Implement behavior
- [ ] 3. Run Gradle tests
- [ ] 4. Validate OpenSpec change
```

## Spec Delta

Use:

```markdown
## ADDED Requirements

### Requirement: <name>

#### Scenario: <scenario>

- GIVEN ...
- WHEN ...
- THEN ...
```

Use `MODIFIED`, `REMOVED`, or `RENAMED` when changing existing behavior.

## Validate

If OpenSpec CLI is available:

```bash
openspec validate <change-id> --strict
```

If unavailable, do file-level checks and report that CLI validation was not run.

## Output

```text
# OpenSpec Change Created

## Change ID

## Files Created

## Validation

## Tasks

## Open Questions

## Ready For Implementation
YES/NO
```
