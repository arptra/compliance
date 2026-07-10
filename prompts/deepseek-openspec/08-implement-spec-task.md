# Implement One OpenSpec Task

Use this prompt after an OpenSpec change exists and the user approves
implementation.

## Required Input

OpenSpec change:

```text
openspec/changes/<change-id>/
```

If no change is provided, list active changes and ask the user to choose one.

## Read First

- `openspec/project.md`
- selected `proposal.md`
- selected `tasks.md`
- selected `design.md` if present
- selected change spec deltas
- current specs for affected capabilities
- `openspec/error-kb/index.yml` if present

Do not scan unrelated source files.

## Implementation Workflow

1. Pick one unchecked task.
2. Explain impacted files before editing.
3. Add or update tests first where practical.
4. Implement the smallest safe code change.
5. Run relevant Gradle tests.
6. If tests fail, check `openspec/error-kb/` before debugging.
7. Fix task-related failures.
8. Record new reusable failure fixes in `openspec/error-kb/`.
9. Update `tasks.md` checkboxes.
10. Run `openspec validate <change-id> --strict` if CLI is available.

Gradle preference:

```bash
./gradlew :module:test
```

Fallback:

```bash
./gradlew test
```

## Output

```text
# OpenSpec Implementation Report

## Change ID

## Task Implemented

## Files Changed

## Tests Added/Updated

## Gradle Commands Run

## OpenSpec Validation

## Error Memory Used Or Updated

## Remaining Tasks
```
