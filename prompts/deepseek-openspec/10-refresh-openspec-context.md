# Refresh OpenSpec Project Context

Use this prompt when the repository changed significantly or OpenSpec context is
stale.

Do not rewrite all specs by default.

## Refresh Targets

- `openspec/project.md`
- affected `openspec/specs/<capability>/spec.md`
- `openspec/error-kb/index.yml` only if entries changed

## Rules

- Read Gradle/build markers first.
- Read only relevant source files.
- Preserve existing confirmed requirements.
- Mark inferred behavior as `INFERRED_FROM_CODE`.
- Mark missing intent as `UNKNOWN`.
- Do not change production code.

## Output

```text
# OpenSpec Context Refreshed

## Files Updated

## Evidence Used

## Specs Affected

## Unknowns

## Recommended Next Action
```
