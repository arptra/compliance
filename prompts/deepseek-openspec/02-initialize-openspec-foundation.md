# Initialize OpenSpec Foundation

Use this prompt only after repository assessment and user approval.

Do not change production Java code.

## Create Or Update

- `openspec/project.md`
- `openspec/specs/README.md`
- `openspec/changes/README.md`
- `openspec/error-kb/README.md`
- `openspec/error-kb/index.yml`
- `openspec/error-kb/entries/.gitkeep`
- `AGENTS.md` with OpenSpec workflow rules

## `openspec/project.md` Must Include

- project purpose as inferred from repository evidence
- Gradle and Java facts
- module/package map
- test commands
- coding conventions visible in the repo
- OpenSpec workflow rules
- `UNKNOWN` section for missing business intent

## `AGENTS.md` Must Require

- OpenSpec change before production code edits
- `openspec validate <change-id> --strict` when CLI is available
- Gradle unit tests after implementation
- error-memory lookup before repeated debugging

## Output

```text
# OpenSpec Foundation Initialized

## Files Created

## Files Updated

## Project Facts

## Unknowns

## Next Menu
```
