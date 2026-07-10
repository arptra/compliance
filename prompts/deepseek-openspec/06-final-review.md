# Final OpenSpec Foundation Review

Use this prompt after OpenSpec initialization or major prompt updates.

Do not make new changes unless they are small fixes to the artifacts being
reviewed.

## Review Checklist

1. Does `openspec/project.md` exist and describe the repo accurately?
2. Do `openspec/specs/` and `openspec/changes/` exist?
3. Does `AGENTS.md` require OpenSpec before code changes?
4. Does the workflow avoid reading the whole repo?
5. Are unknown business facts marked `UNKNOWN`?
6. Are inferred facts marked `INFERRED_FROM_CODE`?
7. Is validation guidance present?
8. Is `openspec/error-kb/` present?
9. Are there accidental production code changes?

## Output

```text
# OpenSpec Foundation Review

## Pass / Fail

## Issues

## Recommended Fixes

## Ready For Feature Work
YES/NO
```
