# Global Rules - OpenSpec For Java/Gradle Repos

You are working inside a Java/Gradle repository.

OpenSpec is the source of truth.

## Non-Negotiable Rules

- Do not read the whole repository.
- Do not edit production Java code without an approved OpenSpec change.
- Do not invent requirements.
- Mark inferred behavior as `INFERRED_FROM_CODE`.
- Mark missing business intent as `UNKNOWN`.
- Keep user interaction simple: numbered menus, short questions, one decision at
  a time.
- Prefer `./gradlew` on Linux.
- If `./gradlew` is missing, ask before using system `gradle`.
- Do not commit, push, delete, or run destructive commands unless the user asks.
- Before debugging repeated failures, check `openspec/error-kb/`.
- After fixing a new recurring failure, write a reusable sanitized entry into
  `openspec/error-kb/`.

## OpenSpec Layout

Use:

```text
openspec/
  project.md
  specs/
  changes/
  error-kb/
```

New work must be modeled as:

```text
openspec/changes/<change-id>/
  proposal.md
  tasks.md
  design.md
  specs/<capability>/spec.md
```

## Validation

If OpenSpec CLI is available:

```bash
openspec validate <change-id> --strict
```

If unavailable, report that only file-level checks were done.

## Implementation Gate

Implementation can start only when:

1. an OpenSpec change exists,
2. `proposal.md`, `tasks.md`, and spec deltas exist,
3. validation has passed or CLI absence is reported,
4. the user explicitly approves implementation.
