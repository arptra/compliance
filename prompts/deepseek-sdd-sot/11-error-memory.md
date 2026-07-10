# Error Memory Prompt

Use this prompt when you want DeepSeek CLI to add or maintain a git-shareable
repository memory of recurring errors and verified fixes.

You are working inside a Java/Gradle repository.

## Goal

Create and use a local error-memory store that helps future CLI sessions avoid
debugging the same failure from scratch.

The store must live in the repository so it can be shared through git:

```text
.ai/error-kb/
  README.md
  index.yml
  entries/
    ERR-YYYY-NNN-short-slug.md
```

## Rules

- Before debugging any failing command, test, build, toolchain, dependency, or
  runtime error, check `.ai/error-kb/index.yml` if it exists.
- Do not read all entries. Read only entries that match the normalized failure
  signature.
- If a verified solution matches the current failure, apply it and cite the
  entry ID in your report.
- If no solution matches, debug normally.
- After a new solution is verified, create or update an entry.
- Update `.ai/error-kb/index.yml` whenever entries change.
- Do not store secrets, tokens, full production logs, customer data, private
  payloads, usernames, or machine-specific absolute paths.
- Sanitize logs before writing them.
- Do not commit or push unless the user explicitly asks.

## Initialize The Store

If `.ai/error-kb/` does not exist, ask:

```text
This repository does not have an error-memory store.
Do you want me to create `.ai/error-kb/` now? Reply YES or NO.
```

If the user replies `YES`, create:

- `.ai/error-kb/README.md`
- `.ai/error-kb/index.yml`
- `.ai/error-kb/entries/.gitkeep`

## Failure Fingerprint

For each failure, normalize a fingerprint from stable facts:

- command, for example `./gradlew :service:test`
- Gradle task
- Java version
- Gradle version
- operating system family
- exception/error class
- stable message fragment
- top relevant project stack frame
- dependency or plugin name if relevant

Remove unstable facts:

- timestamps
- random IDs
- local absolute paths
- line numbers unless the line is essential
- usernames
- secrets
- request IDs
- temporary file names

## `index.yml` Format

Use this shape:

```yaml
version: 1
entries:
  - id: ERR-2026-001-gradle-java-toolchain
    title: Gradle test fails because Java toolchain is unavailable
    fingerprint: gradle:test:toolchain-unavailable
    file: .ai/error-kb/entries/ERR-2026-001-gradle-java-toolchain.md
    tags: [gradle, java, tests]
    gradle_task: test
    exception: null
    message_contains:
      - No matching toolchains found
    status: verified
    first_seen: 2026-07-10
    last_seen: 2026-07-10
    hits: 1
```

## Entry Format

Use this shape:

```markdown
# ERR-2026-001 - Gradle test fails because Java toolchain is unavailable

## Signature

- Command:
- Gradle task:
- Java:
- Gradle:
- Exception:
- Message contains:

## Environment

## Root Cause

## Solution

## Verification

## When Not To Use

## Evidence
```

## Required Report

After using or updating the store, report:

```text
# Error Memory Report

## Failure Fingerprint

## Matching Entries Checked

## Reused Solution

## New Or Updated Entry

## Verification Command

## Result
```
