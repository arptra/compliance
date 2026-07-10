# OpenSpec Error Memory

Use this prompt to add or maintain a git-shareable repository memory of
recurring errors and verified fixes.

The store lives under OpenSpec:

```text
openspec/error-kb/
  README.md
  index.yml
  entries/
    ERR-YYYY-NNN-short-slug.md
```

## Rules

- Before debugging any failing command, test, build, toolchain, dependency, or
  runtime error, check `openspec/error-kb/index.yml` if it exists.
- Do not read all entries. Read only entries that match the normalized failure
  signature.
- If a verified solution matches the current failure, apply it and cite the
  entry ID.
- If no solution matches, debug normally.
- After a new solution is verified, create or update an entry.
- Update `openspec/error-kb/index.yml` whenever entries change.
- Do not store secrets, tokens, full production logs, customer data, private
  payloads, usernames, or machine-specific absolute paths.
- Sanitize logs before writing them.
- Do not commit or push unless the user explicitly asks.

## Initialize The Store

If `openspec/error-kb/` does not exist, ask:

```text
OpenSpec error memory does not exist.
Do you want me to create `openspec/error-kb/` now? Reply YES or NO.
```

If the user replies `YES`, create:

- `openspec/error-kb/README.md`
- `openspec/error-kb/index.yml`
- `openspec/error-kb/entries/.gitkeep`

## Failure Fingerprint

Normalize from stable facts:

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
- usernames
- secrets
- request IDs
- temporary file names

## `index.yml` Format

```yaml
version: 1
entries:
  - id: ERR-2026-001-gradle-java-toolchain
    title: Gradle test fails because Java toolchain is unavailable
    fingerprint: gradle:test:toolchain-unavailable
    file: openspec/error-kb/entries/ERR-2026-001-gradle-java-toolchain.md
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

```text
# OpenSpec Error Memory Report

## Failure Fingerprint

## Matching Entries Checked

## Reused Solution

## New Or Updated Entry

## Verification Command

## Result
```
