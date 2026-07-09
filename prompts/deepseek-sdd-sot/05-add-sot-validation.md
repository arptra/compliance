# Prompt 05 - Add SoT Validation

Create lightweight validation scripts to prevent Source-of-Truth drift.

Do not change production business logic.

Create:

- `scripts/sot/README.md`
- `scripts/sot/check-traceability.py`
- `scripts/sot/check-spec-format.py`
- `scripts/sot/check-req-id-in-changes.py`

If this is a Maven project, suggest how to run the scripts from CI.
If this is a Gradle project, suggest how to run the scripts from CI.
If CI files exist, do not modify them yet unless the integration is obvious and
safe. Instead document the suggested CI step in `scripts/sot/README.md`.

Validation goals:

1. Every spec folder must contain:
   - `01-requirements.md`
   - `03-design.md`
   - `04-tasks.md`
   - `05-traceability.yml`
2. Every requirement must have an ID matching:
   - `REQ-[A-Z0-9]+-[0-9]+`
3. Every task must reference a requirement ID.
4. Every traceability file must map:
   - requirement
   - acceptance criteria
   - tests or `MISSING_TEST`
   - code or `UNKNOWN`
5. Changed production Java files should reference a requirement ID through:
   - branch name
   - commit message
   - PR/MR title
   - traceability file
6. The scripts must be simple, readable, and safe.

Do not introduce heavy dependencies.

After editing, output:

# SoT Validation Added

## Files Created

## How To Run

## What Is Checked

## Current Failures

## Recommended CI Integration
