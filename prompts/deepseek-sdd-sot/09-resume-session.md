# Prompt 09 - Resume SDD Session

You are starting a fresh CLI session inside a Java repository that already has an
SDD/SoT foundation.

Do not read the whole repository.
Do not inspect more than 20 files before producing a session brief.
Do not edit production code in this step.

Your task is to restore the smallest useful working context.

First read only:

- `AGENTS.md`
- `.ai/context-map.yml`
- `.ai/context-packs/README.md`
- `docs/sot/README.md`
- `docs/sot/00-constitution.md`
- `docs/sot/open-questions.md`
- `specs/README.md`

Then inspect:

- existing `specs/` folders
- the active spec folder if I provide one
- the relevant `.ai/context-packs/<context>.md` files for the active work

If I did not provide an active spec folder, ask me which feature/spec/task we are
continuing. Do not scan the full codebase to guess.

Rules:

- Use `.ai/context-map.yml` to decide which context packs are relevant.
- Read source code only after identifying a specific active feature or task.
- If more than 20 files are needed, list the extra files and explain why before
  reading them.
- Prefer context packs over raw source exploration.
- If context packs are stale or missing, propose a small context-pack refresh
  before implementation.
- Keep the answer short and operational.

Output:

# SDD Session Restored

## Loaded Files

## Active Spec / Task

## Relevant Context Packs

## What I Understand

## Missing Context

## Next Safe Action
