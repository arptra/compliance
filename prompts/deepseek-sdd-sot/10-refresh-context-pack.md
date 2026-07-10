# Prompt 10 - Refresh One Context Pack

You are working in a large Java repository with an existing SDD/SoT foundation.

Refresh only one context pack.

Context pack to refresh:

```text
<CONTEXT_PACK_NAME>
```

Reason for refresh:

```text
<WHY THIS PACK NEEDS UPDATING>
```

Rules:

- Do not read the whole repository.
- Do not edit production code.
- Read `.ai/context-map.yml` first.
- Read the current `.ai/context-packs/<CONTEXT_PACK_NAME>.md` if it exists.
- Inspect only files that are relevant to this context pack.
- Do not inspect more than 30 files unless you first explain why.
- Mark inferred behavior as `INFERRED_FROM_CODE`.
- Mark unknown business intent as `UNKNOWN`.
- Keep the context pack short enough for future CLI sessions.

The refreshed context pack must include:

- purpose
- main packages
- entry points
- services
- repositories/entities
- APIs/events/jobs
- tests
- risks/unknowns
- files future agents should read first
- last refreshed date

Output:

# Context Pack Refreshed

## Context Pack

## Files Inspected

## What Changed

## Unknowns

## Recommended Next Step
