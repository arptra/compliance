# Repository Instructions

This file is the small project context for GIGACODE. Do not load the whole repository or all OpenSpec files at startup.

## Start Or Resume

1. Run `git status --short --branch`.
2. Read `openspec/README.md` only.
3. Run `openspec list --json` when the CLI is available.
4. If one active change exists, read its status and only the artifacts returned by OpenSpec.
5. Otherwise wait for the user's normal-language request. Do not reread every spec or archive.

## Source Of Truth

- Current behavior: `openspec/specs/**/spec.md`.
- Proposed behavior: `openspec/changes/<change>/`.
- Historical behavior: `openspec/changes/archive/`; never use it as current truth.
- Implementation evidence: current code and tests. When code contradicts a current spec, report the conflict before editing.
- Navigation and implementation ownership: `openspec/README.md`; it is an index, not a behavioral spec.

## No Guessing

Never invent command parameters, API fields, config keys, paths, defaults, status values, or error behavior.

1. Find the relevant capability in `openspec/README.md`.
2. Read only its current spec.
3. Search the listed code with `rg` and inspect schemas/tests.
4. Answer with file and symbol evidence. If evidence is absent, say `not found` and name what was checked.

## Development

- For a behavior change, use `/opsx-propose <one clear intent>`, review artifacts, then `/opsx-apply`.
- After tests pass, use `/opsx-sync` and `/opsx-archive`.
- For cross-layer work, delegate frontend and backend discovery to the matching read-only subagents in parallel.
- The main agent alone edits files. Subagents return facts, paths, symbols, risks, and suggested tests.
- Preserve the active runtime boundary: `create_app()` mounts `health`, `auth`, `gigachat`, and `records`; the React router exposes GigaChat Lab, lake, background tasks, and profile.

## Verification

- Backend: `PYTHONPATH=src pytest -q <focused tests>`.
- Frontend: `npm run build --prefix apps/dashboard`.
- OpenSpec: `openspec validate --specs` and `openspec validate --all`.
