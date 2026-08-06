# Proposal: Repository SDD Prompt Pack

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Provide a prompt-driven workflow for repository assessment, SSOT bootstrap, grounded answers, session resume, Git ticket reconstruction, and prompt migrations for local DeepSeek agents.

## Historical Evidence

- `1980d9b`, `2cf98e2`: initial prompt pack and usage docs.
- `7e9f0a6`, `306771d`, `0e8cd42`: resume, start, and interactive menu workflows.
- `ce53956`, `96b9e86`, `c9c4157`: updated artifacts, full bootstrap, grounded answers, and migrations.
- `e6d1f5d`, `f2ac15e`: Git ticket history workflow and 2.1 capability docs.

## Outcome

The repository contains reusable prompts under `prompts/deepseek-openspec/`. They remain a separate tooling pack; current repository behavior now lives in the official OpenSpec 1.x `openspec/specs/` layout.
