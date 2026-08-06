# Proposal: Rule Pack Exchange And Matching

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Make Lab rule packs portable through Excel and align server/client matching with real source-field and keyword data.

## Historical Evidence

- `7d96975`: rule pack Excel import/export.
- `5347354`, `8f55e4d`, `194e253`: keyword matching iteration and literal spacing preservation.
- `4419b4a`: reliable profile saves.
- `e9860fb`: partial source-field matching.

## Outcome

Users could exchange rules with spreadsheets, preserve literal keywords, and evaluate rules even when only some configured source columns were present.
