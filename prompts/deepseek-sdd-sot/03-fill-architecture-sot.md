# Prompt 03 - Fill Architecture SoT

Now fill the architecture documentation based on the actual repository.

Do not change production code.

Update:

- `docs/architecture/arc42.md`
- `docs/architecture/c4/README.md`
- `docs/sot/glossary.md`
- `docs/sot/open-questions.md`
- `docs/adr/README.md`

Create initial ADRs if needed:

- `docs/adr/0001-current-architecture-baseline.md`
- `docs/adr/0002-source-of-truth-in-repository.md`

Use only evidence from the repository.

In `docs/architecture/arc42.md`, describe:

1. Introduction and goals
2. Constraints
3. Context and scope
4. Solution strategy
5. Building block view
6. Runtime view
7. Deployment view if Docker, Kubernetes, or CI files exist
8. Crosscutting concepts
9. Architecture decisions
10. Quality requirements
11. Risks and technical debt
12. Glossary

Rules:

- Mark uncertain information as `UNKNOWN`.
- Mark inferred information as `INFERRED_FROM_CODE`.
- Do not invent business intent.
- Keep every section concise.
- Prefer tables where useful.
- Link to actual packages, modules, and files where possible.

After editing, output:

# Architecture SoT Update

## Updated Files

## Architecture Summary

## Main Modules

## Main Risks

## Unknowns

## Recommended Next Step
