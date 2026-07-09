# Prompt 02 - Create SoT Foundation

Now create the initial Spec-Driven Development and Source-of-Truth structure for
this repository.

Do not change production code.

Before editing, briefly re-check the repository structure. Do not rely only on chat
history.

Create or update the following files and folders where appropriate:

- `AGENTS.md`
- `.ai/context-map.yml`
- `.ai/context-packs/README.md`
- `.ai/skills/requirements-analyst/SKILL.md`
- `.ai/skills/java-architect/SKILL.md`
- `.ai/skills/java-implementer/SKILL.md`
- `.ai/skills/test-engineer/SKILL.md`
- `.ai/skills/traceability-guard/SKILL.md`
- `.ai/skills/reviewer/SKILL.md`
- `docs/sot/README.md`
- `docs/sot/00-constitution.md`
- `docs/sot/glossary.md`
- `docs/sot/open-questions.md`
- `docs/architecture/README.md`
- `docs/architecture/arc42.md`
- `docs/architecture/c4/README.md`
- `docs/adr/README.md`
- `specs/README.md`
- `specs/_template/00-intake.md`
- `specs/_template/01-requirements.md`
- `specs/_template/02-acceptance.feature`
- `specs/_template/03-design.md`
- `specs/_template/04-tasks.md`
- `specs/_template/05-traceability.yml`
- `specs/_template/06-test-plan.md`
- `specs/_template/07-changelog.md`
- `.github/pull_request_template.md` if GitHub is used
- `.gitlab/merge_request_templates/sdd-sot.md` if GitLab is used

Requirements for generated content:

- `AGENTS.md` must be short and practical.
- `AGENTS.md` must explain that source of truth is in repository files, not chat
  history.
- `AGENTS.md` must instruct future agents to read only relevant context.
- `AGENTS.md` must forbid broad refactoring without explicit task scope.
- `AGENTS.md` must require every code change to reference a requirement ID.
- `docs/sot/00-constitution.md` must define Java architecture rules:
  - no business logic in controllers
  - domain must not depend on Spring, JPA, HTTP, Kafka, or adapters
  - constructor injection only
  - stable error codes
  - OpenAPI update for public API changes
  - tests required for business rules
- `specs/_template/*` must be ready to copy for every new feature.
- All templates must be concise and machine-readable.
- If repository-specific facts are inferred from code, mark them as
  `INFERRED_FROM_CODE`.
- If a repository fact is unknown, write `UNKNOWN` and add it to
  `docs/sot/open-questions.md`.

After creating files, output:

# Created SoT Foundation

## Files Created

## Files Updated

## Important Notes

## Open Questions

## Recommended Next Step
