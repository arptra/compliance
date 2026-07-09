# DeepSeek SDD / SoT Prompt Pack

This folder contains prompts for running a local model such as `deepseek-v4-flash-262k`
inside an existing Java repository to create a Spec-Driven Development (SDD) and
Source-of-Truth (SoT) foundation.

Use each file as one CLI prompt, in numeric order.
If your CLI session does not preserve context between runs, prepend
`00-global-rules.md` to every numbered prompt.

## Is This Enough?

Yes, this prompt pack is enough to create an initial SDD/SoT foundation for a Java
repository if it is executed in order and the model can inspect the repo.

It is not enough to create approved business requirements by itself. Existing
behavior can only be documented as `INFERRED_FROM_CODE`; missing intent must stay
`UNKNOWN` and go to `docs/sot/open-questions.md`.

## Recommended Order

1. `00-global-rules.md`
2. `01-assess-repository.md`
3. `02-create-sot-foundation.md`
4. `03-fill-architecture-sot.md`
5. `04-create-current-state-specs.md`
6. `05-add-sot-validation.md`
7. `06-final-review.md`

For future feature work use:

- `07-new-feature-spec.md`
- `08-implement-spec-task.md`

## Operating Rules

- Run prompts inside the target Java repo root.
- Do not ask the model to commit or push from these prompts.
- Review generated docs before using them as governance.
- Keep generated files small and maintainable.
- Treat chat history as temporary; repository files are the source of truth.

## Expected Result

After prompts 1-6, the Java repo should contain:

- agent guidance in `AGENTS.md`
- SoT docs under `docs/sot/`
- architecture docs under `docs/architecture/`
- ADR docs under `docs/adr/`
- feature spec templates under `specs/_template/`
- optional current-state specs under `specs/current-state/`
- lightweight SoT validation scripts under `scripts/sot/`
