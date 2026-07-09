# Prompt 04 - Create Current-State Specs

Create current-state specifications for the existing major business capabilities
discovered in the repository.

Do not change production code.

Create specs under:

`specs/current-state/`

For each major capability, create a folder:

`specs/current-state/<capability-name>/`

Inside each folder create:

- `00-intake.md`
- `01-requirements.md`
- `03-design.md`
- `05-traceability.yml`
- `06-test-plan.md`

Important:

1. These are current-state specs, not approved future requirements.
2. Every requirement must be marked as `INFERRED_FROM_CODE`.
3. Do not invent business goals.
4. If the code shows behavior but the business reason is unclear, write
   `UNKNOWN`.
5. Link requirements to actual classes, tests, contracts, migrations, or configs
   where possible.
6. If there are no tests for a behavior, write `MISSING_TEST`.
7. If public API exists but OpenAPI is missing, write `MISSING_CONTRACT`.
8. If database behavior exists, link to migrations, entities, repositories, or
   config evidence.
9. Keep each spec small.
10. Create no more than 5 current-state specs in this step. Choose the most
    important capabilities.

Use this requirement style:

```text
WHEN <condition from code>
THE SYSTEM SHALL <observable behavior>
```

After editing, output:

# Current-State Specs Created

## Specs Created

## Capabilities Covered

## Missing Tests

## Missing Contracts

## Business Unknowns

## Recommended Next Step
