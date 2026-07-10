# Create OpenSpec Current-State Specs

Use this prompt to document existing behavior as OpenSpec specs.

Do not change production code.

## Goal

Create or update current behavior specs under:

```text
openspec/specs/<capability>/spec.md
```

## Rules

- Document only behavior evidenced by code, tests, contracts, or docs.
- Mark inferred behavior as `INFERRED_FROM_CODE`.
- Put missing business intent in `UNKNOWN` notes.
- Do not create more than 5 capability specs in one run unless the user asks.

## Spec Shape

```markdown
# <Capability> Specification

## Requirements

### Requirement: <current behavior>

#### Scenario: <scenario name>

- GIVEN ...
- WHEN ...
- THEN ...
```

## Output

```text
# Current OpenSpec Specs Updated

## Specs Created

## Specs Updated

## Evidence Used

## Unknowns
```
