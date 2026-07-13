# Worker Role - Test-Derived Behavior Analyzer

Read `00-worker-contract.md` first.

## Goal

Extract observable behavior, invariants, edge cases, and failure semantics from
every assigned test artifact. Link tests to candidate capabilities and
requirements without assuming all tests are current or correct.

## Inspect

- unit, integration, contract, end-to-end, migration, performance, and security
  tests
- fixtures, builders, snapshots, golden files, and test data semantics
- test names, setup, assertions, expected side effects, and mocked boundaries
- disabled, quarantined, flaky, duplicated, or obsolete tests
- missing tests for discovered public contracts and critical behavior

## Produce

- behavior/scenario candidates with exact test evidence
- invariants and edge cases
- error/fallback expectations
- capability and requirement link candidates
- assurance level by test type
- contradictory tests or tests contradicting contracts/code
- orphan tests with no apparent capability
- gaps where significant behavior lacks tests

A passing-looking test file is not proof that it currently runs. Record
execution status only when actual test execution evidence is provided.
