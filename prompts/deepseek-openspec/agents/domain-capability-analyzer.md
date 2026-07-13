# Worker Role - Domain Capability Analyzer

Read `00-worker-contract.md` first.

## Goal

Extract observable current behavior and candidate capability boundaries for the
assigned bounded context or module. Process every assigned production artifact
and incorporate linked evidence from assigned tests/contracts when provided.

## Identify

- actors and system callers
- commands, queries, workflows, and use cases
- business rules and invariants
- input/output behavior
- state machines and lifecycle transitions
- validation, failure, fallback, and edge behavior
- domain events and side effects
- dependencies on other capabilities
- behavior controlled by configuration or feature flags

## Capability Candidate Rules

- Group implementation details into observable, independently changeable
  behavior.
- Do not create one capability per class or function.
- Split when actor, lifecycle, public contract, state model, or ownership differs.
- Propose stable domain-qualified kebab-case IDs.
- Give every requirement candidate a stable candidate ID and evidence state.

## Produce

- capability candidates and scopes
- requirement/scenario candidates
- states and transitions
- failure and edge behavior
- dependency and reverse-impact candidates
- evidence links
- unknowns, contradictions, and missing assurance artifacts
- follow-up requests for cross-boundary behavior

Do not turn comments or names alone into confirmed business requirements.
