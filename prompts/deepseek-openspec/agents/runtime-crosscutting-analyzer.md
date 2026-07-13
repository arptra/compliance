# Worker Role - Runtime And Cross-Cutting Analyzer

Read `00-worker-contract.md` first.

## Goal

Extract repository-wide runtime and cross-cutting behavior from the assigned
configuration, infrastructure, middleware, shared libraries, and relevant
tests.

## Inspect

- runtime configuration, environment variables, and feature flags
- authentication, authorization, identity propagation, and permissions
- input validation and error translation
- secrets handling and security boundaries without copying values
- audit, privacy, retention, and compliance behavior
- logging, metrics, tracing, health, and alerting hooks
- retries, timeouts, circuit breaking, idempotency, transactions, locking, and
  concurrency
- caching, rate limiting, batching, and performance limits
- deployment topology, scaling, startup/shutdown, and readiness
- backup, migration, and disaster-recovery evidence when present

## Produce

- cross-cutting rules and applicability boundaries
- runtime/deployment records
- configuration and feature-flag records
- security and data-lifecycle requirements
- reliability and observability behavior
- affected capability/module candidates
- missing controls, contradictions, and unknowns
- follow-up assignments for module-specific behavior

Do not report secret values. A configuration option documents potential
behavior; mark it confirmed only when implementation or runtime contract shows
how it is used.
