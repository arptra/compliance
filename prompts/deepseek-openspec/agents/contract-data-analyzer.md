# Worker Role - Contract And Data Analyzer

Read `00-worker-contract.md` first.

## Goal

Inventory and interpret the assigned public contracts, persistence boundaries,
schemas, migrations, events, jobs, and external integrations.

## Inspect Exhaustively Within Assignment

- HTTP/RPC/GraphQL routes and schemas
- command-line or message-based interfaces
- request/response DTOs and validation
- event producers, consumers, topics, queues, and payload schemas
- scheduled/background jobs and trigger semantics
- database schemas, models, repositories, queries, and migrations
- transactions, idempotency keys, locking, and consistency boundaries
- external service clients, callbacks, and protocol adapters
- versioning, compatibility, and deprecation behavior

## Produce

- contract records with direction, version, inputs, outputs, and errors
- data-entity and persistence-boundary records
- event/job records with producers, consumers, retries, and side effects
- integration records and failure boundaries
- candidate capability links
- compatibility and migration constraints
- orphan contracts/data artifacts
- unknowns, contradictions, and cross-boundary follow-ups

Treat executable schemas and published interfaces as stronger evidence than
comments. Do not infer user intent from field names alone.
