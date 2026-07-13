# Build OpenSpec Project And Architecture Context

Use during full bootstrap synthesis or targeted refresh. Read
`00-global-rules.md` first. Do not change production code.

## Inputs

Read the relevant completed worker findings and manifest/index shards. Return
work to the queue if evidence coverage for a section is insufficient. Do not
replace missing findings with assumptions.

## Update `openspec/project.md`

Maintain a concise repository-level map:

- explicit project purpose and product boundaries
- languages, frameworks, build systems, and toolchains
- module/service/application responsibilities
- repository conventions and ownership evidence when present
- build, test, validation, and local runtime commands
- public contract and schema locations
- links to system architecture, capability, coverage, and unknown indexes
- bootstrap freshness and status

This file is an entry point, not a dump of every module.

## Update Architecture Documents

### `architecture/system-context.md`

- systems, applications, services, libraries, and external actors
- dependency direction and major data flows
- entry points and integration boundaries
- deployment-unit boundaries when evidenced

### `architecture/runtime-and-deployment.md`

- processes and runtime topology
- environment and configuration model
- deployment manifests and infrastructure evidence
- persistence, messaging, scheduled/background execution
- operational dependencies and failure boundaries

### `architecture/cross-cutting-concerns.md`

- authentication and authorization
- security and secret handling
- validation and error semantics
- audit and compliance behavior
- observability
- retries, idempotency, transactions, concurrency, and resilience
- data lifecycle and privacy
- performance mechanisms and limits when evidenced

### `architecture/decisions.md`

- existing ADRs and design records
- architectural constraints evidenced by build, contracts, or code
- status, scope, supersession, and source links when known
- `UNKNOWN` rationale where behavior exists but the original decision cannot be
  recovered

Mark non-applicable sections explicitly. Link every material statement to
worker evidence or repository paths. Record contradictions and unknowns in the
canonical indexes.

## Glossary

Update `openspec/glossary.md` with domain terms, aliases, ambiguous terms, and
the repository evidence that established each meaning. Do not invent business
definitions.

## Output

```text
# Project Context Synthesized

Documents updated: <count>
Modules represented: <count>
Cross-cutting areas covered: <count>/<applicable count>
Unknowns added: <count>
Contradictions added: <count>
```
