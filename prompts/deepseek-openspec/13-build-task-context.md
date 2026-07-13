# Build A Task Or Repository-Question Context Packet

Use before creating, implementing, reviewing, or debugging a non-trivial
OpenSpec change, and before answering a factual repository question. The
purpose is to give a limited-context model a sufficient, auditable projection
without loading the full repository.

Read `00-global-rules.md` first. This step is read-only with respect to
production code and current-state specs. It may create one small context-packet
manifest.

## Mode

The caller must set one:

- `CHANGE_TASK`: build a persistent task/change packet.
- `REPOSITORY_QUESTION`: retrieve exact claim evidence for
  `15-answer-repository-question.md`.

## Input

- the user's request or selected active task
- repository bootstrap status
- `openspec/index/capabilities.yml`
- `openspec/index/commands.yml` and relevant command shards
- relevant traceability and file-index shards
- architecture and glossary entries selected by links
- active OpenSpec change files when continuing work

If status is `PARTIAL_CONTEXT`, include that warning in the packet and change
proposal. If full bootstrap is still running, prefer finishing the relevant
bootstrap work items before implementation.

## Retrieval Order

Use deterministic retrieval first:

1. exact capability, requirement, contract, module, symbol, and path matches
2. capability dependency and reverse-dependency graph
3. artifact-to-capability traceability
4. affected public contracts, schemas, migrations, events, and data models
5. linked tests and error-memory fingerprints
6. exact command, build-task, environment, and config records when applicable
7. applicable cross-cutting requirements and architecture decisions
8. lexical/full-text search for missing references
9. embeddings or semantic search only as an additional candidate source

Never use vector similarity as the sole reason to omit an explicit graph or
contract dependency.

## Dependency Closure

Start with directly affected capabilities. Traverse dependencies until a
stable implementation boundary is reached. Include:

- full specs for capabilities whose behavior may change
- full relevant contracts and schema deltas
- exact source/test files likely to be edited
- concise summaries plus evidence links for unchanged dependencies
- reverse dependents whose public behavior may regress
- cross-cutting rules applicable to the requested operation

If a dependency is unresolved or contradictory, expand the packet or create an
OpenSpec question before implementation.

## Token Budget

Detect the model context limit when available. Reserve space for reasoning,
tool output, generated patches, and final response. As a default, use no more
than 70 percent of the context window for retrieved material.

Rank packet entries:

```text
REQUIRED_FULL
REQUIRED_SUMMARY
OPTIONAL_ON_DEMAND
EXCLUDED_WITH_REASON
```

If required material exceeds the budget, split implementation into explicit
tasks with separate packets. Do not compress contracts, invariants, failure
semantics, or acceptance criteria into lossy summaries merely to fit one task.

## Packet Artifact

For `CHANGE_TASK`, create:

```text
openspec/context-packets/<task-or-change-id>.yml
```

The packet stores references and short evidence-backed summaries, not copied
source files or secrets. Use `templates/task-context-manifest.template.yml`.

For `REPOSITORY_QUESTION`, keep the packet in memory by default. Persist a
`templates/grounded-answer.template.yml` record only when the question is
complex, reusable, under audit, or the user requests it. Always reopen cited
evidence in the current turn even when a persisted record exists.

Required fields:

- request and scope
- bootstrap/coverage status
- directly affected capability and requirement IDs
- exact command/parameter IDs when applicable
- dependency closure
- required specs, architecture, contracts, source, and tests
- applicable cross-cutting concerns
- unknowns and contradictions
- token-budget estimate
- exclusions with reasons
- expansion triggers
- freshness evidence

## Sufficiency Gate

Return:

- `SUFFICIENT`: all material behavior boundaries have evidence.
- `SUFFICIENT_WITH_DECLARED_GAPS`: work may proceed only if the proposal makes
  the gaps visible and the user accepts the risk.
- `INSUFFICIENT`: expand retrieval or bootstrap relevant areas before changing
  production behavior.

## Output

```text
# Task Context Ready

Result: <result>
Capabilities: <count>
Required source/test files: <count>
Cross-boundary dependencies: <count>
Unknowns or contradictions: <count>
Packet: <path>
```
