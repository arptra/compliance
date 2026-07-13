# DeepSeek OpenSpec Prompt Pack For Large Repositories

This prompt pack lets a local DeepSeek model build and use an evidence-backed
current-state SDD for repositories that cannot fit in one context window.

The repository is processed in deterministic, resumable shards. The complete
description lives in OpenSpec specs and machine-readable indexes; daily tasks
receive a bounded context packet assembled from graph and evidence links.

Java/Gradle has first-class command guidance, while bootstrap discovery also
supports polyglot repositories and monorepos.

## Start With One Prompt

Open the local model CLI in the target repository root and enter:

```text
Read `/path/to/prompts/deepseek-openspec/START_HERE.md` and follow it for this repository. Keep the interface simple and respond in my language.
```

Replace the path with the real prompt-pack location. The user should not need
to paste the numbered prompts manually.

## First-Run Modes

- `FULL_BOOTSTRAP` inventories every repository path, processes every in-scope
  artifact in bounded waves, synthesizes current-state specs, builds
  bidirectional traceability, and runs an independent coverage audit.
- `QUICK_BOOTSTRAP` creates only a lightweight foundation and is always labeled
  `PARTIAL_CONTEXT`.

The simple first-run menu recommends full bootstrap. After one `YES`, the model
continues through all non-blocked phases without requesting approval for each
batch.

## Why The Whole Repository Is Not Loaded

Full coverage and full context are different things:

```text
repository inventory
  -> bounded work queue
  -> parallel evidence workers
  -> structured findings
  -> coordinator synthesis
  -> traceability graph
  -> independent coverage audit
```

No single worker sees the whole repository. The persistent queue and indexes
provide system-level memory across workers, context windows, and CLI restarts.

## Subagents

The orchestrator detects whether the local CLI exposes real subagent tools.

- With native subagents, independent assignments run at the highest safe
  concurrency, starting conservatively when the runtime limit is unknown.
- Without native subagents, the same assignments run as isolated sequential
  batches with checkpoints.

Workers never concurrently edit canonical specs. Each writes one structured
finding file; a single coordinator validates and merges results.

## Main OpenSpec Artifacts

```text
openspec/
  project.md
  glossary.md
  architecture/
    decisions.md
  specs/<capability-id>/spec.md
  changes/<change-id>/
  index/
    repository-manifest.yml
    files/
    modules.yml
    capabilities.yml
    traceability.yml
    coverage.yml
    contradictions.yml
    unknowns.yml
  bootstrap/
    state.yml
    work-queue.yml
    exclusions.yml
    findings/
    reports/
  context-packets/
  error-kb/
```

Every discovered requirement carries repository evidence and one of these
states:

```text
CONFIRMED_BY_CONTRACT
CONFIRMED_BY_TEST
OBSERVED_IN_CODE
INFERRED_FROM_CODE
UNKNOWN
CONTRADICTED
```

## Completion Statuses

- `READY`: every applicable coverage gate passed with no declared gaps.
- `READY_WITH_DECLARED_GAPS`: every in-scope area was processed, but explicit
  unknowns, contradictions, missing tests, or missing contracts remain.
- `IN_PROGRESS`: inventory, extraction, synthesis, validation, or audit work
  remains.
- `BLOCKED`: an external dependency prevents remaining work.

The model must not call a repository ready while work remains unprocessed.

## Normal Feature Work

For each non-trivial change:

1. `13-build-task-context.md` selects affected capability specs, dependency
   closure, contracts, code, tests, and cross-cutting rules.
2. `07-new-feature-spec.md` creates the OpenSpec proposal, tasks, design, and
   spec deltas.
3. The user approves implementation once.
4. `08-implement-spec-task.md` implements and verifies bounded tasks.
5. Refresh/archive updates affected current-state specs and traceability.

Task packets store references and short summaries, not copied source trees.

## Prompt Map

- `START_HERE.md`: simple state-aware entry point and menus
- `00-global-rules.md`: evidence, coverage, security, and orchestration rules
- `01-assess-repository.md`: read-only dry-run assessment
- `02-initialize-openspec-foundation.md`: full or quick file foundation
- `03-fill-openspec-project-context.md`: architecture synthesis
- `04-create-current-state-specs.md`: capability-spec synthesis
- `05-add-openspec-validation.md`: structural and traceability validation
- `06-final-review.md`: final bootstrap status review
- `07-new-feature-spec.md`: approved change specification
- `08-implement-spec-task.md`: bounded implementation workflow
- `09-resume-session.md`: checkpoint and active-task resume
- `10-refresh-openspec-context.md`: diff/hash-driven incremental refresh
- `11-error-memory.md`: reusable sanitized failure memory
- `12-full-bootstrap-orchestrator.md`: complete bootstrap scheduler
- `13-build-task-context.md`: retrieval and task packet builder
- `14-audit-sdd-coverage.md`: independent repository-to-SDD audit
- `agents/`: specialized worker contracts
- `templates/`: machine-readable artifact templates

## Safety Boundaries

Bootstrap and refresh do not edit production code. Generated/vendor trees,
binaries, caches, outputs, and secret-bearing content are excluded with an
explicit recorded reason. Commit, push, deletion, installation, and destructive
commands still require an explicit user request.
