# DeepSeek OpenSpec Prompt Pack For Large Repositories

This prompt pack lets a local DeepSeek model build and use an evidence-backed
current-state SDD for repositories that cannot fit in one context window.

The repository is processed in deterministic, resumable shards. The complete
description lives in OpenSpec specs and machine-readable indexes; daily tasks
receive a bounded context packet assembled from graph and evidence links.
Factual repository answers use a claim-level grounding gate and return
`NOT_VERIFIED` instead of inventing missing facts.
An optional Git ticket history workflow extracts exact ticket IDs from local
commit history and uses parallel agents to map each ticket to historical and
current code.

Java/Gradle has first-class command guidance, while bootstrap discovery also
supports polyglot repositories and monorepos.

## Current Version And Capabilities

Current prompt-pack version: `2.1.0`.

The pack can:

- build a complete, evidence-backed current-state SDD for a repository that
  cannot fit into one model context;
- process large repositories in resumable, parallel batches with persistent
  queues and independent coverage audits;
- answer exact repository questions by reopening specs, code, tests, contracts,
  and safe runtime help instead of guessing parameters or defaults;
- create bounded task context packets and run OpenSpec-first feature work;
- resume after a CLI restart without rereading completed repository areas;
- update an existing OpenSpec installation through additive, versioned
  migrations when a newer prompt pack is copied over it;
- index ticket IDs from local Git history and explain each ticket with its
  historical diff, current code, tests, contracts, and later evolution.

### What Is New In `2.1.0`

- deterministic exact-prefix ticket extraction from commit messages, Git notes,
  reflogs, and local ref names;
- persistent ticket registry, queue, signatures, analyses, and stale detection;
- cost-bounded parallel ticket analysis and an independent auditor role;
- an interactive command to show one ticket with historical and current code;
- restart-safe reuse of completed unchanged ticket analyses;
- additive migration `add-git-ticket-history-v1` for existing installations.

See [CHANGELOG.md](CHANGELOG.md) for version-by-version release notes.

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
  meta.yml
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
    commands.yml
    commands/
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
  migrations/
  history/
    git-tickets/
  context-packets/
  error-kb/
```

Every discovered requirement carries repository evidence and one of these
states:

```text
CONFIRMED_BY_CONTRACT
CONFIRMED_BY_TEST
CONFIRMED_BY_RUNTIME
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

## Verified Repository Questions

The user may ask a question directly, for example:

```text
Which exact flag retries failed imports, and what is its default?
```

The model must:

1. search fresh OpenSpec and command indexes;
2. reopen linked current code/contracts/tests;
3. search authoritative declarations if the index is missing or stale;
4. use side-effect-free runtime help only when necessary;
5. validate each atomic claim in a separate pass;
6. return verified evidence, a conflict, or `NOT_VERIFIED`.

Exact flags, aliases, defaults, choices, environment variables, config keys,
API fields, build tasks, and paths cannot be reconstructed from conventions.
The command-interface bootstrap worker records them in
`openspec/index/commands.yml` with exact evidence.

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

## Git Ticket History

Choose the Git ticket workflow or ask directly:

```text
Index all tickets from Git history.
```

The model asks one question for the exact prefix including its separator, for
example `PROJ-`. It then runs the bundled deterministic parser over local
`git log --all`, reflogs, Git notes, and ref names. It never fetches remotes.

The parser groups exact IDs and stores compact records under
`openspec/history/git-tickets/`. It does not persist full patches or commit
bodies. DeepSeek subagents analyze new/stale tickets in cost-bounded parallel
batches and create evidence-backed descriptions, historical code paths, current
code locations, tests, contracts, and later evolution.

After indexing, ask:

```text
Show PROJ-123 with its code and description.
```

Completed unchanged tickets are reused across CLI sessions. A rescan preserves
completed analyses and queues only new/stale ticket signatures.

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
- `15-answer-repository-question.md`: evidence-gated factual repository answers
- `16-upgrade-existing-openspec.md`: idempotent prompt-pack migration
- `17-index-git-tickets.md`: deterministic Git ticket discovery and parallel
  ticket analysis
- `18-explain-git-ticket.md`: historical/current code explanation for one ticket
- `CHANGELOG.md`: version-by-version user-visible changes
- `agents/`: specialized worker contracts
- `scripts/git_ticket_history.py`: dependency-free local Git parser and context
  builder
- `templates/`: machine-readable artifact templates

## Updating The Prompt Pack In Place

The prompt pack is versioned by `prompt-pack.yml`. The target repository stores
the applied version and computed fingerprint in `openspec/meta.yml`.

After prompt files are replaced or updated, start with the same
`START_HERE.md`. It detects:

- a newer version requiring migration;
- changed files with the same version;
- an unversioned legacy OpenSpec installation;
- target artifact schemas newer than the current pack.

`16-upgrade-existing-openspec.md` performs an idempotent, non-destructive
migration. It preserves specs, active changes, queue state, worker findings,
unknown fields, and user edits. New indexes are backfilled only from relevant
artifacts instead of restarting full repository bootstrap.

## Safety Boundaries

Bootstrap and refresh do not edit production code. Generated/vendor trees,
binaries, caches, outputs, and secret-bearing content are excluded with an
explicit recorded reason. Commit, push, deletion, installation, and destructive
commands still require an explicit user request.
