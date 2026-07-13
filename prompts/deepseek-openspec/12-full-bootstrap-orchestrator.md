# Full Bootstrap Orchestrator

Use after the user approves complete current-state initialization. This prompt
coordinates deterministic inventory, parallel evidence extraction, synthesis,
validation, and independent coverage audit across repositories that cannot fit
in one model context.

Read these prompt-pack files before starting:

1. `00-global-rules.md`
2. `02-initialize-openspec-foundation.md`
3. `03-fill-openspec-project-context.md`
4. `04-create-current-state-specs.md`
5. `05-add-openspec-validation.md`
6. `06-final-review.md`
7. `15-answer-repository-question.md`
8. `16-upgrade-existing-openspec.md`
9. all role prompts under `agents/` that apply to the repository
10. templates under `templates/`

Resolve that list relative to this prompt pack. Resolve generated `openspec/`
paths relative to the target repository root.

## Phase 0 - Prompt-Pack Compatibility

Read `prompt-pack.yml`. If an existing target has `openspec/meta.yml`, compare
version, artifact schema, and computed pack fingerprint before resuming queue
work. If `openspec/` exists without `meta.yml`, treat it as
`LEGACY_UNVERSIONED`. Execute `16-upgrade-existing-openspec.md` when required.
Never let a new prompt pack silently reinterpret an older queue or index schema.

Do not change production code. Once approved, continue autonomously through all
non-blocked phases. Keep the user informed with short progress summaries.

## Recovery Before New Work

If `openspec/bootstrap/state.yml` exists:

1. parse it with a structured YAML parser when available;
2. validate queue and manifest references;
3. preserve completed work;
4. move abandoned `running` items back to `pending`, recording the recovery;
5. resume from the earliest incomplete phase.

Do not repeat completed work whose evidence hashes and manifest records are
still current.

## Orchestration Capability Detection

Inspect the current CLI/tool environment for real subagent, worker, task, or
parallel-dispatch capabilities.

- If available, set `orchestration_mode: NATIVE_SUBAGENTS` and record the
  actual mechanism.
- If unavailable, set `orchestration_mode: ISOLATED_BATCHES` and execute the
  same worker assignments as bounded sequential passes.
- Never claim subagent execution without evidence from the tool environment.

Use the maximum safe concurrency reported by the runtime. If unknown, begin
with at most four independent workers. Increase up to eight after successful
batches only when resources remain healthy. On OOM, timeout, or resource
pressure, reduce concurrency and retry unfinished assignments.

## Phase 1 - Foundation

If the full structure does not exist, execute
`02-initialize-openspec-foundation.md` with `FULL_BOOTSTRAP`.

Checkpoint:

```yaml
phase: manifest
status: IN_PROGRESS
```

## Phase 2 - Deterministic Repository Manifest

Build the inventory with filesystem, version-control, build-system, and search
tools before model interpretation.

Inventory all repository paths and record:

- stable path
- tracked/untracked status when available
- file type/language
- size and content hash or version-control object ID when practical
- likely module or root
- artifact class
- scope status and exclusion reason
- analysis status
- manifest shard

Allowed artifact classes include:

```text
production-source
test-source
public-contract
data-schema
migration
event-or-job
configuration
infrastructure
documentation
build-tooling
command-interface
generated
vendor-or-dependency
binary
cache-or-output
secret-bearing
unknown
```

Default exclusions are generated artifacts, dependency/vendor trees, binaries,
caches, outputs, and secret-bearing content. Do not infer that an unfamiliar
directory is generated; record evidence for the exclusion. Configuration and
infrastructure are in scope unless policy excludes them.

Write summary metadata to `openspec/index/repository-manifest.yml` and shard
file records under `openspec/index/files/`. Shard by module or bounded path so
future task-context retrieval does not load the global file list.

Every path must be represented directly or by a documented directory-level
exclusion rule whose match count is recorded.

Checkpoint manifest counts and continue to partitioning.

## Phase 3 - Topology And Work Queue

Use build/workspace manifests and directory structure to identify candidate:

- applications and services
- deployable units
- modules and shared libraries
- source and test roots
- contract/schema roots
- migration and persistence roots
- infrastructure and runtime roots
- likely bounded contexts

Create stable work items in `openspec/bootstrap/work-queue.yml`. Each item must
have:

- unique ID
- role prompt
- bounded path/module/context assignment
- explicit in-scope artifact IDs or manifest shard
- non-overlap rule
- dependencies on prior items
- output finding path
- status, attempt count, and last error

Create these global assignments when applicable:

- repository cartography
- command interfaces and build/operator tasks using
  `agents/command-interface-analyzer.md`
- contracts and data boundaries
- test-derived behavior
- runtime and cross-cutting behavior

Create one or more domain-capability assignments for every in-scope bounded
context. Split oversized assignments by submodule or artifact count. Merge tiny
ones only when they share a domain boundary.

## Phase 4 - Parallel Evidence Extraction

Dispatch runnable queue items in waves. Each worker receives only:

- `00-global-rules.md`
- `agents/00-worker-contract.md`
- its role prompt
- its work-item record
- assigned manifest shard or artifact list
- already-confirmed minimal shared vocabulary if required

Workers must not receive unrelated canonical specs as a substitute for direct
repository evidence. Workers are read-only except for their unique findings
file.

Keep available worker slots filled while independent runnable items remain;
do not wait for an entire slow wave before dispatching replacements into free
slots. The coordinator alone owns scheduling and retries.

After every worker result:

1. validate its YAML structure;
2. verify referenced paths exist and are within assignment scope;
3. reject unsupported claims and fabricated counts;
4. mark the work item `completed`, `retry`, or `blocked`;
5. update file analysis statuses;
6. atomically checkpoint queue and state before dispatching the next wave.

When workers discover cross-boundary dependencies, create targeted follow-up
items. Do not let one worker recursively absorb the rest of the repository.

Continue until no extraction item is pending or runnable.

## Phase 5 - Synthesis

Use a single coordinator/synthesizer to avoid canonical write conflicts.

1. Execute `03-fill-openspec-project-context.md` from validated findings.
2. Synthesize exact command records with
   `templates/commands-index.template.yml` into
   `openspec/index/commands.yml` and declared command shards using
   command-interface findings.
3. Link commands to entry points, artifacts, tests, capabilities, and evidence.
4. Group capability candidates by bounded context and resolve duplicates.
5. Create synthesis work items for every candidate group.
6. Execute `04-create-current-state-specs.md` in context-safe waves.
7. Update capabilities, bidirectional traceability, unknowns, contradictions,
   and manifest analysis status after each wave.

When findings disagree, preserve both evidence records and create a
contradiction. Do not choose a business interpretation merely because one
worker used stronger language.

## Phase 6 - Validation And Repair

Execute `05-add-openspec-validation.md`.

- Structural or link errors create repair work items.
- Missing evidence creates extraction work items.
- Unclassified or unanalyzed artifacts create manifest/domain work items.
- Explicitly absent tests or contracts create declared gaps, not fabricated
  links.

Run repair waves until validation passes or a genuine blocker remains.

## Phase 7 - Independent Coverage Audit

Dispatch `agents/coverage-auditor.md` with no synthesis responsibility, then
execute `14-audit-sdd-coverage.md`.

The auditor must compare repository inventory against the completed SDD, not
only review documents for internal consistency. Audit findings become work
items. Re-run extraction, synthesis, and validation for material omissions.

Repeat audit and repair until:

- all coverage gates pass, or
- every remaining issue is an explicit declared gap or external blocker.

## Phase 8 - Finalization

Execute `06-final-review.md` and set the honest final status:

- `READY`
- `READY_WITH_DECLARED_GAPS`
- `IN_PROGRESS`
- `BLOCKED`

Update final counts, shard references, freshness data, queue summary, and audit
result. Ensure no production files were modified during bootstrap.

Show the simple final menu from `START_HERE.md`. Do not expose internal worker
logs unless the user requests diagnostics.

## Interruption Handling

If the user interrupts or the session ends:

- finish only the current atomic finding/index write;
- checkpoint queue and state;
- leave unfinished work as `pending` rather than `completed`;
- report the exact resume state;
- do not mark bootstrap ready.
