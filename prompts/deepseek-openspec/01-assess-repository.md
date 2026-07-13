# Assess Repository For OpenSpec

Use this prompt for a read-only assessment before initialization or when the
user chooses to inspect the plan.

Read `00-global-rules.md` first. Do not create or modify files in this step.

## Lightweight Discovery

Use deterministic filesystem and repository tools before asking the model to
interpret source code. Inspect:

- tracked and untracked path counts
- top-level directories
- build, workspace, dependency, and package manifests
- language distribution by path and extension
- source, test, schema, migration, infrastructure, and documentation roots
- apparent generated, vendor, binary, cache, output, and secret-bearing paths
- existing agent instructions
- current `prompt-pack.yml`, `openspec/meta.yml`, and migration state
- existing `openspec/` state and active changes

Do not read every source body during assessment. This phase estimates and
partitions the future bootstrap; the approved full bootstrap performs the
complete in-scope analysis.

## Detect

- monorepo or single project
- languages and build systems
- applications, services, modules, and shared libraries
- likely deployment units
- API/contract technologies
- persistence and migration technologies
- event, queue, scheduler, or background-job technologies
- test frameworks and test roots
- available OpenSpec CLI
- available native subagent/worker tools
- CLI frameworks, executable entry points, build task systems, and command-help
  surfaces
- obvious repository-local instructions and security restrictions

## Estimate Full Bootstrap

Produce a provisional partition plan:

- proposed manifest shards
- proposed bounded contexts or module groups
- worker roles that apply
- candidate parallelism based on runtime capabilities
- paths that should be excluded, with reasons
- risks such as generated source, duplicated services, sparse tests, or missing
  build metadata

Do not promise elapsed time. Do not call the repository fully covered.

## Determine State

Return one:

```text
UNINITIALIZED_OPENSPEC
PARTIAL_CONTEXT
FULL_BOOTSTRAP_IN_PROGRESS
OPENSPEC_READY
OPENSPEC_READY_WITH_DECLARED_GAPS
OPENSPEC_ACTIVE_CHANGES
OPENSPEC_NEEDS_REFRESH
OPENSPEC_UPGRADE_REQUIRED
PROMPT_PACK_DRIFT
TARGET_NEWER_THAN_PACK
```

## Output

Keep it short and user-facing:

```text
# Repository Assessment

## State

## Repository Snapshot

## Proposed Analysis Partitions

## Exclusions To Record

## Risks Or Unknowns

## Recommended Menu Choice
```
