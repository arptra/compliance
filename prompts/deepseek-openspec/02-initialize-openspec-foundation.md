# Initialize OpenSpec Foundation

Use only after repository assessment and user approval. Read
`00-global-rules.md` first. Do not change production code.

The caller must set one mode:

- `FULL_BOOTSTRAP`: create the complete resumable structure, then return
  control to `12-full-bootstrap-orchestrator.md`.
- `QUICK_BOOTSTRAP`: create a lightweight structure and label it
  `PARTIAL_CONTEXT`.

## Preserve Existing Work

- Never overwrite existing confirmed requirements or agent instructions.
- Merge OpenSpec rules into an existing `AGENTS.md` under a clearly named
  section.
- If an existing OpenSpec layout differs, record the difference and adapt
  without destructive migration.

## Create The Foundation

Create missing directories and initial files:

```text
openspec/
  project.md
  glossary.md
  architecture/
    system-context.md
    runtime-and-deployment.md
    cross-cutting-concerns.md
    decisions.md
  specs/README.md
  changes/README.md
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
  context-packets/README.md
  error-kb/
    README.md
    index.yml
    entries/.gitkeep
```

Create a root `AGENTS.md` when absent, or carefully merge the OpenSpec section
into the existing file while preserving all repository-specific instructions.

Use templates from this prompt pack when present. Keep YAML machine-readable;
do not place prose paragraphs inside fields intended as IDs, statuses, counts,
or paths.

## Initial Bootstrap State

For `FULL_BOOTSTRAP` initialize:

```yaml
mode: FULL_BOOTSTRAP
status: IN_PROGRESS
phase: foundation
orchestration_mode: UNDETECTED
```

For `QUICK_BOOTSTRAP` initialize:

```yaml
mode: QUICK_BOOTSTRAP
status: PARTIAL_CONTEXT
phase: foundation-complete
orchestration_mode: NONE
```

Include timestamps only when the environment can produce them reliably. Do not
fabricate hashes, counts, owners, or runtime facts.

## `project.md` Initial Content

Include only confirmed lightweight facts:

- repository purpose from explicit docs, otherwise `UNKNOWN`
- detected languages and build systems
- top-level project/module candidates
- repository-native build and test commands when confirmed
- current bootstrap mode and status
- links to canonical indexes
- distinction between discovered current behavior and approved future changes

Initialize `architecture/decisions.md` as an evidence-backed index of existing
ADR files and important architectural constraints. Do not invent decision
rationale that is not present in repository history, docs, contracts, or code.

Detailed architecture and behavior are filled by later bootstrap phases.

## `AGENTS.md` Requirements

Add or merge rules requiring:

- task context packets before non-trivial feature work
- approved OpenSpec change before production behavior changes
- repository-native tests after implementation
- OpenSpec and traceability validation
- current-state evidence labels
- error-memory lookup before repeated debugging
- incremental index refresh after affected code changes

## Output

```text
# OpenSpec Foundation Created

Mode: <FULL_BOOTSTRAP|QUICK_BOOTSTRAP>
Files created: <count>
Files updated: <count>
Status: <IN_PROGRESS|PARTIAL_CONTEXT>
Next internal phase: <manifest|none>
```

In `FULL_BOOTSTRAP`, do not show a user menu here. Return to the orchestrator.
