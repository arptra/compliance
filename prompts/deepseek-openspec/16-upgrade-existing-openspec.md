# Upgrade Existing OpenSpec For A New Prompt Pack

Use when the current `prompt-pack.yml` version or fingerprint differs from the
version recorded in the target repository's `openspec/meta.yml`, or when an
unversioned legacy OpenSpec installation is detected.

Read `00-global-rules.md`, the current `prompt-pack.yml`, and every migration
prompt listed between the installed and current versions. Do not change
production code.

## Goals

- allow prompt-pack files to be replaced or updated in place;
- preserve confirmed specs, active changes, worker findings, queue state,
  unknown fields, and user edits;
- apply only missing additive or explicitly versioned transformations;
- backfill new indexes without forcing a complete repository restart;
- make interrupted migrations resumable and idempotent.

## Version Detection

Read:

- current prompt-pack version and fingerprint rules
- `openspec/meta.yml` when present
- per-artifact `schema_version` values
- bootstrap state and active migration record

Determine one state:

```text
UP_TO_DATE
LEGACY_UNVERSIONED
UPGRADE_REQUIRED
SAME_VERSION_PROMPT_DRIFT
TARGET_NEWER_THAN_PACK
MIGRATION_IN_PROGRESS
```

Compute the prompt-pack fingerprint from sorted paths and file contents using
the algorithm and include patterns in `prompt-pack.yml`. Do not include target
repository files in this fingerprint.

Use the canonical `path-null-size-null-content-null-v1` format:

1. resolve include globs relative to the prompt-pack root;
2. include regular files only and do not follow symlinks;
3. normalize relative path separators to `/` and sort by UTF-8 path bytes;
4. for each file hash UTF-8 relative path, NUL, decimal raw-byte length, NUL,
   raw file bytes, and a final NUL;
5. use lowercase hexadecimal SHA-256 output.

Compare versions as semantic versions, never as lexical strings. Treat a
missing version only as `LEGACY_UNVERSIONED`.

If target artifact schemas are newer than the pack supports, return
`TARGET_NEWER_THAN_PACK` and stop. Never downgrade or rewrite newer structures.

## Preflight And Dry Run

Before writing:

1. inventory existing `openspec/` files and schema versions;
2. detect uncommitted or partially written OpenSpec artifacts without reverting
   them;
3. list migrations not yet recorded as applied;
4. list files to create and existing files to merge;
5. validate that every migration has an idempotency key and verification step;
6. write or update
   `openspec/migrations/<migration-id>.yml` using
   `templates/migration-record.template.yml` with status `planned`.

Show the user one short summary and request one approval for the complete
non-destructive migration. If the user already selected the explicit upgrade
choice in `START_HERE.md`, that selection is the approval; do not ask the same
question again. Do not ask again for each file or backfill batch.

## Migration Rules

- Parse YAML/JSON with structured parsers and preserve unknown fields.
- Never replace an existing spec, active change, unknown, contradiction,
  finding, queue item, or error-memory entry wholesale.
- Add missing fields with conservative defaults.
- Do not change stable IDs unless a migration explicitly provides an alias map.
- Record before/after content hashes for every modified OpenSpec artifact.
- Mark migration status before each atomic phase and checkpoint after it.
- Append the migration ID to `migrations_applied` only after validation passes.
- Re-running an applied migration must produce no changes.
- If interrupted, resume the migration record rather than restarting bootstrap.

## Migration `add-versioning-and-grounded-answers-v1`

For legacy or pre-2.0.0 repositories:

1. Create `openspec/meta.yml` from
   `templates/openspec-meta.template.yml` if missing.
2. Add `openspec/index/commands.yml` and optional command shards without
   overwriting existing custom command documentation.
3. Add command-index and grounded-answer coverage dimensions.
4. Add repository-question grounding rules to the OpenSpec section of
   `AGENTS.md`, preserving all existing instructions.
5. Add command-interface extraction work items for manifest artifacts already
   classified as CLI, build tooling, scripts, config, entry points, or unknown
   command surfaces.
6. If full bootstrap previously completed, change status to `IN_PROGRESS` only
   for the targeted command backfill; preserve all completed work and counts.
7. Run command extraction, synthesis, validation, and targeted coverage audit.
8. Restore `READY` or `READY_WITH_DECLARED_GAPS` based on the new gates.
9. Record current pack version and computed fingerprint only after success.

Do not rescan unrelated domain implementation solely because the command index
is new. Follow discovered links only when command behavior cannot otherwise be
established.

## Future Overlay Updates

Future prompt packs must:

- increment `prompt-pack.yml` version when behavior or artifact contracts
  change;
- append a migration record instead of editing an already-applied migration;
- keep old migration prompts available while supported;
- specify compatible artifact schema versions;
- use additive fields by default;
- require explicit migration logic for removals, renames, or schema changes.

When only wording changes and artifact contracts do not change, a changed
fingerprint with the same version produces `SAME_VERSION_PROMPT_DRIFT`. Run a
compatibility audit and update the fingerprint only after validation; do not
rerun all bootstrap work.

## Validation

After migration verify:

- all pre-existing spec/change IDs still exist;
- queue and bootstrap completed states were preserved;
- YAML/JSON structures parse;
- new indexes and reverse links validate;
- command coverage backfill has no pending runnable items;
- no production file changed;
- the migration is idempotent on a dry second pass;
- `openspec/meta.yml` records version, fingerprint, artifact schema, and applied
  migration IDs.

## Output

```text
# OpenSpec Prompt-Pack Upgrade

From: <legacy or version>
To: <version>
Migration: <id>
Existing specs preserved: <count>/<count>
Files created: <count>
Files merged: <count>
Targeted backfill items: <count>
Validation: <result>
Final coverage status: <status>
```
