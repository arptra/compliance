# Resume OpenSpec Session

Use at the start of a fresh CLI session. Read `00-global-rules.md` first. Keep
the user interaction to one short menu.

## Read State, Not The Repository

Read:

- current `prompt-pack.yml` and target `openspec/meta.yml`
- `openspec/bootstrap/state.yml`
- `openspec/bootstrap/work-queue.yml` summary
- `openspec/index/coverage.yml`
- active change proposals and task lists
- referenced task context packets
- Git ticket registry and queue summaries when present
- `openspec/error-kb/index.yml` only when relevant

Do not reread source files until a selected queue item or task context packet
identifies them.

## Prompt-Pack Upgrade Takes Priority

If version, fingerprint, or supported artifact schema differs, route to
`16-upgrade-existing-openspec.md` before resuming bootstrap or active changes.
Preserve the existing queue and task state. If target schemas are newer than the
pack, stop and request the matching/newer pack instead of downgrading.

Before dispatching resumed work, read the shared queue's `scheduler` block. If
`rate_limit_latched` is true, keep `GLOBAL_SERIAL_QUEUE`, honor
`retry_not_before`, and resume at most one request from FIFO order. Never create
a fresh parallel wave merely because the CLI process restarted.

## Bootstrap Takes Priority

If bootstrap status is `IN_PROGRESS`:

- detect abandoned `running` items;
- preserve completed findings;
- show Menu B from `START_HERE.md`;
- choice `1` resumes `12-full-bootstrap-orchestrator.md` at the checkpoint.

If bootstrap is `BLOCKED`, show the blocker and the count of non-blocked work
remaining. Offer to continue non-blocked work before asking for external help.

## Active Changes

If bootstrap is ready and active changes exist, show:

```text
# OpenSpec Session Restored

1. Continue the next unfinished task.
2. Show active changes and task counts.
3. Run tests and investigate failures.
4. Start a new change.
5. Refresh stale task context.
6. Stop.

Reply with 1-6.
```

Translate to the user's language. For choice `1`, validate packet freshness
before reading source or editing.

If there are no active changes, show Menu C from `START_HERE.md`.

Treat unfinished Git ticket analysis queues as resumable work. When the user
chooses to continue them, execute `17-index-git-tickets.md` with the recorded
prefix and process only pending/stale/failed items. Do not reread completed
ticket analyses whose signatures are unchanged.

## Output Discipline

Show counts, next task, validation status, and declared gaps. Do not dump worker
findings, all specs, or all source paths unless requested.
