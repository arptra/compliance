# Index Git Ticket History

Use when the user asks to extract, discover, parse, import, or index ticket IDs
from local Git history. Read `00-global-rules.md` first. Do not change
production code and do not fetch from remotes.

## Ask One Prefix Question

On the first index, if the prefix was not supplied, ask exactly:

```text
Как выглядит префикс тикета до номера, включая разделитель?
Например: PROJ- для тикетов PROJ-123.
```

Use the supplied prefix literally. Do not guess separators or silently search
variants such as `PROJ123`, `PROJ_123`, or `PROJ 123`. Advanced users may run
separate indexes for additional exact prefixes.

When resuming and exactly one existing Git ticket index has unfinished items,
reuse its recorded prefix without asking again. When multiple incomplete
prefixes exist, show their exact recorded prefixes and ask the user to choose.

## Deterministic Extraction

Resolve `scripts/git_ticket_history.py` relative to this prompt pack. Verify the
target is a Git repository, then run:

```bash
python3 <prompt-pack>/scripts/git_ticket_history.py scan \
  --repo . \
  --prefix '<exact-prefix>' \
  --case-sensitive \
  --include-reflog \
  --include-ref-names
```

Use the repository's available Python 3 executable. Do not install Python or
dependencies. The script uses only the standard library.

The scan is local and read-only with respect to Git. It covers commits reachable
from all local and remote-tracking refs, local reflogs, Git notes, and ticket IDs
in ref names. It does not run `git fetch`, access an issue tracker, or inspect
unreachable objects outside reflogs.

Persistent records live under:

```text
openspec/history/git-tickets/
  registry.json
  <prefix-key>/
    meta.json
    index.json
    queue.json
    tickets/<ticket-id>.json
    analyses/<ticket-id>.md
    findings/<worker-id>.yml
```

Raw patches and commit text must not be stored in the index. Ticket records
contain commit IDs, dates, parents, match line numbers, refs, and signatures.
Agents retrieve full message/diff evidence only while analyzing a bounded
ticket assignment. The `context` command returns at most 50 commits by default;
workers must follow `commit_window.next_offset` until `has_more` is false,
checkpointing each window instead of requesting unbounded history.

## Parallel Ticket Analysis

After deterministic extraction, parse `queue.json` with a structured parser.
Use native subagents when available; otherwise run the same assignments as
isolated bounded passes.

Use the highest safe concurrency. Keep worker slots filled while runnable
ticket items remain. Reduce concurrency after non-429 resource failures without
losing completed analyses. An explicit HTTP 429 activates `Global HTTP 429
Queue` from `00-global-rules.md`: log the worker/request and Retry-After, stop
filling parallel slots, return rate-limited tickets to retry state, and process
every waiting/retry ticket and auditor request through one FIFO queue with
concurrency `1`.

Partition by cost:

- high-cost ticket: one ticket per worker;
- low-cost batch: at most 10 tickets and at most 50 total commits;
- split any ticket whose context would exceed 70 percent of the worker context
  budget into history, current-code, and test/contract assignments.

Each worker receives:

- `agents/00-worker-contract.md`
- `agents/git-ticket-batch-analyzer.md`
- `templates/git-ticket-worker-result.template.yml`
- assigned ticket IDs and ticket files
- access to `scripts/git_ticket_history.py context`
- current OpenSpec indexes only when linked paths/capabilities are relevant

Before dispatch, the coordinator marks the assigned queue items `running`.
Workers write only unique analysis files and unique findings files. Only the
coordinator updates shared `index.json` and `queue.json` through the script's
`mark` command.

When a worker/batch returns an explicit HTTP 429, record that distinct request
once against one representative assigned ticket before waiting:

```bash
python3 <prompt-pack>/scripts/git_ticket_history.py mark \
  --repo . \
  --prefix '<exact-prefix>' \
  --ticket '<ticket-id>' \
  --status rate_limited \
  --http-status 429 \
  --worker-id '<worker-id>' \
  --request-id '<sanitized-id-if-available>' \
  --retry-after '<value-if-available>'
```

Relay both returned `log_records` to the runtime logger and show the returned
`user_message` translated to the user's language. The persisted scheduler
rejects a second running ticket and out-of-order dispatch while the global 429
latch is active.

For a multi-ticket batch, return its other assigned `running` tickets to
`pending` with a short reference to the same worker request, but do not record
additional fake 429 events. One actual rate-limited request produces one warning
event even when several ticket assignments must be requeued.

## Required Ticket Analysis

For each ticket create the file named by its queue item using
`templates/git-ticket-analysis.template.md`. Include:

- exact ticket ID and evidence status
- associated commits and message/ref provenance
- concise description grounded in commit/diff evidence
- historical behavior/code introduced, changed, or removed
- changed files and key symbols
- tests, contracts, migrations, config, and operational impact
- where the relevant implementation lives in the current checkout
- evolution after the ticket, including later rewrites or removal
- shared/mixed commits that make attribution ambiguous
- unknowns and confidence

Do not treat a ref-name-only match as proof that every commit on the branch
belongs to the ticket. Do not assign the full content of a multi-ticket commit
to each ticket without stating the ambiguity.

## Audit And Completion

Dispatch `agents/git-ticket-analysis-auditor.md` for completed batches. The
auditor reopens commit/path evidence and checks ticket separation. Mark a ticket
`completed` only after audit passes. Failed analyses become `failed` with a
short error and may be retried by another worker.

Checkpoint after every completed ticket/batch. On interruption, leave unfinished
items as `pending` or `stale` so the next CLI session resumes without repeating
completed ticket analyses.

## Reindexing

Running the same scan again is safe. The deterministic scan rechecks local Git
refs, preserves `completed` status when a ticket signature is unchanged, and
marks only changed ticket records `stale`. Analyze new/stale tickets instead of
rereading every completed ticket.

## Output

```text
# Git Ticket History Indexed

Prefix: <prefix>
Commits scanned: <count>
Commits with ticket references: <count>
Tickets found: <count>
Analyses completed: <count>
Pending/stale/failed: <counts>
Storage: <path>

Напишите ID тикета, например PROJ-123, чтобы увидеть код и описание.
```
