# Explain A Git Ticket With Code

Use when the user asks to show, explain, inspect, or locate the code for one
ticket previously indexed from Git history. Read `00-global-rules.md` and
`15-answer-repository-question.md` first.

## Input

If the ticket ID was not supplied, ask one short question for it. Do not ask for
the prefix when the ID uniquely matches an existing Git ticket index.

Use `scripts/git_ticket_history.py status` and `context` to locate the index and
build bounded raw evidence. If no index contains the ticket, tell the user to
run `17-index-git-tickets.md`; do not guess commits from a loose text search and
present them as indexed history.

## Freshness

- If Git refs changed, rerun the deterministic scan with the prefix and options
  stored in `meta.json`.
- If the ticket signature is unchanged and its analysis is `completed`, reuse
  historical analysis.
- If the signature changed, analysis is absent/failed/stale, or current-code
  evidence predates the current checkout, refresh only this ticket.
- Historical commit evidence is immutable while the object exists; current-code
  locations and descriptions must be revalidated after repository changes.

## Fast Targeted Analysis

When refresh is required and native subagents exist, dispatch these independent
read-only assignments in parallel:

1. `agents/git-ticket-code-locator.md` in `HISTORICAL_DIFF` mode.
2. `agents/git-ticket-code-locator.md` in `CURRENT_CODE` mode.
3. `agents/git-ticket-code-locator.md` in `TESTS_CONTRACTS` mode.

Without native subagents, execute the three modes as separate bounded passes.
The coordinator synthesizes the result into the ticket analysis template, then
uses `agents/git-ticket-analysis-auditor.md` before marking it completed.

## Evidence Rules

- Every behavioral statement must cite a commit plus file/symbol evidence.
- Distinguish what the ticket changed historically from what the current code
  does now.
- Label descriptions derived only from commit/diff evidence as
  `INFERRED_FROM_GIT_HISTORY`.
- Use `CONFIRMED_BY_TEST`, `CONFIRMED_BY_CONTRACT`, or
  `CONFIRMED_BY_CURRENT_CODE` only after reopening that evidence.
- If commits mention multiple tickets, show the overlap and avoid exclusive
  attribution.
- If a ticket appears only in a branch/tag/ref name, report `REF_ONLY` and do
  not invent a ticket scope.
- If the implementation was removed or cannot be located, say so and show the
  last proven historical location.

## User-Facing Output

Keep code excerpts bounded and high signal:

```text
# <TICKET-ID>

## Что изменили
<evidence-backed description>

## Коммиты
<hash, subject, date, provenance>

## Код тогда
<file, symbol, relevant diff excerpt>

## Код сейчас
<current file, symbol, relevant excerpt, or removed/not found>

## Тесты и контракты
<linked evidence>

## Что происходило позже
<later commits/refactors affecting the same code>

## Неизвестно или неоднозначно
<mixed-ticket commits, missing intent, ref-only evidence>
```

Do not dump full patches or every changed file. Start with the key implementation
and tests; provide additional files only when the user asks.
