# Implement An Approved OpenSpec Task

Use after an OpenSpec change exists and the user approved implementation. Read
`00-global-rules.md` first.

## Restore The Task Boundary

Read:

- selected proposal, tasks, design, and spec deltas
- selected change `context.yml`
- affected current-state specs and graph records
- relevant error-memory index entries

Re-run `13-build-task-context.md` when the packet is stale, the selected task is
broader than the packet, source hashes changed, or implementation discovers a
new cross-boundary dependency.

Do not continue on an `INSUFFICIENT` context result.

## Use Parallelism Safely

When real subagents are available, use them for independent read-only work:

- impact and reverse-dependency review
- test discovery and test-plan review
- contract/schema compatibility review
- security, migration, reliability, and performance review
- independent post-implementation review

Parallel code edits are allowed only for explicitly separate tasks with
non-overlapping file ownership and independent validation commands. Otherwise,
the coordinator performs canonical edits after gathering worker findings.

Use the highest safe runtime concurrency and reduce it after resource failures.
Never pretend subagents ran if the environment does not provide them.

## Implementation Workflow

Before running a repository-specific command, task, module selector, flag, or
script, verify its exact token and parameters in the fresh command index,
current build/parser declaration, or side-effect-free help/task listing. Do not
infer a Gradle task path, npm script, CLI flag, default, or environment variable
from conventions. If it cannot be verified, use safe discovery or report
`NOT_VERIFIED` instead of executing a guessed command.

1. Choose the next unchecked, dependency-ready task.
2. State the files and behavior boundary before editing.
3. Add or update tests first where practical.
4. Implement the smallest change satisfying the approved requirement.
5. Preserve unrelated user changes.
6. Run the narrowest relevant repository-native tests.
7. Run broader contract/module tests when blast radius crosses boundaries.
8. On failure, search `openspec/error-kb/index.yml` by normalized fingerprint
   before debugging from scratch.
9. Record a verified reusable fix without secrets when a new recurring failure
   is solved.
10. Run an independent diff/spec/test review for risky changes.
11. Update task checkboxes only after their verification succeeds.
12. Validate the OpenSpec change and its context/traceability references.

For Gradle repositories, prefer the wrapper and targeted module tests:

```bash
./gradlew :module:test
```

Use broader tasks only when impact requires them.

## New Scope During Implementation

If implementation requires behavior outside the approved delta:

- stop that part of implementation;
- expand the task context packet;
- update proposal/design/spec delta;
- validate again;
- ask for approval only when the user-visible or risk-bearing scope materially
  changed.

Do not silently widen the change.

## After Each Task

Record:

- files changed
- tests added or updated
- commands and results
- requirements satisfied
- new or changed traceability links
- error-memory entries used or created
- remaining tasks

Do not rewrite all current-state specs during implementation. Apply final
deltas and incremental index freshness updates when the change is archived.

## Output

```text
# Implementation Progress

Change: <change-id>
Task: <task>
Result: <completed|blocked|failed>
Tests: <short result>
OpenSpec validation: <result>
Independent review: <result or unavailable>
Remaining tasks: <count>
```
