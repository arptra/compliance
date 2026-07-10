# Resume OpenSpec Session

Use this prompt at the start of a fresh CLI session when the repository is
already initialized for OpenSpec.

## Read Only

- `openspec/project.md`
- active `openspec/changes/*/proposal.md`
- active `openspec/changes/*/tasks.md`
- `openspec/error-kb/index.yml` if it exists

Do not read all source files.

## Decide

If there are no active changes, show the normal menu.

If there are active changes, show a numbered list and ask what to do:

```text
1. Continue next unfinished task.
2. Show task list.
3. Run tests and fix failures.
4. Create new OpenSpec change.
5. Archive completed change.
6. Stop.
```

## Output

```text
# OpenSpec Session Restored

## Active Changes

## Suggested Next Task

## Validation Status

## Menu
```
