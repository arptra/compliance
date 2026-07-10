# DeepSeek OpenSpec Prompt Pack

This folder contains prompts for running a local model such as
`deepseek-v4-flash-262k` inside an existing Java/Gradle repository with an
OpenSpec-first workflow.

OpenSpec is the source of truth:

```text
openspec/
  project.md
  specs/
  changes/
  error-kb/
```

For daily use, start with `START_HERE.md`.

If you are not a developer and want a menu-driven workflow, read
`USER_WORKFLOW_RU.md`.

## Самая короткая строка для CLI

Откройте DeepSeek CLI в корне Java/Gradle проекта и вставьте:

```text
Прочитай файл `/path/to/prompts/deepseek-openspec/START_HERE.md`, работай строго по OpenSpec для текущей Java/Gradle-репы и веди меня через меню. Я не разработчик: задавай вопросы по одному, проси отвечать цифрами и не перечитывай весь проект без необходимости.
```

Replace `/path/to/prompts/deepseek-openspec/START_HERE.md` with the real path.

## What The Start Prompt Does

1. Detects whether the repository has OpenSpec initialized.
2. If not initialized, asks before creating `openspec/` files.
3. If initialized, shows a small numbered menu.
4. Creates every new feature as an OpenSpec change under
   `openspec/changes/<change-id>/`.
5. Validates changes with `openspec validate <change-id> --strict` when the CLI
   is available.
6. Implements only approved OpenSpec tasks.
7. Runs Gradle unit tests after implementation.
8. Checks `openspec/error-kb/` before debugging repeated failures.
9. Saves verified failure fixes into `openspec/error-kb/` so they can be shared
   through git.

## Normal Menu

After initialization, DeepSeek should show:

```text
1. Create a new OpenSpec change.
2. Create an OpenSpec change and implement it after approval.
3. Continue an existing OpenSpec change/task.
4. Fix a failing test/build/error using OpenSpec error memory first.
5. Refresh OpenSpec project context.
6. Archive a completed OpenSpec change.
```

## Manual Prompt Order

Use numbered files only when you want to run a specific step manually.

1. `00-global-rules.md`
2. `01-assess-repository.md`
3. `02-initialize-openspec-foundation.md`
4. `03-fill-openspec-project-context.md`
5. `04-create-current-state-specs.md`
6. `05-add-openspec-validation.md`
7. `06-final-review.md`

For feature work:

- `07-new-feature-spec.md`
- `08-implement-spec-task.md`

For fresh CLI sessions and context maintenance:

- `09-resume-session.md`
- `10-refresh-openspec-context.md`

For reusable failure fixes:

- `11-error-memory.md`

For non-developer workflow guidance:

- `USER_WORKFLOW_RU.md`

## OpenSpec Change Shape

Each feature/change must live here:

```text
openspec/changes/<verb-led-change-id>/
  proposal.md
  tasks.md
  design.md              # optional, required for broad or risky changes
  specs/
    <capability>/
      spec.md
```

Change IDs should be kebab-case and verb-led:

- `add-report-export`
- `change-payment-status-flow`
- `remove-legacy-auth`
- `refactor-notification-sender`

Spec deltas should use OpenSpec-style sections:

```markdown
## ADDED Requirements

### Requirement: Export filtered report

#### Scenario: User exports filtered report

- GIVEN filtered report rows are visible
- WHEN the user exports the report
- THEN the exported file contains only filtered rows
```

Use `MODIFIED`, `REMOVED`, or `RENAMED` sections when changing existing
behavior.

## Implementation Rule

Do not implement production Java code until:

1. an OpenSpec change exists,
2. `proposal.md`, `tasks.md`, and spec deltas are written,
3. validation has passed or CLI absence has been reported,
4. the user approved implementation.

After implementation, run relevant Gradle tests:

```bash
./gradlew test
```

Prefer module tests when the module is clear:

```bash
./gradlew :module:test
```

Then update `tasks.md` checkboxes and report the result.
