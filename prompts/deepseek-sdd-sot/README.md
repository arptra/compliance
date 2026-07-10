# DeepSeek SDD / SoT Prompt Pack

This folder contains prompts for running a local model such as `deepseek-v4-flash-262k`
inside an existing Java repository to create a Spec-Driven Development (SDD) and
Source-of-Truth (SoT) foundation.

For daily use, start with `START_HERE.md`.

Use the numbered files only when you want to run a specific step manually.
If your CLI session does not preserve context between runs, prepend
`00-global-rules.md` to every numbered prompt.

## Is This Enough?

Yes, this prompt pack is enough to create an initial SDD/SoT foundation for a Java
repository if it is executed in order and the model can inspect the repo.

It is not enough to create approved business requirements by itself. Existing
behavior can only be documented as `INFERRED_FROM_CODE`; missing intent must stay
`UNKNOWN` and go to `docs/sot/open-questions.md`.

## Recommended Order

### Normal Daily Workflow

Use one file:

```bash
cat /path/to/prompts/deepseek-sdd-sot/START_HERE.md
```

Paste that into DeepSeek CLI inside the target Java repo.

The prompt will:

1. detect whether the repo is already initialized for SDD/SoT;
2. ask whether to initialize if it is not initialized;
3. ask for the feature/mode if it is initialized;
4. load only context-map, context-packs, and the active spec;
5. avoid rereading thousands of files;
6. use Gradle/`./gradlew`;
7. run unit tests after implementation and fix task-related failures.

### Manual Foundation Workflow

1. `00-global-rules.md`
2. `01-assess-repository.md`
3. `02-create-sot-foundation.md`
4. `03-fill-architecture-sot.md`
5. `04-create-current-state-specs.md`
6. `05-add-sot-validation.md`
7. `06-final-review.md`

For future feature work use:

- `07-new-feature-spec.md`
- `08-implement-spec-task.md`

For fresh CLI sessions and context maintenance use:

- `09-resume-session.md`
- `10-refresh-context-pack.md`

## How To Apply In A Java Repo

### 1. Prepare The Java Repo

Open a terminal in the root of the target Java repository.

Make sure the working tree is clean or intentionally contains only changes you
want DeepSeek to see:

```bash
git status
```

Do not run these prompts from this prompt-pack repository. Run them from the Java
repository that needs SDD/SoT.

### 2. Run The Assessment First

Send `00-global-rules.md` and then `01-assess-repository.md` to DeepSeek CLI.

If your CLI keeps one continuous conversation, send them as two messages in the
same session.

If your CLI starts a fresh model context for every command, concatenate them:

```bash
cat /path/to/prompts/deepseek-sdd-sot/00-global-rules.md \
  /path/to/prompts/deepseek-sdd-sot/01-assess-repository.md
```

Expected result: DeepSeek prints a repository assessment and does not edit files.

Human check before moving on:

- Did it identify Maven or Gradle correctly?
- Did it identify Spring Boot or the actual framework correctly?
- Did it list real modules and packages?
- Did it mark unknown business intent as `UNKNOWN`?
- Did it avoid editing files?

If the assessment is wrong, correct it in the next prompt before continuing.

### 3. Create The Foundation

After the assessment is acceptable, send `02-create-sot-foundation.md`.

If the CLI does not preserve context, prepend `00-global-rules.md` again:

```bash
cat /path/to/prompts/deepseek-sdd-sot/00-global-rules.md \
  /path/to/prompts/deepseek-sdd-sot/02-create-sot-foundation.md
```

Expected result: DeepSeek creates the initial SoT/SDD folders and templates.

Human check:

- `AGENTS.md` is short and practical.
- `docs/sot/open-questions.md` contains unknowns instead of invented answers.
- `specs/_template/` files are concise and usable.
- No production Java code changed.

### 4. Fill Architecture From Evidence

Send `03-fill-architecture-sot.md`.

Expected result: architecture docs describe the actual repository, not an ideal
generic Java service.

Human check:

- Important statements link to real packages, modules, configs, or tests.
- `INFERRED_FROM_CODE` is used for inferred behavior.
- `UNKNOWN` is used where business intent is missing.
- ADRs are short and decision-focused.

### 5. Create Current-State Specs

Send `04-create-current-state-specs.md`.

Expected result: up to 5 as-is specs under `specs/current-state/`.

Human check:

- Specs describe existing code behavior only.
- Every requirement is marked `INFERRED_FROM_CODE`.
- Missing tests are marked `MISSING_TEST`.
- Missing contracts are marked `MISSING_CONTRACT`.

### 6. Add Validation Scripts

Send `05-add-sot-validation.md`.

Expected result: lightweight scripts under `scripts/sot/`.

Run the scripts manually after creation. The exact commands should be documented
by DeepSeek in `scripts/sot/README.md`, but they will usually look like:

```bash
python3 scripts/sot/check-spec-format.py
python3 scripts/sot/check-traceability.py
python3 scripts/sot/check-req-id-in-changes.py
```

Human check:

- Scripts are readable and safe.
- Scripts do not require heavy dependencies.
- Scripts report clear failures.
- CI integration is documented, not forced, unless obvious and safe.

### 7. Final Review

Send `06-final-review.md`.

Expected result: a final report and only small cleanup edits to generated docs.

Human check:

- No production code changed.
- No generic filler remains.
- Open questions are visible.
- The next feature workflow is clear.

### 8. Commit Manually

These prompts intentionally tell DeepSeek not to commit or push.

After human review, commit manually:

```bash
git status
git diff
git add AGENTS.md .ai docs specs scripts/sot
git commit -m "Add SDD source of truth foundation"
```

Push only after reviewing the generated files.

## How To Use For A New Feature

Use `07-new-feature-spec.md` before coding.

Replace:

```text
<PASTE BUSINESS REQUEST HERE>
```

with the real business request, then send the prompt to DeepSeek.

Expected result: a new folder like:

```text
specs/REQ-2026-001-short-name/
```

DeepSeek should fill intake, requirements, acceptance criteria, design, tasks,
traceability, test plan, and changelog.

Do not code until `Ready For Implementation: YES`.

## How To Implement One Task

After a feature spec is ready, use `08-implement-spec-task.md`.

Fill in:

```text
Task ID: <TASK-ID>
Requirement ID: <REQ-ID>
Spec folder: specs/<REQ-FOLDER>/
```

Run one task per prompt. This keeps context small and makes changes reviewable.

Expected result:

- relevant tests added or updated
- smallest safe production change
- `05-traceability.yml` updated
- relevant tests run
- no unrelated refactoring

## How To Resume After Closing DeepSeek CLI

Do not ask DeepSeek to read the whole repo again.

Open the Java repo and send:

```bash
cat /path/to/prompts/deepseek-sdd-sot/00-global-rules.md \
  /path/to/prompts/deepseek-sdd-sot/09-resume-session.md
```

Then tell it the active work item:

```text
Continue spec: specs/REQ-2026-001-short-name/
Current task: TASK-REQ-2026-001-02
```

Expected behavior:

- DeepSeek reads only the SoT bootstrap files.
- DeepSeek reads `.ai/context-map.yml`.
- DeepSeek reads only relevant `.ai/context-packs/*`.
- DeepSeek reads the active spec folder.
- DeepSeek does not scan thousands of source files.
- DeepSeek asks before loading extra source files.

If DeepSeek says it needs the whole repo, stop it and rerun `09-resume-session.md`
with the active spec/task stated explicitly.

## How To Keep Context Packs Fresh

If the repo changed and a context pack is stale, use `10-refresh-context-pack.md`.

Fill:

```text
<CONTEXT_PACK_NAME>
<WHY THIS PACK NEEDS UPDATING>
```

Example:

```text
Context pack to refresh:
payments

Reason for refresh:
Payment status workflow changed in the last feature branch.
```

This updates one small pack instead of forcing DeepSeek to rediscover the whole
repository.

## Practical Tips For DeepSeek CLI

- Keep one CLI session for prompts 1-6 if possible.
- If the session resets, prepend `00-global-rules.md` to the current prompt.
- On every new session after the foundation exists, start with
  `09-resume-session.md`.
- If DeepSeek invents business meaning, stop and ask it to replace invented text
  with `UNKNOWN`.
- If DeepSeek edits production Java code during prompts 1-6, reject that output
  and rerun with a stricter reminder.
- If generated docs are too large, ask it to compress them instead of adding more
  files.
- If it cannot inspect the repository, do not continue. The prompts rely on repo
  evidence.

## Operating Rules

- Run prompts inside the target Java repo root.
- Do not ask the model to commit or push from these prompts.
- Review generated docs before using them as governance.
- Keep generated files small and maintainable.
- Treat chat history as temporary; repository files are the source of truth.

## Expected Result

After prompts 1-6, the Java repo should contain:

- agent guidance in `AGENTS.md`
- SoT docs under `docs/sot/`
- architecture docs under `docs/architecture/`
- ADR docs under `docs/adr/`
- feature spec templates under `specs/_template/`
- optional current-state specs under `specs/current-state/`
- lightweight SoT validation scripts under `scripts/sot/`
