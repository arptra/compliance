# Create A New OpenSpec Change

Use when the user describes a feature, behavior change, removal, migration, or
refactor. Do not change production code in this step.

Read `00-global-rules.md`. A task context packet from
`13-build-task-context.md` is required for non-trivial work.

## Simple User Input

If the user has not described the change, ask for one or two sentences. Do not
ask them to choose modules, capability IDs, build tasks, or technical scope.

Resolve technical scope from indexes and evidence. Ask one follow-up question
only when product intent materially changes the resulting behavior and cannot
be derived safely.

## Context Gate

Read:

- the generated task context packet
- affected current-state capability specs
- linked architecture and cross-cutting sections
- directly affected contracts, schemas, source, and tests
- active change names to avoid collisions

If packet result is `INSUFFICIENT`, expand it or finish the relevant bootstrap
work before drafting an implementation-ready change.

If result is `SUFFICIENT_WITH_DECLARED_GAPS`, list those gaps in the proposal
and do not hide the associated risk.

## Create

Choose a unique verb-led kebab-case change ID:

```text
add-...
change-...
remove-...
refactor-...
migrate-...
```

Create:

```text
openspec/changes/<change-id>/
  proposal.md
  tasks.md
  context.yml
  specs/<capability-id>/spec.md
```

Copy or reference the task context manifest as `context.yml`. Create
`design.md` for broad, risky, cross-module, data-model, API, auth, migration,
concurrency, reliability, performance-sensitive, or ambiguous work.

## `proposal.md`

Include:

- problem and desired outcome
- affected capability and requirement IDs
- proposed behavior
- non-goals
- compatibility and migration impact
- security, data, reliability, and operational impact
- evidence and context-packet reference
- risks
- declared context gaps
- open product questions

## `tasks.md`

Create small verifiable tasks. Each implementation task should fit one task
context packet and identify affected capability IDs. Include tests, validation,
traceability refresh, and documentation/archive work.

Example:

```markdown
## Tasks

- [ ] 1. Add contract and behavior tests for CAP-REQ-ID.
- [ ] 2. Implement the behavior within the documented boundary.
- [ ] 3. Run targeted repository-native tests.
- [ ] 4. Validate OpenSpec and traceability.
- [ ] 5. Refresh affected current-state indexes after archiving.
```

## Spec Delta

Use OpenSpec sections:

- `## ADDED Requirements`
- `## MODIFIED Requirements`
- `## REMOVED Requirements`
- `## RENAMED Requirements`

Every requirement has a stable ID and at least one GIVEN/WHEN/THEN scenario.
Preserve current-state evidence separately from proposed intent.

## Parallel Analysis

For broad changes, use available subagents in parallel for non-overlapping,
read-only impact analysis such as contracts, tests, data/migrations, security,
and reverse dependencies. Each returns findings to the coordinator; only the
coordinator writes the OpenSpec change.

Do not claim subagents were used when the CLI does not provide them.

## Validate And Ask Once

Execute `05-add-openspec-validation.md` for the new change. Then report a short
summary and ask one question:

```text
The change specification is ready. Start implementation? Reply YES or NO.
```

Do not implement before `YES`.
