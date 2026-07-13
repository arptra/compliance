# Create Complete Current-State Capability Specs

Use as a resumable synthesis phase of `FULL_BOOTSTRAP` or for affected
capabilities during refresh. Read `00-global-rules.md` first. Do not change
production code.

## Input Contract

Process completed domain and artifact findings from
`openspec/bootstrap/findings/`. Each synthesis work item must name:

- bounded context or module shard
- candidate capabilities
- evidence files
- related contracts, tests, data, events, and cross-cutting findings
- related command IDs and command-interface findings
- known overlaps with other work items

Do not impose an arbitrary limit on total capability specs. Limit each worker
or synthesis wave by safe context size, checkpoint it, and continue until the
queue is exhausted.

## Capability Boundaries

A capability represents observable user or system behavior, not every class or
private helper. Split a capability when it has a distinct actor, lifecycle,
contract, owner, state machine, or independently changeable behavior. Merge
duplicate names only after comparing evidence.

Use stable lowercase kebab-case capability IDs. Prefer domain-qualified IDs,
for example `billing-invoice-lifecycle`, to avoid collisions in monorepos.

## Required Spec Shape

Create or update:

```text
openspec/specs/<capability-id>/spec.md
```

Each spec must include:

```markdown
# <Capability> Specification

## Identity

- Capability ID:
- Domain:
- Status:
- Owners: UNKNOWN when not evidenced

## Purpose And Scope

## Inputs And Outputs

## States And Transitions

## Requirements

### Requirement: <observable behavior>

- Requirement ID: <stable ID>
- Evidence state: <allowed evidence state>

#### Scenario: <scenario name>

- GIVEN ...
- WHEN ...
- THEN ...

## Failure And Edge Behavior

## Data, Events, And External Contracts

## Commands And Operator Interfaces

## Cross-Cutting Requirements

## Dependencies

## Evidence

## Unknowns And Contradictions
```

Omit an empty state-machine section only when the capability is genuinely
stateless; state that explicitly. Every requirement needs at least one scenario
and one evidence reference. Multiple evidence states may be recorded, with the
strongest first.

## Traceability Updates

For every synthesized capability update:

- `openspec/index/capabilities.yml`
- `openspec/index/traceability.yml` or its declared shard
- `openspec/index/commands.yml` links for invokable behavior
- relevant file-manifest analysis statuses
- `openspec/index/unknowns.yml`
- `openspec/index/contradictions.yml`
- bootstrap work-item state

Traceability must support both directions:

```text
capability -> requirements -> contracts/commands/code/tests/dependencies
repository artifact -> capabilities/requirements
```

Only the coordinator writes canonical files. Parallel workers submit findings,
not direct spec edits.

## Output Per Wave

```text
# Capability Synthesis Progress

Work items completed: <count>
Specs created: <count>
Specs updated: <count>
Evidence links added: <count>
Remaining synthesis items: <count>
Declared gaps added: <count>
```

Do not present this as final completion. Final status comes from the independent
coverage audit.
