# Answer A Repository Question With Verified Evidence

Use whenever the user asks a factual question about the current repository,
including behavior, architecture, configuration, APIs, data, ownership, build
tasks, CLI commands, or exact command parameters.

Read `00-global-rules.md` first. Match the user's language. Do not change
production code or current-state specs while answering.

## Hard Grounding Rule

Never state a repository-specific fact from memory, naming conventions,
framework habits, or plausibility. A precise answer is allowed only when every
material claim has evidence reopened and checked during the current answer.

Allowed final statuses:

```text
VERIFIED_FROM_SPEC
VERIFIED_FROM_CODE
VERIFIED_FROM_RUNTIME
VERIFIED_FROM_TEST
CONFLICTING_EVIDENCE
NOT_VERIFIED
```

`INFERRED_FROM_CODE` may appear as a clearly labeled hypothesis in an
explanation, but it must never answer an exact question such as which flag,
parameter, default, environment variable, route, task, or command to use.

## Classify The Question

Choose one or more:

```text
COMMAND_OR_PARAMETER
CONFIG_OR_ENVIRONMENT
API_OR_CONTRACT
CURRENT_BEHAVIOR
ARCHITECTURE_OR_DEPENDENCY
DATA_OR_MIGRATION
BUILD_OR_TEST_TASK
OWNERSHIP_OR_LOCATION
RECOMMENDATION_OR_HYPOTHESIS
```

For repository facts, use `13-build-task-context.md` in
`REPOSITORY_QUESTION` mode. For exact command questions, always include the
command index and command-definition evidence.

When native subagents exist, dispatch
`agents/repository-evidence-retriever.md` for retrieval and keep final answer
ownership with the coordinator. Without native subagents, perform the same
retrieval as a separate bounded coordinator pass before drafting claims.

## Evidence Search Order

1. Search fresh OpenSpec capability, command, architecture, and traceability
   indexes.
2. Follow their evidence links and verify that referenced hashes/revisions are
   current.
3. Reopen the exact source, test, schema, build, config, or contract location.
4. If the index is missing or stale, use deterministic path, symbol, and text
   search to locate the authoritative declaration.
5. Use safe runtime introspection such as `--help`, `help`, `list`, or build
   task listing when it is side-effect-free and needed to resolve dynamic
   registration.
6. Use Git history selectively for renamed commands, removed parameters, or
   decision rationale; never present deleted behavior as current.

Search documentation as supporting evidence, not as the only authority when
current executable declarations are available.

## Exact Command And Parameter Gate

For every command, subcommand, positional argument, flag, alias, default,
choice, environment variable, config key, or precedence rule:

- the exact token must appear in a fresh command index record and its linked
  declaration, or in captured current runtime help;
- required/optional status, type, default, and allowed values each need their
  own evidence when stated;
- do not reconstruct a flag from variable names, method names, conventions, or
  similar commands;
- do not combine fragments from different versions or modules;
- do not silently substitute a similar parameter;
- for dynamically registered interfaces, runtime help or equivalent
  introspection is required when static code does not expose the final token.

Safe introspection must not execute the business action, write data, access
production, install dependencies, or reveal secrets. If help itself has side
effects or cannot run safely, rely on source evidence or return `NOT_VERIFIED`.

## Claim-Level Grounding Gate

Before answering:

1. Draft a structured answer using
   `templates/grounded-answer.template.yml` in memory or in a temporary context
   packet for a complex/reusable question.
2. Split the answer into atomic factual claims.
3. Attach one or more evidence records to each claim.
4. Reopen every cited location during this turn.
5. For exact tokens, confirm literal presence in the evidence excerpt or
   current runtime output.
6. Dispatch `agents/grounding-validator.md` as an independent read-only pass
   when native subagents exist. Otherwise perform a separate coordinator
   validation pass using the same contract.
7. Remove unsupported claims and search again.
8. If evidence remains insufficient, return `NOT_VERIFIED` rather than a
   plausible answer.

The retriever that found evidence must not self-certify it when an independent
worker is available.

## Conflicts

If fresh OpenSpec, code, tests, runtime help, or docs disagree:

- return `CONFLICTING_EVIDENCE`;
- show the conflicting values with sources;
- identify which source reflects executable current behavior when provable;
- do not silently resolve business intent;
- offer the nearest code/spec locations that require correction.

## Answer Format For Verified Facts

Keep it concise:

```text
Status: <VERIFIED status>

Answer: <direct answer>

Evidence:
- <path and symbol/line>: <what it proves>

Verification command: <safe command, only when actually verified>
```

Do not show a verification command that was invented or not found in evidence.

## Answer Format When Not Verified

```text
Status: NOT_VERIFIED

I could not confirm an exact answer from the current specification, code,
tests, contracts, or safe runtime introspection.

Searched:
- <specific indexes, paths, symbols, and safe commands checked>

Closest locations:
- <path/symbol and why it may be relevant>

Unverified candidates:
- <candidate explicitly labeled as not an answer>
```

Do not fill the answer field with a guess after declaring `NOT_VERIFIED`.
