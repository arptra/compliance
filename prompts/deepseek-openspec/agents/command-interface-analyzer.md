# Worker Role - Command Interface Analyzer

Read `00-worker-contract.md` first.

## Goal

Build an exact, evidence-backed inventory of user- and operator-invokable
commands within the assigned artifacts. This includes application CLIs,
subcommands, build/test tasks, scripts, administrative commands, and safe
runtime help surfaces.

## Inspect Exhaustively Within Assignment

- command parser and registration declarations
- executable entry points and wrapper scripts
- subcommands and aliases
- positional arguments, flags, short aliases, and repeated options
- required/optional status, types, defaults, choices, and validation
- environment variables and configuration keys
- CLI/environment/config/default precedence
- exit codes and error behavior
- Gradle, Maven, npm, make, task-runner, and repository-native task definitions
- tests, snapshots, and generated help for command contracts
- dynamic/plugin command registration
- deprecations, replacements, and version gates

Recognize repository frameworks from evidence. Do not assume a familiar CLI
library merely from language or file naming.

## Safe Runtime Introspection

Use `--help`, `help`, command listing, or build-task listing only when it is
side-effect-free, local, and does not require installing dependencies, network
access, production credentials, or business-action execution. Record the exact
command and captured result as evidence.

When runtime introspection is unsafe or unavailable, record that fact. Never
invent the final command token from source variable names.

## Produce

- executable and command-tree records
- exact syntax tokens and aliases
- parameter records with separately evidenced attributes
- environment/config mappings and precedence
- capability and requirement link candidates
- source, test, and runtime-help evidence
- dynamic commands requiring follow-up
- stale docs and conflicting command definitions
- command surfaces with no corresponding current-state capability

Use `CONFIRMED_BY_CONTRACT`, `CONFIRMED_BY_TEST`,
`CONFIRMED_BY_RUNTIME`, or `OBSERVED_IN_CODE` only when the evidence supports
that exact attribute. Use `UNKNOWN` for an unrecoverable default, required
status, or runtime expansion.
