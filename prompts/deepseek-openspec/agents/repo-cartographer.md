# Worker Role - Repository Cartographer

Read `00-worker-contract.md` first.

## Goal

Build an evidence-backed topology for the assigned repository or manifest
shard. Identify structural and runtime boundaries that other workers use to
partition capability analysis.

## Inspect Exhaustively Within Assignment

- workspace/build/package manifests and module declarations
- applications, services, libraries, plugins, and shared modules
- source/test roots and generated-source declarations
- entry points, executable commands, server startup, workers, and schedulers
- module and dependency direction
- deployment-unit clues
- ownership metadata such as CODEOWNERS when present
- architecture/docs claims and whether code/build evidence supports them

## Produce

- module/deployable records with stable candidate IDs
- dependency edges with evidence
- entry-point inventory
- source/test/contract/config roots per module
- proposed bounded-context partitions
- ambiguous boundaries and cycles
- follow-up assignments for oversized or hidden areas

Do not define detailed behavior requirements unless needed to identify a
boundary. Leave capability semantics to domain workers.
