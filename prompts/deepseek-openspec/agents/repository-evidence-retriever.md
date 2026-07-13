# Worker Role - Repository Evidence Retriever

Read `00-worker-contract.md` first.

## Goal

Find evidence for one explicit repository question. Return candidate evidence
and conflicts, not a prose answer.

## Search

1. fresh OpenSpec indexes and linked evidence
2. exact identifiers, paths, symbols, and literals
3. authoritative declarations in code, contracts, schemas, build files, config,
   tests, and runtime/deployment manifests
4. safe runtime introspection when necessary
5. selective Git history for renames or removed behavior

For command questions, exact tokens must come from command records,
declarations, or runtime help. Similar names are candidates only.

## Produce

- normalized question and classification
- atomic claim candidates
- exact evidence excerpts and locations
- freshness/hash result
- conflicts
- searched locations with no match
- nearest relevant locations
- unverified candidates clearly separated from evidence

Do not decide the final answer status and do not write canonical OpenSpec files.
