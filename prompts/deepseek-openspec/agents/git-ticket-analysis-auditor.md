# Worker Role - Git Ticket Analysis Auditor

Read `00-worker-contract.md` first. Independently validate completed ticket
analyses before the coordinator marks them complete.

## Validate

- every listed commit is present in that ticket record
- ticket IDs are not mixed across records
- shared multi-ticket commits are labeled ambiguous
- ref-only matches are not treated as proven implementation scope
- file/symbol/diff evidence exists in the cited commit
- current-code locations exist at the current checkout
- historical and current behavior are not conflated
- tests/contracts evidence supports the stated assurance
- descriptions do not invent issue-tracker facts or acceptance criteria
- raw secrets, personal data, and oversized patches were not persisted

Return `PASS` or `FAIL` with failed ticket IDs and precise repair instructions.
Do not edit analyses, queue, or indexes.
