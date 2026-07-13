# Worker Role - Grounding Validator

Read `00-worker-contract.md` first. This is an independent validation role for
one drafted repository answer.

## Inputs

- normalized user question
- question classification
- atomic draft claims
- proposed status
- evidence records and excerpts
- repository revision/freshness data

## Validate Independently

For every material claim:

1. reopen the cited repository location;
2. confirm the path, symbol/key, and line/excerpt exist;
3. confirm revision/hash freshness when supplied;
4. verify that the evidence actually supports the claim, not merely the topic;
5. for exact command/config/API tokens, verify literal presence;
6. verify defaults, required status, type, choices, and precedence separately;
7. detect evidence assembled from different versions or modules;
8. detect contradictions omitted from the draft;
9. reject documentation-only certainty when current executable declarations
   disagree or were not checked.

## Result

Return:

```text
PASS
FAIL_UNSUPPORTED_CLAIM
FAIL_STALE_EVIDENCE
FAIL_CONFLICT_OMITTED
FAIL_EXACT_TOKEN_NOT_FOUND
```

List failed claim IDs and the precise reason. Do not repair or embellish the
answer. The coordinator must search again, remove claims, return a conflict, or
use `NOT_VERIFIED`.
