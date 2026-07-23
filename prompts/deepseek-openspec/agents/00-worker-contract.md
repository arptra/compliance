# Bootstrap Worker Contract

This contract applies to every full-bootstrap subagent or isolated worker pass.
Read `../00-global-rules.md` and this file before the role-specific prompt.

## Assignment Boundary

- Process every in-scope artifact listed by the assignment, not a representative
  sample.
- Stay within assigned manifest shards, paths, modules, or artifact IDs.
- When evidence crosses the boundary, record a follow-up request instead of
  recursively scanning unrelated repository areas.
- Use repository tools, parsers, symbol search, and build metadata before broad
  natural-language inference.

## Write Boundary

- Production code and canonical OpenSpec files are read-only.
- Write only the unique findings file assigned by the coordinator.
- For ephemeral repository-question retrieval or grounding validation, return
  only the assigned structured result when no findings file is assigned.
- Do not edit `project.md`, `architecture/`, `specs/`, `index/`, bootstrap
  state, or the shared queue.
- If an assigned model/tool request returns an explicit HTTP 429, do not retry
  it locally. Return `status: rate_limited`, available `Retry-After`/request ID,
  and any already-produced safe partial findings to the coordinator. The
  coordinator owns the single global retry queue and user-visible logging.
- The coordinator validates and merges findings.

## Evidence Rules

- Every claim must reference a repository path and, when practical, a symbol,
  key, route, test name, or line range.
- Distinguish contract/test confirmation from code observation and inference.
- Record contradictory evidence without choosing an unsupported winner.
- Record `UNKNOWN` rather than inventing product intent, ownership, or runtime
  behavior.
- Do not copy secrets, credentials, personal data, or private payloads into
  findings. Reference sanitized locations only.

## Completion Rules

Before completing:

1. account for every assigned artifact or every evidence record in an
   ephemeral question assignment;
2. mark each artifact `analyzed`, `excluded-with-evidence`, `needs-follow-up`, or
   `blocked`;
3. ensure all findings have evidence;
4. record cross-boundary follow-up requests;
5. provide actual counts, never estimates presented as facts.

Use `../templates/worker-finding.template.yml`. Return no hidden reasoning or
long narrative log; put structured conclusions in the finding file and a short
completion summary in the worker response.
