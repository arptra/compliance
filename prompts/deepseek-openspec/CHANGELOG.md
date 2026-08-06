# Changelog

All notable user-visible changes to the DeepSeek OpenSpec prompt pack are
recorded here. The format follows Keep a Changelog, and prompt-pack versions
follow Semantic Versioning.

## [Unreleased]

### Changed

- Added an OpenSpec 1.x compatibility gate: repositories with
  `openspec/config.yaml` keep the official OPSX `specs/` and `changes/` layout
  instead of receiving the prompt pack's legacy extended bootstrap tree.
- Relabeled the pack's `project.md`/`index/`/`bootstrap/` tree as a legacy
  extension rather than canonical OpenSpec structure.

## [2.2.0] - 2026-07-23

### Added

- Immediate structured logging for every explicit HTTP 429, including worker,
  sanitized request ID, and Retry-After when available.
- A user-visible scheduler message stating that parallel dispatch stopped and
  requests entered one global FIFO queue.
- Persistent rate-limit scheduler fields for resumable work queues and Git
  ticket queues.
- FIFO and single-running-ticket enforcement in the bundled Git history parser
  while the rate-limit latch is active.

### Changed

- An explicit HTTP 429 now switches all request types from parallel dispatch to
  `GLOBAL_SERIAL_QUEUE` with concurrency `1` for the rest of the run.
- Workers return rate-limit metadata instead of starting independent retry
  loops; the coordinator alone honors Retry-After/backoff and dispatches retries.
- A restarted CLI preserves an unfinished serial queue instead of opening a new
  parallel wave.

## [2.1.0] - 2026-07-13

### Added

- Exact-prefix ticket discovery from local Git commit messages, Git notes,
  reflogs, and branch/tag/ref names.
- A dependency-free local parser with persistent ticket records, queue state,
  signatures, bounded context windows, and stale detection.
- Cost-aware parallel ticket analysis with dedicated history, current-code,
  tests/contracts, and independent audit roles.
- An interactive workflow for showing one ticket with an evidence-backed
  description, historical diff, current implementation, tests, contracts, and
  later evolution.
- Restart-safe reuse of completed ticket analyses when their Git signatures are
  unchanged.
- Migration `add-git-ticket-history-v1` for installing the workflow over an
  existing OpenSpec repository without restarting full SDD bootstrap.

### Changed

- The main interactive menu now includes Git ticket history operations.
- Prompt-pack fingerprinting now includes bundled Python scripts and Markdown
  templates.
- Refresh, resume, and validation workflows now understand optional Git ticket
  history without treating it as a current-state SDD coverage requirement.

### Security

- Git ticket indexing remains local-only and does not fetch remotes or contact
  an issue tracker.
- Persistent ticket records exclude full commit messages and patches; raw Git
  evidence is loaded only into bounded analysis contexts.
