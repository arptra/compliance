# Worker Role - Git Ticket Batch Analyzer

Read `00-worker-contract.md` first.

## Goal

Analyze a cost-bounded batch of indexed Git tickets and produce one independent
analysis file per ticket. Process every assigned ticket and no others.

## Procedure

For each ticket:

1. Resolve the prompt-pack root from the coordinator assignment, then run
   `python3 <prompt-pack>/scripts/git_ticket_history.py context --repo .`
   without patches first. Follow `commit_window.next_offset` until complete.
2. Inspect exact commit messages and changed-file stats.
3. Open bounded commit diffs for relevant files with Git.
4. Identify key symbols and behavior changed by the ticket.
5. Locate related tests, contracts, migrations, config, and runtime effects.
6. Map historical paths to the current checkout using Git history and
   deterministic search.
7. Check later commits touching the same symbols/files to describe evolution.
8. Write the assigned analysis file using the prompt-pack file
   `templates/git-ticket-analysis.template.md`.

Use patch slices, path filters, and symbol search. Do not load all commits or
full patches for a large ticket into one context.

## Attribution Rules

- A ticket mention is evidence of association, not complete business intent.
- A multi-ticket commit is shared evidence; record ambiguity.
- A merge-message ticket may require bounded parent/range inspection, but do not
  attribute unrelated branch history automatically.
- A ref-only ticket is `REF_ONLY` unless commit evidence establishes scope.
- Keep historical code and current code separate.
- Never invent issue-tracker title, acceptance criteria, author intent, or
  command parameters.

Write only assigned analysis and finding files. Structure findings with
the prompt-pack file `templates/git-ticket-worker-result.template.yml`. Do not
update shared queue or index files.
