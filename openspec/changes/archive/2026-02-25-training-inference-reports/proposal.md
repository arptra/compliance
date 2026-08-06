# Proposal: Training And Inference Reports

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Stabilize Parquet-based local model training/inference and make category/subcategory quality visible in reports.

## Historical Evidence

- `6572055`, `0348bae`: mixed-object Parquet normalization fixes.
- `17f4ead`, `249b8dd`, `a487d42`, `9e13002`: report visuals, diagnostics, subcategories, and validation window.
- `21dc77f`: taxonomy-constrained inferred subcategories.

## Outcome

Local model artifacts and reports became robust to mixed Excel types and taxonomy drift, with localized category and subcategory diagnostics.
