# Proposal: Settings, Background Tasks, And Exports

> Retrospective reconstruction from Git; this was not the original proposal.

## Intent

Turn Lab configuration and workbook processing into persistent, repeatable workflows with background execution and Excel export.

## Historical Evidence

- `f9a0c33`: background tasks and settings versions.
- `e069f75`, `7cf836d`: rule validation, tabs, and loading feedback.
- `1074daa`, `56360ea`, `63b7c8e`: persisted rule/settings edits and settings version updates.
- `0bc3da3`, `77183fc`, `5d77aa0`, `5c4db04`: workbook/selected-row exports and control performance.
- `b5bee0e`, `2da03e9`: removal of obsolete tag panel and final-prompt save action.

## Outcome

Users could save versioned settings, queue labeling, inspect task state, and export selected rows while the UI removed redundant settings actions.
