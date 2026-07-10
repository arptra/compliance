# Fill OpenSpec Project Context

Use this prompt to improve `openspec/project.md`.

Do not change production code.

## Read

- `openspec/project.md`
- Gradle settings/build files
- only source files needed to identify modules, entry points, and test layout

Do not inspect more than 40 source files without explaining why.

## Update `openspec/project.md`

Add or refine:

- architecture overview
- module responsibilities
- package conventions
- test commands
- Java/Gradle/toolchain notes
- deployment/runtime notes if obvious from repo
- public API/contract locations
- known unknowns

Every inferred statement must be marked `INFERRED_FROM_CODE` when business
intent is not explicit.

## Output

```text
# OpenSpec Project Context Updated

## Sections Updated

## Evidence Used

## Unknowns

## Next Recommended Action
```
