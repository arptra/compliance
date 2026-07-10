# Assess Repository For OpenSpec

Use this prompt to inspect a Java/Gradle repository before initializing or using
OpenSpec.

## Inspect Only Lightweight Markers

Read only:

- `settings.gradle`, `settings.gradle.kts`
- `build.gradle`, `build.gradle.kts`
- `gradle.properties`
- `gradlew`
- `gradle/wrapper/gradle-wrapper.properties`
- `AGENTS.md`
- `openspec/project.md`
- `openspec/specs/`
- `openspec/changes/`
- `openspec/error-kb/index.yml`

If the OpenSpec CLI exists, run:

```bash
openspec list
openspec list --specs
```

Do not install tools without asking.

## Determine State

Return one:

- `UNINITIALIZED_OPENSPEC`
- `OPENSPEC_READY`
- `OPENSPEC_ACTIVE_CHANGES`
- `OPENSPEC_NEEDS_REFRESH`

## Output

```text
# OpenSpec Repository Assessment

## State

## Gradle / Java Snapshot

## OpenSpec Files Found

## OpenSpec Files Missing

## Active Changes

## Existing Specs

## Recommended Next Action

## Menu
```

Use a simple menu. Do not make changes in this step.
