# OpenSpec SSOT

`openspec/specs/` is the single source of truth for current observable behavior. This file only routes a small model to the minimum useful context and to implementation evidence.

## Read Budget

For a question or change, read in this order:

1. This file.
2. One capability spec from the table below.
3. The listed implementation entry points.
4. Focused tests found with `rg`.

Do not load all specs. Do not read `changes/archive/` unless the question is historical. Do not treat the old prompt pack under `prompts/deepseek-openspec/` as repository SSOT.

## Active Runtime

The current web product is GigaChat Lab. Backend registration in `src/complaints_trends/api/app.py` mounts only `health`, `auth`, `gigachat`, and `records`. Frontend registration in `apps/dashboard/src/routes/index.tsx` exposes `/gigachat`, `/gigachat/lake`, `/gigachat/background`, and `/profile`, plus login and registration.

Python analytics pipelines and removed dashboard pages remain in Git/code for compatibility and history, but they are not current mounted web behavior unless a current spec says otherwise.

## Capability Index

| Need | Current spec | Primary implementation |
| --- | --- | --- |
| Routes, auth gate, navigation | `specs/system/application-shell/spec.md` | `apps/dashboard/src/routes/index.tsx`, `apps/dashboard/src/components/Layout.tsx` |
| Registration, login, profile, sessions | `specs/identity/user-access/spec.md` | `api/routers/auth.py`, `api/services/catalog_service.py`, `features/auth/` |
| GigaChat mTLS/token status and API calls | `specs/gigachat/transports/spec.md` | `api/services/gigachat_connection_service.py`, `gigachat_api/`, `gigachat_mtls.py` |
| Settings versions and rule packs | `specs/gigachat/settings-and-rules/spec.md` | `api/services/gigachat_lab_service.py`, `RulePackEditor.tsx`, `GigaChatSettingsForm.tsx` |
| Upload, sheets, table, and exports | `specs/gigachat/workbooks/spec.md` | `api/routers/gigachat.py`, `gigachat_lab_service.py`, `GigaChatPage.tsx`, `WorkbookSheetTable.tsx` |
| Local rules, model labels, reclassification | `specs/gigachat/labeling/spec.md` | `gigachat_lab_service.py`, `rulePackMatcher.ts`, `GigaChatPage.tsx` |
| Workers, background tasks, HTTP 429 queue | `specs/gigachat/background-processing/spec.md` | `gigachat_lab_service.py`, `gigachat_api/rate_limit.py`, background task pages |
| Parquet layers, search, delete, import | `specs/data/record-lake/spec.md` | `api/routers/records.py`, `parquet_lake_service.py`, `GigaChatLakePage.tsx` |

Backend paths in the table are relative to `src/complaints_trends/`; frontend paths are relative to `apps/dashboard/src/`.

## Frontend Map

- Bootstrap: `apps/dashboard/src/main.tsx`, `App.tsx`.
- Route truth: `apps/dashboard/src/routes/index.tsx`.
- Shared HTTP/auth behavior: `apps/dashboard/src/lib/api.ts`, `features/auth/`.
- Main orchestration: `apps/dashboard/src/pages/GigaChatPage.tsx`.
- Lake and background tasks: `GigaChatLakePage.tsx`, `GigaChatBackgroundTasksPage.tsx`.
- Reusable Lab UI: `apps/dashboard/src/features/gigachat/`.
- API DTO mirror: `apps/dashboard/src/features/gigachat/types.ts`.
- Verification: `npm run build --prefix apps/dashboard`.

## Backend Map

- Runtime composition: `src/complaints_trends/api/app.py`, `api/deps.py`.
- Public contracts: `api/routers/auth.py`, `api/routers/gigachat.py`, `api/routers/records.py`, `api/schemas.py`.
- Lab orchestration/persistence: `api/services/gigachat_lab_service.py`.
- Users, tokens, workspaces, settings catalog: `api/services/catalog_service.py` backed by `data/app.sqlite`.
- Record layers: `api/services/parquet_lake_service.py` under `data/lake/`.
- GigaChat transports: `api/services/gigachat_connection_service.py`, `gigachat_api/`, `gigachat_mtls.py`.
- Configuration: `src/complaints_trends/config.py`, `configs/project.yaml`, environment overrides.
- Verification: focused files in `tests/`, then `PYTHONPATH=src pytest -q` when risk warrants it.

## Normal GIGACODE Workflow

Open GIGACODE in the repository root and use `GIGACODE.md` as the project instruction file.

```text
/opsx-explore "как лучше добавить ..."
/opsx-propose "добавить одну конкретную возможность"
/opsx-apply <change-name>
/opsx-sync <change-name>
/opsx-archive <change-name>
```

The six `/opsx-*` command and skill templates in `.gigacode/` were generated from OpenSpec 1.8.0 and adapted for GIGACODE. Restart GIGACODE after pulling them. The OpenSpec CLI requires Node.js 20.19 or newer; then install it with `npm install -g @fission-ai/openspec@latest`.

For ordinary questions, use plain language. GIGACODE must follow the no-guessing path in `GIGACODE.md`: spec, code, schema/test evidence, then answer.

## Historical Archive

`changes/archive/` contains reconstructed completed changes grouped from Git history. Each proposal names its evidence commits. The archive explains evolution, including legacy analytics later removed from the mounted UI/API; it does not override current specs.
