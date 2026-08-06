# Record Lake Specification

## Purpose

Определяет workspace-изолированные Parquet-слои raw/rules/gigachat, поиск, фильтрацию, удаление и перенос строк обратно в Lab.

## Requirements

### Requirement: Три слоя записей
Lake SHALL хранить и искать записи в стадиях `raw`, `rules` и `gigachat` с системными provenance полями.

#### Scenario: Регистрация workbook
- **WHEN** workbook успешно загружен
- **THEN** его строки регистрируются в raw layer с record id, workspace, source filename, sheet, row number и ingestion metadata

### Requirement: Авторизованный поиск
`/api/records/search` SHALL требовать действующего пользователя, применять его workspace и возвращать columns, rows, total и engine.

#### Scenario: Pagination
- **WHEN** клиент задаёт limit и offset
- **THEN** API ограничивает limit диапазоном 1..1000 и возвращает соответствующую страницу

#### Scenario: Нет авторизации
- **WHEN** запрос не содержит действующего token
- **THEN** API возвращает HTTP 401

### Requirement: Поддерживаемые фильтры
Lake SHALL поддерживать операции `eq`, `ne`, `contains`, `in`, `between`, `gte`, `lte` и `exists_any` над доступными columns.

#### Scenario: Проверка наличия нескольких полей
- **WHEN** фильтр `exists_any` получает список field names
- **THEN** поиск возвращает rows, где существует хотя бы одно из указанных полей по правилам lake service

### Requirement: Удаление записей
Авторизованный пользователь SHALL удалять выбранные record ids или очищать текущий stage только внутри своего workspace.

#### Scenario: Удаление выбранных строк
- **WHEN** пользователь отправляет record ids на `/api/records/delete`
- **THEN** API возвращает количество удалённых rows и затронутых Parquet files

#### Scenario: Очистка слоя
- **WHEN** пользователь подтверждает `/api/records/clear` для stage
- **THEN** удаляются записи этого stage и workspace, но не другого workspace

### Requirement: Импорт lake selection в Lab
Frontend SHALL позволять выбрать найденные lake rows, убрать системные columns и открыть их в GigaChat Lab как временный workbook import.

#### Scenario: Импорт выбранного набора
- **WHEN** пользователь выбирает rows и запускает импорт в Lab
- **THEN** frontend сохраняет payload в sessionStorage и переходит на `/gigachat?import=lake`
