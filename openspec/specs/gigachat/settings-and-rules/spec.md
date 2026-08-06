# GigaChat Settings And Rules Specification

## Purpose

Определяет редактирование Lab settings, версионирование профилей и обмен локальными rule packs.

## Requirements

### Requirement: Текущие Lab settings
Lab SHALL загружать доступные setting options и текущие values и SHALL сохранять разрешённые изменения через API.

#### Scenario: Открытие Lab
- **WHEN** пользователь открывает `/gigachat`
- **THEN** frontend получает settings contract и строит поля из серверных options/values

#### Scenario: Сохранение изменений
- **WHEN** пользователь сохраняет settings
- **THEN** API валидирует values и возвращает фактически сохранённое состояние

### Requirement: Версии settings
Пользователь SHALL иметь возможность создавать, читать, обновлять и удалять версии settings с title, description, status и visibility.

#### Scenario: Создание версии от базовой
- **WHEN** пользователь создаёт новую версию с `base_version_id`
- **THEN** новая версия получает собственный id и копию применимых base values

#### Scenario: Удаление версии
- **WHEN** авторизованный пользователь удаляет разрешённую версию
- **THEN** API возвращает обновлённый список версий без удалённой записи

### Requirement: Пользовательское хранение профилей
Settings versions SHALL сохраняться в catalog с owner/workspace контекстом и SHALL соблюдать public/private visibility.

#### Scenario: Private profile
- **WHEN** версия имеет private visibility
- **THEN** она доступна владельцу в разрешённом workspace и не становится общим публичным профилем

### Requirement: Rule packs
Версия settings SHALL хранить rule packs, которые задают source fields, keywords и результат локального совпадения до модельного запроса.

#### Scenario: Частичное наличие source fields
- **WHEN** правило перечисляет несколько source fields и в строке присутствует только часть из них
- **THEN** matcher проверяет доступные поля и не отклоняет правило только из-за отсутствующих колонок

#### Scenario: Literal keyword spacing
- **WHEN** keyword содержит значимые пробелы
- **THEN** импорт/сохранение не меняет его литеральное значение незаметно для пользователя

### Requirement: Excel exchange для rule packs
Пользователь SHALL иметь возможность экспортировать rule packs версии в Excel и импортировать совместимый Excel обратно с серверной валидацией.

#### Scenario: Невалидный импорт
- **WHEN** workbook не содержит обязательной структуры rules
- **THEN** API возвращает ошибку и не заменяет сохранённые rule packs
