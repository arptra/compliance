# GigaChat Labeling Specification

## Purpose

Определяет локальную проверку правил, формирование prompt, модельную разметку, нормализацию tags и переклассификацию строк.

## Requirements

### Requirement: Локальная оценка rule packs
Lab SHALL оценивать строки локальными rules до обращения к модели и возвращать rule hits, matched keywords/fields и suggested actions/topics.

#### Scenario: Совпадение keyword
- **WHEN** значение разрешённого source field удовлетворяет keyword rule
- **THEN** результат строки содержит идентификатор правила и проверяемое evidence совпадения

#### Scenario: Нет совпадений
- **WHEN** ни одно активное правило не совпало
- **THEN** строка получает пустые rule hits без выдуманного локального tag

### Requirement: Prompt preview
Lab SHALL строить финальный prompt из settings, доступных колонок, локального результата и row data и SHALL позволять получить preview без вызова модели.

#### Scenario: Preview
- **WHEN** пользователь запрашивает final prompt
- **THEN** API возвращает сгенерированный prompt и timestamp, не выполняя разметку строки

### Requirement: Разметка строки
Lab SHALL отправлять выбранную строку через выбранный transport и возвращать нормализованную классификацию, tags, evidence и решения по rules.

#### Scenario: Модель подтверждает локальный tag
- **WHEN** модель подтверждает предложенное rule совпадение
- **THEN** tag отражается в confirmed rule hits и итоговом наборе tags

#### Scenario: Структурированный tag response
- **WHEN** модель возвращает tags в одном из поддерживаемых структурированных представлений
- **THEN** backend нормализует их в стабильный список строк и decision fields

### Requirement: Переклассификация
Lab SHALL поддерживать отдельный reclassification flow на основе импортированных правил и одного или нескольких context fields.

#### Scenario: Найдена новая тема
- **WHEN** reclassification rule и модель определяют новую тему
- **THEN** результат содержит source classification, new class, reclassified topic, final topic и decision source

#### Scenario: Только переклассификация
- **WHEN** запрос устанавливает `reclassification_only=true`
- **THEN** backend выполняет reclassification contract без обязательного полного tagging flow

### Requirement: Validation export
Lab SHALL позволять экспортировать полный annotated результат и отдельный validation workbook для проверки решений.

#### Scenario: Экспорт завершённой выборки
- **WHEN** пользователь экспортирует обработанные rows
- **THEN** файл содержит source row и нормализованные local/model/reclassification поля
