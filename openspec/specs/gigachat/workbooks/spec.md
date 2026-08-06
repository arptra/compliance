# GigaChat Workbooks Specification

## Purpose

Определяет загрузку CSV/XLSX/ZIP данных, выбор листа, работу с табличными строками и Excel-экспорт.

## Requirements

### Requirement: Загрузка workbook
Авторизованный пользователь SHALL иметь возможность загрузить поддерживаемый workbook и получить upload id, формат, листы, колонки и preview rows.

#### Scenario: Обычная загрузка
- **WHEN** пользователь отправляет поддерживаемый файл на `/api/gigachat/lab/workbooks/upload`
- **THEN** API сохраняет upload в workspace, регистрирует raw records и возвращает metadata

#### Scenario: ZIP с CSV
- **WHEN** загруженный ZIP содержит поддерживаемый CSV
- **THEN** backend извлекает и разбирает данные по текущим правилам encoding/delimiter

### Requirement: Chunked upload
Lab SHALL использовать chunked upload для больших файлов и SHALL предоставлять status/cancel для upload task.

#### Scenario: Завершение всех chunks
- **WHEN** все заявленные chunks приняты и клиент вызывает complete
- **THEN** backend ставит сборку/разбор в задачу и возвращает её id

#### Scenario: Отмена upload
- **WHEN** пользователь отменяет session или task
- **THEN** backend помечает процесс отменённым и не публикует незавершённый workbook как готовый

### Requirement: Выбор листа
Пользователь SHALL выбирать sheet загруженного workbook и получать его колонки, total rows и ограниченный набор строк.

#### Scenario: Выбор существующего листа
- **WHEN** клиент отправляет sheet name и row limit
- **THEN** API возвращает данные этого листа и стабильные row indexes

### Requirement: Выбор строк
Frontend SHALL позволять выбирать отдельные, отфильтрованные или все доступные строки без изменения исходных данных.

#### Scenario: Большая таблица
- **WHEN** workbook содержит много строк и колонок
- **THEN** таблица использует виртуализацию/ограничение rendering так, чтобы выбор строки не требовал полного DOM списка

### Requirement: Экспорт workbook rows
Пользователь SHALL экспортировать выбранные строки в Excel, сохраняя исходные колонки и добавляя rule/model annotation columns без конфликтов имён.

#### Scenario: Конфликт имени колонки
- **WHEN** исходный workbook уже содержит имя annotation column
- **THEN** exporter выбирает безопасное отдельное имя и не перезаписывает исходную колонку

#### Scenario: Недопустимые Excel символы
- **WHEN** ячейка содержит символы, запрещённые форматом Excel
- **THEN** exporter очищает значение и завершает выгрузку
