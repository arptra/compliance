# GigaChat Transports Specification

## Purpose

Определяет выбор, проверку и использование mTLS и token-транспортов GigaChat, включая модель, диагностику и ограничение частоты.

## Requirements

### Requirement: Статус транспортов
API SHALL возвращать статус mTLS и token транспортов, их configured/ready признаки, адреса и проверяемые артефакты без раскрытия секретов.

#### Scenario: Просмотр статуса
- **WHEN** frontend запрашивает `/api/gigachat/status`
- **THEN** API возвращает configured mode, текущую модель и список доступных транспортов

### Requirement: Проверка выбранного транспорта
Пользователь SHALL иметь возможность выполнить probe выбранного транспорта и получить сообщение API и список доступных моделей.

#### Scenario: Успешный token probe
- **WHEN** token transport настроен и OAuth/API отвечают успешно
- **THEN** probe возвращает `ok=true` и доступные модели для selector

#### Scenario: Ошибка конфигурации
- **WHEN** обязательный key, certificate или endpoint отсутствует
- **THEN** probe возвращает проверяемое сообщение об отсутствующем условии, не выдумывая успешную готовность

### Requirement: Выбранная модель
Lab SHALL отправлять запросы с моделью из текущих settings, а UI SHALL предлагать модели, подтверждённые probe/API, когда список доступен.

#### Scenario: Модель из профиля
- **WHEN** пользователь запускает разметку с сохранённым settings profile
- **THEN** backend использует модель этого профиля при формировании chat completion запроса

### Requirement: Структурированный HTTP вызов
Транспорт SHALL отправлять chat completion через httpx и SHALL нормализовать поддерживаемый структурированный ответ в Lab response contract.

#### Scenario: Компактный JSON модели
- **WHEN** GigaChat возвращает допустимый компактный JSON
- **THEN** backend приводит его к ожидаемой схеме разметки

#### Scenario: Неуспешный HTTP ответ
- **WHEN** GigaChat возвращает неуспешный статус, не исчерпанный retry policy
- **THEN** transport передаёт ответ adaptive limiter для решения о повторе

### Requirement: Общий adaptive limiter
Все запросы к одному API base URL SHALL координироваться общим limiter, чтобы HTTP 429 был виден всем параллельным workers.

#### Scenario: Лимит для общего endpoint
- **WHEN** один worker получает HTTP 429
- **THEN** остальные workers этого limiter учитывают общую очередь и retry boundary
