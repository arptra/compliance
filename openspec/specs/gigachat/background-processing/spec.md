# GigaChat Background Processing Specification

## Purpose

Определяет параллельную и фоновую обработку строк, прогресс, отмену и адаптивное поведение после HTTP 429.

## Requirements

### Requirement: Настраиваемая параллельность
Lab SHALL принимать `async_workers` и обрабатывать независимые строки одновременно вплоть до указанного безопасного количества workers.

#### Scenario: Восемь workers
- **WHEN** задача содержит не менее восьми строк и `async_workers=8`
- **THEN** backend создаёт до восьми concurrent row workers, пока limiter не требует иной режим

#### Scenario: Один worker
- **WHEN** `async_workers=1`
- **THEN** следующая строка начинает модельный вызов после освобождения единственного worker

### Requirement: Фоновые задачи
Пользователь SHALL запускать пакет строк как background task, получать status/progress/logs, отменять незавершённую задачу и загружать результат завершённой.

#### Scenario: Запуск задачи
- **WHEN** API принимает допустимый список строк
- **THEN** задача появляется со статусом `queued`, затем переходит в `running` и хранит worker count

#### Scenario: Завершение задачи
- **WHEN** все строки обработаны или учтены как ошибки
- **THEN** задача переходит в terminal status и result endpoint возвращает накопленные rows/logs

### Requirement: Общая очередь после HTTP 429
При HTTP 429 limiter SHALL записать rate-limit событие и временно перевести запросы общего endpoint в FIFO очередь с concurrency 1.

#### Scenario: Получен HTTP 429
- **WHEN** любой параллельный запрос получает 429
- **THEN** лог содержит `[GIGACHAT_RATE_LIMIT]`, retry metadata и action `global_serial_queue`
- **THEN** лог очереди сообщает, что parallel dispatch приостановлен и запросы помещены в общую FIFO очередь

### Requirement: Возврат к параллельной обработке
После первого успешного контрольного запроса в serial mode limiter SHALL освободить общую очередь и разрешить следующим запросам снова конкурировать параллельно до нового 429.

#### Scenario: Успешный probe после 429
- **WHEN** последовательный контрольный запрос завершается без 429 и retry boundary пройдена
- **THEN** limiter пишет переход `from=serial to=parallel`, освобождает ожидающие requests и следующий dispatch использует parallel mode

#### Scenario: Повторный 429
- **WHEN** после восстановления параллельности новый запрос снова получает 429
- **THEN** limiter повторно включает общую serial FIFO очередь

### Requirement: Наблюдаемая параллельность
Limiter SHALL логировать dispatch/completion с mode, active, queued, thread и timing/throughput данными, достаточными для отличия параллельного выполнения от последовательного.

#### Scenario: Одновременные запросы
- **WHEN** несколько workers получили leases одновременно
- **THEN** dispatch logs показывают `mode=parallel` и `active` больше 1 до завершения этих запросов

### Requirement: Минимальная искусственная задержка
Параллельный режим SHALL не добавлять фиксированную межзапросную задержку; ожидание допускается только из-за очереди, retry-after/backoff, serial probe или внешней длительности API.

#### Scenario: Нет rate limit
- **WHEN** API отвечает без 429 и workers доступны
- **THEN** следующий независимый request не ждёт фиксированный client-side interval
