# User Access Specification

## Purpose

Определяет регистрацию, вход, профиль, смену пароля, выход и изоляцию данных по workspace для текущего web-приложения.

## Requirements

### Requirement: Регистрация пользователя
API SHALL создавать пользователя по email и паролю, выдавать bearer token и возвращать публичный профиль без password hash.

#### Scenario: Успешная регистрация
- **WHEN** клиент отправляет допустимые `email`, `password` и `display_name` на `/api/auth/register`
- **THEN** API возвращает access token, срок действия и профиль пользователя

#### Scenario: Невалидная регистрация
- **WHEN** catalog отклоняет данные или email уже нельзя использовать
- **THEN** API возвращает HTTP 400 с причиной

### Requirement: Вход и проверка сессии
API SHALL проверять email/password, выдавать token успешному пользователю и отклонять неверные credentials.

#### Scenario: Неверный пароль
- **WHEN** credentials не проходят проверку
- **THEN** `/api/auth/login` возвращает HTTP 401

#### Scenario: Получение текущего пользователя
- **WHEN** `/api/auth/me` получает действующий Bearer token
- **THEN** API возвращает пользователя, его `workspace_id` и `workspace_role`

### Requirement: Управление профилем
Авторизованный пользователь SHALL иметь возможность обновить имя и display name и сменить пароль после проверки текущего пароля.

#### Scenario: Обновление профиля
- **WHEN** авторизованный пользователь отправляет PATCH `/api/auth/me`
- **THEN** API сохраняет разрешённые поля и возвращает обновлённый публичный профиль

#### Scenario: Смена пароля без сессии
- **WHEN** запрос смены пароля не содержит действующего token
- **THEN** API возвращает HTTP 401 и не меняет пароль

### Requirement: Выход
API SHALL отзывать переданный authorization token при вызове `/api/auth/logout`.

#### Scenario: Выход из интерфейса
- **WHEN** пользователь нажимает выход
- **THEN** frontend вызывает logout, удаляет локальный token и возвращает пользователя в публичный auth flow

### Requirement: Изоляция workspace
Операции с workbook/settings/lake, которые принимают пользователя, SHALL привязывать данные к его workspace и не смешивать их с другим workspace.

#### Scenario: Поиск записей
- **WHEN** два пользователя из разных workspace выполняют одинаковый поиск в lake
- **THEN** каждый получает только записи своего `workspace_id`
