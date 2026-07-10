# Как пользоваться DeepSeek + OpenSpec обычному пользователю

Этот файл для ситуации, когда вы не хотите разбираться в разработке, Gradle,
тестах и структуре проекта.

## Что вставить в CLI

Откройте DeepSeek CLI в корне Java/Gradle проекта и вставьте:

```text
Прочитай файл `/path/to/prompts/deepseek-openspec/START_HERE.md`, работай строго по OpenSpec для текущей Java/Gradle-репы и веди меня через меню. Я не разработчик: задавай вопросы по одному, проси отвечать цифрами и не перечитывай весь проект без необходимости.
```

Замените `/path/to/prompts/deepseek-openspec/START_HERE.md` на настоящий путь к
файлу.

## Первый запуск

1. Вы открываете терминал в корне Java/Gradle проекта.
2. Запускаете DeepSeek CLI.
3. Вставляете строку выше.
4. DeepSeek проверяет, есть ли папка `openspec/`.
5. Если OpenSpec еще не настроен, он покажет меню:

```text
1. Initialize OpenSpec for guided feature development.
2. Only inspect and explain what is missing.
3. Stop.
```

6. Вы отвечаете `1`.
7. Он спросит подтверждение `YES` или `NO`.
8. Вы отвечаете `YES`.
9. Он создает OpenSpec-структуру:

```text
openspec/
  project.md
  specs/
  changes/
  error-kb/
```

На первом запуске production Java код не должен меняться.

## Последующая работа с фичами

Каждый новый заход в CLI начинайте той же строкой.

Если OpenSpec уже настроен, DeepSeek покажет меню:

```text
1. Create a new OpenSpec change.
2. Create an OpenSpec change and implement it after approval.
3. Continue an existing OpenSpec change/task.
4. Fix a failing test/build/error using OpenSpec error memory first.
5. Refresh OpenSpec project context.
6. Archive a completed OpenSpec change.
```

Обычно выбирайте:

- `1`, если хотите сначала только описать фичу.
- `2`, если хотите описать фичу и потом, после вашего `YES`, реализовать.
- `3`, если фича уже начата.
- `4`, если что-то упало с ошибкой.
- `5`, если проект сильно изменился.
- `6`, если изменение уже завершено и его нужно перенести в постоянные specs.

## Самый частый сценарий новой фичи

1. Вставьте стартовую строку.
2. Выберите `2`.
3. Когда DeepSeek спросит описание, напишите одну фразу:

```text
Нужно добавить экспорт отчета в Excel с фильтрами по дате и статусу.
```

4. DeepSeek создаст OpenSpec change:

```text
openspec/changes/add-report-export/
  proposal.md
  tasks.md
  specs/report-export/spec.md
```

5. DeepSeek проверит change через OpenSpec validation.
6. DeepSeek спросит, можно ли начинать реализацию.
7. Вы отвечаете `YES`.
8. Он реализует задачи по одной.
9. После реализации запускает Gradle unit tests.
10. Если тесты упали, сначала проверяет `openspec/error-kb/`.
11. Если нашел новое решение, записывает его в `openspec/error-kb/`.

## Что делать после перезапуска CLI

Снова вставьте стартовую строку. DeepSeek должен сам прочитать:

- `openspec/project.md`
- активные `openspec/changes/*`
- нужные `openspec/specs/*`
- `openspec/error-kb/index.yml`

Он не должен перечитывать весь проект без необходимости.

## Что писать, если DeepSeek начал усложнять

```text
Стоп. Вернись к меню из START_HERE.md. Работай строго по OpenSpec. Я хочу отвечать цифрами и короткими фразами. Не читай весь проект заново.
```

## Что писать, если тесты упали

```text
Сначала проверь `openspec/error-kb`, есть ли известное решение этой ошибки. Если нет, исправь ошибку, запусти Gradle unit tests и запиши найденное решение в `openspec/error-kb`.
```
