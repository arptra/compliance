# complaints-trends

Пользовательское описание дашборда, логика обработки жалоб и пошаговый порядок работы: [`DASHBOARD_USER_GUIDE.md`](DASHBOARD_USER_GUIDE.md).

PDF-версия пользовательского руководства: [`output/pdf/gigachat-lab-user-guide.pdf`](output/pdf/gigachat-lab-user-guide.pdf).

Команды запуска для разработки: [`LOCAL_DEVELOPMENT.md`](LOCAL_DEVELOPMENT.md).

Максимально подробное руководство по проекту анализа жалоб из Excel:
- подготовка данных с **GigaChat только для нормализации/weak labels**,
- обучение и аналитика только на локальных моделях `scikit-learn`,
- инференс нового месяца, тренды MoM и поиск новых тем (novelty).

---

## OpenSpec SSOT и GIGACODE

Актуальное поведение frontend/backend зафиксировано в официальной структуре OpenSpec 1.x:

- точка входа и карта минимального контекста: [`openspec/README.md`](openspec/README.md);
- текущий SSOT: `openspec/specs/**/spec.md`;
- активные изменения: `openspec/changes/<change-name>/`;
- восстановленная история фич: `openspec/changes/archive/YYYY-MM-DD-<change-name>/`;
- постоянные правила для GIGACODE: [`GIGACODE.md`](GIGACODE.md);
- адаптированные OpenSpec-команды и skills для GIGACODE: `.gigacode/commands/` и `.gigacode/skills/`.

После установки Node.js 20.19+ и OpenSpec CLI:

```bash
npm install -g @fission-ai/openspec@latest
openspec list
openspec validate --specs
```

Перезапустите GIGACODE после первого получения `.gigacode/`. Дальше можно работать обычным текстом или командами:

```text
/opsx-explore "обсудить изменение"
/opsx-propose "добавить одну конкретную фичу"
/opsx-apply <change-name>
/opsx-sync <change-name>
/opsx-archive <change-name>
```

GIGACODE при новом сеансе начинает с короткого `GIGACODE.md` и `openspec/README.md`, затем читает одну нужную capability-спеку и её код. Весь репозиторий и архив в контекст не загружаются. Старый `prompts/deepseek-openspec/` остаётся самостоятельным prompt pack, но не является SSOT этого проекта.

---

## 1. Что решает проект

Проект предназначен для обработки ежемесячных Excel-файлов контакт-центра/поддержки (десятки колонок, длинные диалоги, шумные роли вроде `CLIENT/OPERATOR/CHATBOT`).

Ключевая задача: нормализовать обращения на основе полного контекста диалога и затем обучить локальные модели на подготовленном текстовом поле.

- На этапе `prepare` в LLM уходит **контекст всего диалога** (`full_dialog_text` + `dialog_context` + `signal_fields`) — поле `client_first_message` не используется для принятия решения LLM.
- На этапе `train` итоговая локальная модель обучается на поле `training.text_field` из prepared parquet (по умолчанию `client_first_message`, но это настраивается в `configs/project.yaml`).

Такой подход позволяет разделить: (1) богатую контекстную нормализацию через LLM и (2) стабильное локальное ML-обучение на фиксированном текстовом представлении.

---

## 2. Архитектура и принципы

### 2.1 Принцип разделения этапов

- **Этап подготовки (`prepare`)**: можно использовать GigaChat для нормализации и weak labels.
- **Этап обучения/инференса/сравнения (`train`, `infer-month`, `trends`, `compare`)**: только локальные ML-модели (`scikit-learn`), без LLM.

### 2.2 Почему так

- GigaChat хорошо подходит для первичной псевдо-разметки и приведения текстов к единообразной структуре.
- Локальные модели дают предсказуемую скорость, повторяемость и отсутствие внешних зависимостей в боевой аналитике.

### 2.3 Подключение к GigaChat: `mtls` и `tls`

Поддерживаются два режима:

- `llm.mode: "mtls"` — клиентский сертификат + ключ + CA bundle (взаимная TLS-аутентификация).
- `llm.mode: "tls"` — обычный TLS без клиентского сертификата (только проверка серверного сертификата).

> Важно: для `https://gigachat.devices.sberbank.ru/api/v1` обычно нужен именно `mtls`.

В `mtls` режиме клиент строит единый `SSLContext` (CA + client cert/key + optional password) и использует его и для OAuth auth-запросов, и для API-запросов. Это важно, чтобы поведение было одинаковым на обоих каналах и не возникало ошибок handshake на этапе получения токена.

---

## 3. Структура репозитория

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── project.yaml
│   ├── categories_seed.yaml
│   ├── deny_tokens.txt
│   └── extra_stopwords.txt
├── src/complaints_trends/
│   ├── cli.py
│   ├── config.py
│   ├── io_excel.py
│   ├── extract_client_first.py
│   ├── pii_redaction.py
│   ├── text_cleaning.py
│   ├── gigachat_mtls.py
│   ├── gigachat_schema.py
│   ├── prepare_dataset.py
│   ├── features.py
│   ├── train_models.py
│   ├── infer_month.py
│   ├── novelty.py
│   ├── trends.py
│   ├── compare.py
│   └── reports/
│       ├── render.py
│       └── templates/*.j2
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── reports/
├── exports/
└── tests/
```

---

## 4. Установка и быстрый старт

## 4.1 Требования
- Python 3.11+
- Linux/macOS (Windows тоже возможен, но команды в README даны для bash)

### 4.2 Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

### 4.3 Настройка окружения

Скопируйте `.env.example` в `.env` и проверьте пути:

```env
GIGACHAT_CA_BUNDLE_FILE=certs/ca.pem
GIGACHAT_CERT_FILE=certs/client.pem
GIGACHAT_KEY_FILE=certs/client.key
GIGACHAT_KEY_PASSWORD=
GIGACHAT_BASE_URL=https://gigachat.devices.sberbank.ru/api/v1
```


### 4.3.2 Режим вопросов (`llm.category_mode: questions`)

Поддерживается третий режим категоризации: `questions` (кроме `taxonomy` и `discover`).

Пример конфига:

```yaml
llm:
  category_mode: "questions"
  questions_file: "configs/questions_categories.json"
```

Формат `questions_file` (строгий JSON):

```json
{
  "version": 1,
  "categories": [
    {"question_ru": "Есть ли жалоба на платежи?"},
    {"question_ru": "Есть ли жалоба на вход в приложение?"}
  ]
}
```

При запуске prepare в этом режиме:
- вопросы из JSON валидируются и получают стабильные question-коды `q_<index>_<sha1_8>`,
- GigaChat по каждому вопросу определяет, какую бизнес-категорию использовать,
- первое присвоение `question_code -> category_code/category_name` сохраняется и дальше переиспользуется (модель больше не переименовывает эту категорию),
- `category_name` должен быть коротким названием категории (не равным тексту вопроса); если модель вернула текст вопроса, система автоматически сокращает имя до короткой формы,
- mapping вопросов сохраняется в `data/interim/questions_taxonomy.json`,
- mapping `question_code -> category_code/category_name` сохраняется в `data/interim/questions_category_map.json`,
- после `prepare` сохраняется итоговый JSON-свод по вопросам/категориям `data/interim/questions_prepare_categories.json` (включая счётчики категорий),
- в parquet пишутся стандартные поля (`is_complaint_llm`, `complaint_category_llm`, `complaint_subcategory_llm`, `keywords_llm`, `notes_llm`),
- fallback категория `OTHER` означает "не попало в вопросы".

### 4.3.1 Приоритет config vs .env для LLM

Сейчас реализован следующий порядок при загрузке:
1. читается `configs/project.yaml`;
2. затем для `llm` применяются overrides из `.env`/переменных окружения (если они заданы).

То есть, если в `.env` есть переменные ниже, они **переопределят** значения из YAML:
- `GIGACHAT_BASE_URL` -> `llm.base_url`
- `GIGACHAT_CA_BUNDLE_FILE` -> `llm.ca_bundle_file`
- `GIGACHAT_CERT_FILE` -> `llm.cert_file`
- `GIGACHAT_KEY_FILE` -> `llm.key_file`
- `GIGACHAT_MODEL` -> `llm.model`
- `GIGACHAT_VERIFY_SSL_CERTS` -> `llm.verify_ssl_certs`
- `GIGACHAT_KEY_PASSWORD_ENV` -> `llm.key_file_password_env` (опционально)

Важно: пароль ключа хранится в переменной, имя которой задается через `llm.key_file_password_env` (по умолчанию `GIGACHAT_KEY_PASSWORD`).

### 4.4 Сертификаты
1. Положите сертификаты клиента и ключ в `certs/`.
2. Укажите валидный CA bundle (например Russian Trusted Root CA).
3. Рекомендуется оставлять `verify_ssl_certs: true`.



## 4.5 Минимальный пример `configs/project.yaml` (MVP)

Если хотите стартовать быстро, можно начать с такого минимального конфига:

```yaml
input:
  input_dir: "data/raw"
  file_glob: "*.xlsx"
  file_names: null  # можно явно перечислить файлы
  datetime_column: "created_at"
  datetime_format: "%Y-%m-%d %H:%M:%S"
  id_column: null
  signal_columns: ["dialog_text", "call_text", "comment_text", "summary_text", "subject", "channel", "product", "status"]
  dialog_column: "dialog_text"  # legacy fallback
  dialog_columns: ["dialog_text", "call_text", "comment_text", "summary_text"]
  encoding: "utf-8"

client_first_extraction:
  enabled: true
  client_markers: ["CLIENT", "КЛИЕНТ", "USER"]
  operator_markers: ["OPERATOR", "ОПЕРАТОР", "SUPPORT"]
  chatbot_markers: ["CHATBOT", "БОТ"]
  stop_on_markers: ["OPERATOR", "ОПЕРАТОР", "CHATBOT", "БОТ"]
  fallback_mode: "first_paragraph"
  fallback_first_n_chars: 600
  min_client_len: 20
  take_second_client_if_too_short: true

pii:
  enabled: true
  replace_email: "<EMAIL>"
  replace_phone: "<PHONE>"
  replace_url: "<URL>"
  replace_card: "<CARD>"
  replace_account: "<ACCOUNT>"

llm:
  enabled: true
  mode: "mtls"
  base_url: "https://gigachat.devices.sberbank.ru/api/v1"
  ca_bundle_file: "certs/ca.pem"
  cert_file: "certs/client.pem"
  key_file: "certs/client.key"
  key_file_password_env: "GIGACHAT_KEY_PASSWORD"
  verify_ssl_certs: true
  model: "GigaChat"
  max_workers: 4
  batch_size: 10
  max_text_chars: 1200
  cache_db: "data/interim/gigachat_cache.sqlite"
  prompt_version: "v1"
  token_batch_size: 12000
  batch_mode: false
  request_metrics_enabled: true
  async_mode: false
  parallel_mode: false

prepare:
  pilot_limit: 1000
  date_from: null
  date_to: null
  output_parquet: "data/processed/all_prepared.parquet"
  pilot_parquet: "data/processed/pilot_prepared.parquet"
  pilot_review_xlsx: "exports/pilot_review.xlsx"

training:
  text_field: "client_first_message"
  complaint_threshold: 0.5
  vectorizer:
    word_ngram: [1, 2]
    char_ngram: [3, 5]
    max_features_word: 100000
    max_features_char: 50000
    min_df: 3
    max_df: 0.8
  classifier:
    complaint: "logreg"
    category: "linearsvc"
  validation:
    split_mode: "time"
  model_dir: "models"

analysis:
  novelty:
    enabled: true
    method: "kmeans_distance"
    svd_components: 100
    kmeans_k: 20
    threshold_percentile: 98
    min_cluster_size: 10
  reports_dir: "reports"

files:
  deny_tokens_path: "configs/deny_tokens.txt"
  extra_stopwords_path: "configs/extra_stopwords.txt"
  categories_seed_path: "configs/categories_seed.yaml"
```

---

## 4.6 Первый запуск (пошагово)

1. Положите 1–2 Excel файла в `data/raw/` (например `2025-10.xlsx`, `2025-11.xlsx`).
2. Проверьте сертификаты в `certs/` и `.env`.
3. Запустите pilot-подготовку:

```bash
python -m complaints_trends.cli prepare --config configs/project.yaml --pilot --date-from "2025-10-01 00:00:00" --date-to "2025-10-31 23:59:59" --limit 1000
```

4. Проверьте руками:
   - `exports/pilot_review.xlsx`
   - `reports/pilot_report.html`
5. Запустите полную подготовку:

```bash
python -m complaints_trends.cli prepare --config configs/project.yaml
```

6. Обучите модели:

```bash
python -m complaints_trends.cli train --config configs/project.yaml
```

7. Постройте тренды:

```bash
python -m complaints_trends.cli trends --config configs/project.yaml
```

8. Прогон нового месяца:

```bash
python -m complaints_trends.cli infer-month --config configs/project.yaml --excel data/raw/2025-12.xlsx --month 2025-12
```

9. Сравните с baseline:

```bash
python -m complaints_trends.cli compare --config configs/project.yaml --new-month 2025-12 --baseline-range 2025-10..2025-11
```

Для полностью локального smoke без реального GigaChat используйте:

```bash
python -m complaints_trends.cli demo
```

---

## 5. Как конфигурировать проект (ПОДРОБНО)


### Важно: обучение теперь строится по фильтру периода `event_time`
- Основной сценарий: `input.datetime_column` (например `created_at`) с форматом `2025-01-09 12:55:29`.
- Для отбора данных используйте диапазон:
  - `prepare.date_from`
  - `prepare.date_to`
- Также можно передать через CLI: `prepare --date-from ... --date-to ...`.


Главный файл: `configs/project.yaml`.

Ниже — ключевые блоки и их влияние.

## 5.1 `input`
- `input_dir`, `file_glob`: где искать Excel.
- `id_column`: если нет стабильного ID, `row_id` будет сгенерирован.
- `signal_columns`: дополнительные поля для LLM/аналитики (например `subject/channel/product/status`). **Диалоговые поля (`dialog_columns`) сюда включать не обязательно** — они и так обрабатываются отдельно.
- `dialog_column`: legacy-колонка с полным диалогом (fallback).
- `dialog_columns`: список нескольких текстовых полей (например чат/звонок/комментарий/суммаризация). Пайплайн автоматически выберет наиболее содержательное непустое поле как `raw_dialog`, а также передаст все непустые поля в `dialog_context` для GigaChat.
- `signal_columns` и `dialog_columns` логически разделены: из `signal_columns` в prompt уходят только недиалоговые поля (`signal_fields`).

**Практика:**
- Если есть надежный бизнес-идентификатор, обязательно задайте `id_column` (упростит merge с gold-разметкой).

## 5.2 `client_first_extraction`
Управляет критически важной логикой выделения первого клиентского запроса.
- `client_markers`, `operator_markers`, `chatbot_markers`: словари ролей.
- `stop_on_markers`: какие роли прерывают извлечение клиентского фрагмента.
- `fallback_mode`: что делать, если маркеры не найдены.
- `min_client_len`, `take_second_client_if_too_short`: защита от слишком коротких реплик.

**Как улучшать качество:**
- Добавляйте реальные маркеры ваших чатов (в т.ч. с префиксами каналов/CRM).
- Проверьте короткие шаблонные реплики (например, “Здравствуйте”) — часто полезно включать `take_second_client_if_too_short`.

## 5.3 `pii`
Параметры редактирования персональных данных перед LLM:
- email/phone/url/card/account заменяются на токены.

**Важно:** `raw_dialog` хранится как исходник, но в LLM уходит редактированная версия.

## 5.4 `llm`
- `enabled`: включение/выключение GigaChat.
- `mode`: режим TLS подключения (`mtls` или `tls`).

### Режим `mtls`
Используйте, когда endpoint требует клиентский сертификат.

Обязательные поля:
- `ca_bundle_file`, `cert_file`, `key_file`.

Опционально:
- `key_file_password_env` — имя env-переменной с паролем приватного ключа.

Пример:

```yaml
llm:
  enabled: true
  mode: "mtls"
  base_url: "https://gigachat.devices.sberbank.ru/api/v1"
  ca_bundle_file: "certs/ca.pem"
  cert_file: "certs/client.pem"
  key_file: "certs/client.key"
  key_file_password_env: "GIGACHAT_KEY_PASSWORD"
  verify_ssl_certs: true
```

### Режим `tls`
Используйте, когда endpoint не требует client certificate.

- `cert_file`/`key_file` не нужны.
- `ca_bundle_file` можно оставить (для кастомного CA) или не задавать.

Пример:

```yaml
llm:
  enabled: true
  mode: "tls"
  base_url: "https://your-tls-endpoint.example/api/v1"
  ca_bundle_file: "certs/ca.pem"
  cert_file: null
  key_file: null
  verify_ssl_certs: true
```

Общие параметры для обоих режимов:
- `max_workers`, `batch_size`: скорость/нагрузка.
- `max_text_chars`: ограничение длины входа в LLM.
- `cache_db`: sqlite-кэш ответов.
- `prompt_version`: меняйте при изменении промпта, чтобы не смешивать старые кэши.
- `token_batch_size`: лимит суммарных токенов в одном batch-запросе к LLM.
- `batch_mode`: если `true`, `prepare` группирует строки в батчи так, чтобы сумма токенов по строкам в одном POST была меньше `token_batch_size`.
- `request_metrics_enabled`: включает подсчет токенов через `/tokens/count` и INFO-логи об успешной доставке/латентности LLM-запросов.
- `async_mode`: включает асинхронный режим отправки запросов к LLM (конкурентные задачи через `asyncio`, ограничение по `max_workers`).
- `parallel_mode`: включает параллельный режим через пул потоков (`ThreadPoolExecutor`, ограничение по `max_workers`).

## 5.5 `prepare`
- `pilot_limit`: пилотный режим (ограничение строк).
- `date_from`/`date_to`: период отбора данных по `event_time`.
- `output_parquet`, `pilot_parquet`, `pilot_review_xlsx`: куда сохранять артефакты.

## 5.6 `training`
- `text_field`: поле из prepared parquet, которое реально идет в обучение локальных моделей. По умолчанию `client_first_message` (см. `configs/project.yaml`), но можно переключить, например, на `raw_dialog` или другое текстовое поле из датасета.
- `complaint_threshold`: порог бинарной классификации.
- `vectorizer.*`: диапазоны n-gram и ограничения словаря.
- `classifier.complaint/category`: выбор модели.
- `validation.split_mode`: `time` или `random`.

## 5.7 `analysis.novelty`
- `method`: `kmeans_distance` или `lof`.
- `svd_components`: размерность пространства novelty.
- `kmeans_k`: число центроидов нормы.
- `threshold_percentile`: порог новизны.
- `min_cluster_size`: минимальный размер кластера новых тем.

## 5.8 `files`
- `deny_tokens_path`: мусорные токены (client/operator/chatbot и т.п.).
- `extra_stopwords_path`: доменные стоп-слова.
- `categories_seed_path`: начальное стабильное дерево категорий. Поддерживается расширенный формат: `label_ru`, `subcategories` как словарь объектов и отдельный блок `loan_products` (используется в LLM-правилах как независимый признак).

---

## 6. Что происходит на каждой стадии

## 6.1 `prepare`
Команды:

```bash
python -m complaints_trends.cli prepare --config configs/project.yaml --pilot --date-from "2025-09-01 00:00:00" --date-to "2025-09-30 23:59:59" --limit 5000
python -m complaints_trends.cli prepare --config configs/project.yaml
```

Что делает:
1. Читает все Excel и определяет месяц.
2. Собирает несколько текстовых полей (`dialog_columns`), выбирает основной источник `dialog_source_field`, сохраняет его в `raw_dialog` и весь контекст в `dialog_context_map`.
3. Извлекает `client_first_message` из выбранного основного поля.
4. Делает PII-редакцию.
5. Формирует компактный payload для LLM (без простыни полного диалога).
6. Нормализует в строгий JSON-контракт.
7. Кэширует LLM-ответы в SQLite.
8. Пишет parquet + pilot review Excel/отчеты.

Выход:
- `data/processed/all_prepared.parquet`
- `data/processed/pilot_prepared.parquet`
- `exports/pilot_review.xlsx`
- `reports/pilot_report.html`, `reports/pilot_report.md`


### 6.1.1 Batch-режим LLM по токенам (`llm.batch_mode`)

Если включить:

```yaml
llm:
  batch_mode: true
  token_batch_size: 12000
  request_metrics_enabled: true
```

то `prepare` работает так:
1. Для каждой записи считает токены через `POST /tokens/count` (тот же механизм, что уже используется для токен-оценки запросов).
2. Собирает батчи жадно в исходном порядке строк: добавляет запись в текущий batch, пока сумма токенов не превысит `token_batch_size`.
3. Отправляет один POST `/chat/completions` на весь batch (`task=normalize_tickets`, `inputs=[...]`).
4. Если LLM вернул не все записи (например отправили 24, получили 18), пайплайн вычисляет пропущенные индексы и переотправляет только их отдельным retry-batch.
5. Если после retry часть записей все равно не обработалась или batch-запрос упал — делает fallback на поштучную обработку оставшихся записей.

Важно:
- Ограничение применяется к **сумме токенов записей** в batch.
- Если `/tokens/count` недоступен, используется безопасная оценка по длине prompt (эвристика), чтобы batching не ломал pipeline.


### 6.1.2 Режимы ускорения LLM: async и parallel

В `llm` добавлены два переключателя:

```yaml
llm:
  max_workers: 8
  async_mode: false
  parallel_mode: false
```

Как это работает:
- `async_mode: true` — асинхронный режим (паттерн *Producer/Consumer* + ограничение конкуренции через `Semaphore`), запросы выполняются конкурентно через `asyncio`.
- `parallel_mode: true` — параллельный режим (паттерн *Thread Pool*), запросы выполняются в нескольких потоках.
- если оба режима выключены — классический синхронный режим.
- если включены оба, приоритет у `async_mode` (он уже конкурентный), `parallel_mode` игнорируется с логом.

Рекомендации по включению:
- для I/O-bound API (GigaChat) сначала пробуйте `async_mode: true`;
- если в окружении нельзя/неудобно использовать async event-loop в этом шаге — используйте `parallel_mode: true`;
- тюнинг скорости делается через `max_workers` (слишком большое значение может упереться в лимиты API).

Примеры:

**Асинхронный режим**
```yaml
llm:
  async_mode: true
  parallel_mode: false
  max_workers: 8
```

**Параллельный режим**
```yaml
llm:
  async_mode: false
  parallel_mode: true
  max_workers: 8
```

**Полностью синхронный режим**
```yaml
llm:
  async_mode: false
  parallel_mode: false
```

## 6.2 `train`
Команда:

```bash
python -m complaints_trends.cli train --config configs/project.yaml
```

### На каких полях обучается итоговая модель (точно по коду)

Ниже поля, которые участвуют именно в обучении локальной модели (`train_models.py`):

| Поле | Откуда берется | Где используется | Зачем |
|---|---|---|---|
| `training.text_field` (обычно `client_first_message`) | `data/processed/all_prepared.parquet` | `df[text_field] -> text_clean -> TF-IDF` | Основной текстовый вход в модель |
| `is_complaint_gold` | `exports/pilot_review.xlsx` (если есть) | `y_bin` | Приоритетная ручная метка бинарного класса |
| `is_complaint_llm` | результат `prepare` (LLM) | fallback для `y_bin` | Weak label, когда нет gold |
| `category_gold` | `exports/pilot_review.xlsx` (если есть) | `y_cat` | Приоритетная ручная метка категории |
| `complaint_category_llm` | результат `prepare` (LLM) | fallback для `y_cat` | Weak label категории |
| `event_time` | исходные Excel | split при `validation.split_mode: time` | Временная валидация |

Важно:
- Модель **не обучается напрямую** на `full_dialog_text`, `dialog_context`, `signal_fields`.
- Эти поля используются в `prepare` для LLM-нормализации и генерации weak labels, которые затем попадают в `*_llm` колонки.

### Какая именно модель обучается

Фактически это двухступенчатый локальный пайплайн `scikit-learn`:
1. **Векторизация**: `TF-IDF(word) + TF-IDF(char)` (hstack).
2. **Бинарная модель жалобы**:
   - `LogisticRegression(class_weight='balanced')`, если `training.classifier.complaint=logreg`,
   - или `LinearSVC` + `CalibratedClassifierCV`, если `linearsvc`.
3. **Модель категории** (только для строк, где `is_complaint=True`):
   - `LinearSVC(class_weight='balanced')` (по умолчанию),
   - либо `LogisticRegression(multinomial, class_weight='balanced')`.

Итоговые артефакты: `vectorizers.joblib`, `complaint_model.joblib`, `category_model.joblib`, `label_encoder.joblib`.

Что делает:
1. Загружает prepared parquet.
2. Подмешивает gold-разметку из `pilot_review.xlsx` (если есть).
3. Формирует целевые переменные:
   - `y_bin`: `is_complaint_gold` (если размечено) иначе `is_complaint_llm`.
   - `y_cat`: `category_gold` (если размечено) иначе `complaint_category_llm`.
4. Берет текстовое поле `training.text_field` (по умолчанию `client_first_message`) и чистит его в `text_clean`.
5. Строит объединенные TF-IDF признаки:
   - word n-gram (`TfidfVectorizer`, `training.vectorizer.word_ngram`),
   - char n-gram (`analyzer=char_wb`, `training.vectorizer.char_ngram`).
6. Обучает 2 локальные модели scikit-learn:
   - бинарный классификатор жалобы (`training.classifier.complaint`: `logreg` или `linearsvc` + калибровка),
   - классификатор категорий только на жалобах (`training.classifier.category`: `linearsvc` или `logreg`).
7. Валидирует (time split/random split) и считает метрики (`complaint_f1`, `category_macro_f1`).
8. Сохраняет артефакты и training report.

Выход:
- `models/vectorizers.joblib`
- `models/complaint_model.joblib`
- `models/category_model.joblib`
- `models/label_encoder.joblib`
- `models/training_metadata.json`
- `reports/training_report.html`, `reports/training_report.md` (человеко-читаемый текст)
- `reports/training_predicted_complaint_distribution.png`
- `reports/training_category_hist_ru.png`
- `reports/training_subcategory_hist_ru.png`
- `reports/training_category_subcategory_scatter_ru.png`

## 6.3 `trends`
Команда:

```bash
python -m complaints_trends.cli trends --config configs/project.yaml
```

Что делает:
- агрегирует долю жалоб по месяцам,
- строит распределение категорий по месяцам,
- подготавливает отчет по динамике.

Выход:
- `reports/trends_report.html`

## 6.4 `infer-month`
Команда:

```bash
python -m complaints_trends.cli infer-month --config configs/project.yaml --excel data/raw/2025-12.xlsx --month 2025-12
```

Что делает:
1. Читает новый сырой Excel.
2. Извлекает `client_first_message` (или другое поле, если вы сменили `training.text_field` и синхронизировали preprocessing) и подает его в тот же векторизатор.
3. Применяет локальные модели.
4. Сохраняет разметку и отчет.
5. Перед сохранением parquet нормализует mixed object-колонки к строкам, чтобы избежать ошибок Arrow/pyarrow на реальных Excel.

Выход:
- `exports/month_labeled_2025-12.xlsx`
- `data/interim/month_2025-12.parquet`
- `reports/month_report_2025-12.html`

## 6.5 `compare`
Команда:

```bash
python -m complaints_trends.cli compare --config configs/project.yaml --new-month 2025-12 --baseline-range 2025-06..2025-11
```

Что делает:
1. Сравнивает новый месяц с baseline.
2. Считает изменения долей категорий.
3. Ищет новизну через SVD + KMeans distance/LOF.
4. Кластеризует новые жалобы.
5. Экспортирует список новых тем для ручного ревью.

Выход:
- `exports/new_topics_2025-12.xlsx`
- `reports/compare_2025-12_vs_baseline.html`

---

## 7. Как валидировать каждую стадию (чеклисты)

## 7.1 Валидация `prepare`
1. Откройте `exports/pilot_review.xlsx`.
2. Проверьте 50–100 строк вручную:
   - корректно ли выделен `client_first_message`;
   - нет ли ролей/мусора в summary/keywords;
   - логична ли категория.
3. Проверьте `reports/pilot_report.html`:
   - доля жалоб,
   - топ категорий,
   - примеры жалоб/не-жалоб.

Сигналы проблем:
- в summary/keywords появляются `CLIENT/OPERATOR/CHATBOT`;
- слишком много `OTHER`;
- слишком короткие `client_first_message`.

## 7.2 Валидация `train`
1. Смотрите `reports/training_report.html`.
2. Для бинарной модели: `precision/recall/F1`.
3. Для категорий: `macro-F1` и разбор классов.
4. Проверьте вручную ошибки на границе классов.

Сигналы проблем:
- высокий recall и низкий precision (слишком много ложных жалоб);
- сильный перекос в одну категорию;
- ухудшение на последнем месяце при time split.

## 7.3 Валидация `infer-month`
1. Проверьте `month_labeled_*.xlsx`:
   - разумность доли жалоб,
   - категории и confidence на примерах.
2. Сверьте с бизнес-ожиданиями месяца.

## 7.4 Валидация `compare`
1. Смотрите `compare_*.html`:
   - где реальные MoM-сдвиги,
   - какие категории выросли.
2. Проверьте `new_topics_*.xlsx`:
   - действительно ли это новые темы,
   - не является ли новизна артефактом очистки/шумом.

---

## 8. Как вносить изменения в проект

Рекомендуемый процесс:
1. Создайте ветку под задачу.
2. Обновите конфиги/код.
3. Прогоните минимум:
   - `pytest`
   - `demo` smoke
4. Проверьте отчеты и артефакты.
5. Зафиксируйте изменения с понятным commit message.
6. Обновите README при любом изменении логики.

### 8.1 Где менять извлечение первого сообщения
- `src/complaints_trends/extract_client_first.py`
- словари ролей в `configs/project.yaml`

### 8.2 Где менять правила PII
- `src/complaints_trends/pii_redaction.py`
- замены в `configs/project.yaml -> pii`

### 8.3 Где менять LLM-нормализацию
- `src/complaints_trends/gigachat_mtls.py` (prompt/repair/cache)
- `src/complaints_trends/gigachat_schema.py` (контракт)
- `configs/categories_seed.yaml` (стабильность категорий)

### 8.4 Где менять ML-модели
- `src/complaints_trends/features.py`
- `src/complaints_trends/train_models.py`
- параметры в `configs/project.yaml -> training`

### 8.5 Где менять novelty
- `src/complaints_trends/novelty.py`
- `configs/project.yaml -> analysis.novelty`

---

## 9. Какие параметры влияют на качество (и как)

Ниже practical tuning guide.

## 9.1 Извлечение client-first
- `min_client_len` ↑: меньше коротких/пустых реплик, но риск пропуска реально коротких жалоб.
- `take_second_client_if_too_short=true`: часто повышает качество.
- Недостаток маркеров ролей => шум в тексте => хуже всё downstream.

## 9.2 Параметры LLM
- `max_text_chars` ↑: больше контекста, но дольше и дороже; иногда растет шум.
- `prompt_version`: обязательно менять при изменении prompt/schema.
- `batch_size/max_workers`: влияет на throughput, но не на качество напрямую.

## 9.3 TF-IDF
- `word_ngram` (1,2) обычно базово хорошо.
- `char_ngram` помогает при опечатках/транслите.
- `min_df` ↑: меньше шума, но теряются редкие сигналы.
- `max_features_*` ↑: потенциально лучше качество, но больше RAM/время.

## 9.4 Бинарный классификатор жалоб
- `complaint_threshold`:
  - ниже => выше recall, ниже precision;
  - выше => выше precision, ниже recall.

## 9.5 Категоризация
- `linearsvc` часто устойчив на sparse-признаках.
- `logreg multinomial` удобен вероятностями.
- Критично качество weak/gold labels и баланс классов.

## 9.6 Novelty
- `threshold_percentile`:
  - 98 = более строгая новизна,
  - 95 = больше кандидатов, но больше false positives.
- `svd_components` слишком мало => потеря нюансов, слишком много => шум/медленнее.
- `kmeans_k`:
  - мало кластеров => грубая норма,
  - слишком много => переобучение нормы.

---

## 10. Какие результаты получаются и как интерпретировать

## 10.1 Pilot report
- **Complaint share**: грубая первичная оценка жалобности потока.
- **Top categories**: структура проблем по weak labels.
- **Примеры**: главное место ручной проверки корректности.

Интерпретация:
- Если category распределение нелогично, сначала чините extraction и категориальный seed.

## 10.2 Training report
- `precision/recall/F1` по жалобам:
  - бизнесу обычно нужен баланс, но для мониторинга часто важнее recall.
- `macro-F1` по категориям:
  - показывает устойчивость на редких классах.

Интерпретация:
- Высокий micro и низкий macro = модель плохо видит редкие категории.

## 10.3 Month report
- Доля жалоб за месяц,
- Топ категорий,
- Примеры полных `client_first_message`.

Интерпретация:
- Используйте как операционный мониторинг: что болит сейчас.

## 10.4 Compare report
- baseline vs new по долям,
- категории с ростом/падением,
- кластеры новых тем.

Интерпретация:
- Новая тема = либо реально новый паттерн, либо деградация данных/экстракции.
- Всегда проверяйте примеры руками перед управленческими решениями.

---

## 11. Полный список CLI команд

```bash
python -m complaints_trends.cli prepare --config configs/project.yaml --pilot --date-from "2025-09-01 00:00:00" --date-to "2025-09-30 23:59:59" --limit 5000
python -m complaints_trends.cli prepare --config configs/project.yaml
python -m complaints_trends.cli train --config configs/project.yaml
python -m complaints_trends.cli trends --config configs/project.yaml
python -m complaints_trends.cli infer-month --config configs/project.yaml --excel data/raw/2025-12.xlsx --month 2025-12
python -m complaints_trends.cli compare --config configs/project.yaml --new-month 2025-12 --baseline-range 2025-06..2025-11
python -m complaints_trends.cli demo
```

---

## 12. Smoke и тестирование

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m complaints_trends.cli demo
```

`demo` генерирует синтетические данные, проходит полный пайплайн и проверяет, что все стадии связаны корректно.

---

## 13. Частые проблемы и решения

1. **Нет входных файлов**
   - Проверьте `input_dir`, `file_glob`, имена файлов.

2. **Неправильный период выборки**
   - Проверьте `prepare.date_from`/`prepare.date_to` и timezone/формат дат.

3. **Много мусора в summary/keywords**
   - Обновите deny tokens, prompt и логику repair.

4. **Плохая точность категории**
   - Увеличьте объем gold-меток из pilot-review,
   - пересоберите categories seed,
   - подберите `min_df/max_features`.

5. **Слишком много/мало novel topics**
   - Подстройте `threshold_percentile`, `kmeans_k`, `svd_components`.

6. **Дата в колонке Excel не парсится**
   - Проверьте `input.datetime_column` и `input.datetime_format`.
   - Ожидаемый формат: `2025-01-09 12:55:29`.

7. **`TLSV13_ALERT_CERTIFICATE_REQUIRED` при обращении к GigaChat**
   - Сервер требует client certificate.
   - Если у вас `llm.mode: "tls"`, переключитесь на `llm.mode: "mtls"` и задайте `ca_bundle_file/cert_file/key_file`.
   - Если уже `llm.mode: "mtls"`, проверьте пути `llm.ca_bundle_file`, `llm.cert_file`, `llm.key_file` и env overrides `GIGACHAT_CA_BUNDLE_FILE`, `GIGACHAT_CERT_FILE`, `GIGACHAT_KEY_FILE`.
   - Убедитесь, что файлы реально существуют и не пустые.

7. **`KeyError: is_complaint_llm` в pilot prepare**
   - Обычно это пустой диапазон `date_from/date_to` (после фильтра 0 строк).
   - Сейчас пайплайн корректно обрабатывает пустой диапазон и строит пустой pilot-отчет без падения.

9. **Несколько файлов для обработки**
   - Используйте `input.file_names` для явного списка файлов.

10. **Что указывать для периода обучения?**
   - Всегда используйте datetime-колонку Excel: `input.datetime_column` + `input.datetime_format`.
   - Выборку задавайте только через `prepare.date_from` / `prepare.date_to` или CLI `--date-from` / `--date-to`.


---

## 14. Что улучшать дальше

- Добавить детальные графики (PR curve, confusion matrix heatmap, MoM stacked plots).
- Добавить отдельную команду `label-new-topics` (ручная доразметка новых кластеров).
- Добавить контроль дрейфа по токенам и каналам.
- Добавить расширенные unit/integration тесты на реальные форматы диалогов.


## 5.9 Визуальный анализ предсказаний (`viz-build`, `viz-view`)

Ниже — **полный рабочий флоу**, какие команды запускать и как интерпретировать графики.

### Что нужно, чтобы всё заработало

Минимально:
1. есть подготовленный датасет: `data/processed/all_prepared.parquet` (после `prepare`),
2. для режима `--label-source pred` есть обученные модели (`models/*.joblib`) после `train`,
3. для режима `--label-source llm` достаточно prepared parquet с LLM-колонками (`is_complaint_llm`, `complaint_category_llm`).

### Команды (пошагово, с нуля)

```bash
export PYTHONPATH=src

# 1) Подготовка weak labels (LLM) и parquet
python -m complaints_trends.cli prepare --config configs/project.yaml

# 2) Обучение локальных моделей (нужно для label-source=pred)
python -m complaints_trends.cli train --config configs/project.yaml

# 3) Построение визуального отчёта по предсказаниям модели
python -m complaints_trends.cli viz-build   --config configs/project.yaml   --tag demo_pred   --label-source pred   --freq D   --top-n 8   --baseline-range 2025-10..2025-11   --new-month 2025-12

# 5) (Опционально) Построение отчёта по weak labels LLM
python -m complaints_trends.cli viz-build   --config configs/project.yaml   --tag demo_llm   --label-source llm   --freq D   --top-n 8

# 6) Интерактивный локальный просмотр (matplotlib GUI)
python -m complaints_trends.cli viz-view --tag demo_pred
# или явно путь
python -m complaints_trends.cli viz-view --state data/interim/viz_state_demo_pred.parquet
```

### Что создаётся

После `viz-build`:
- `data/interim/all_predicted.parquet` (только для `--label-source pred`, если отсутствует или задан `--force-materialize`),
- `data/interim/viz_state_<tag>.parquet` — агрегированная витрина,
- `data/interim/viz_meta_<tag>.json` — параметры запуска,
- `reports/viz_<tag>/stacked_area_counts.png`,
- `reports/viz_<tag>/share_lines.png`,
- `reports/viz_<tag>/pareto_categories.png`,
- `reports/viz_<tag>/heatmap_dow_hour.png`,
- `reports/viz_<tag>/delta_bars.png`,
- `reports/viz_<tag>/report.md`.

### Параметры `viz-build`

- `--label-source pred|llm`
  - `pred`: использовать локальные модели и materialize predictions,
  - `llm`: использовать weak labels из prepared parquet.
- `--freq D|W|M` — дневная/недельная/месячная агрегация.
- `--top-n` — сколько категорий оставлять явно (остальные сворачиваются в `OTHER`).
- `--date-from`, `--date-to` — фильтр периода.
- `--baseline-range`, `--new-month` — для delta-графика изменений.
- `--force-materialize` — пересоздать `all_predicted.parquet`.

### Как анализировать графики

1. **stacked_area_counts**
   - показывает абсолютный объём жалоб по категориям во времени,
   - ищите резкие всплески по отдельным категориям.
2. **share_lines**
   - показывает долю категорий среди жалоб,
   - помогает отличать рост общего трафика от реального сдвига структуры.
3. **pareto_categories**
   - ранжирование категорий по объёму + кумулятивная линия,
   - удобно выбирать приоритетные категории для улучшений.
4. **heatmap_dow_hour**
   - паттерн "день недели × час" для жалоб,
   - помогает планировать операционные ресурсы/нагрузку.
5. **delta_bars**
   - вклад категорий в изменение между baseline и new month (в pp),
   - быстрый ответ: какие категории дали основной рост/падение.

### Быстрый troubleshooting

- Ошибка про отсутствие `models/*.joblib` при `--label-source pred`:
  сначала выполните `train`.
- Пустые графики:
  проверьте фильтр дат (`--date-from/--date-to`) и наличие `event_time` в исходных данных.
- `viz-view` не открывает окно:
  запускайте локально с доступным GUI backend matplotlib (не headless CI).


### Как учитывается `infer-month` в визуальном отчёте

Если при `viz-build --label-source pred` передан `--new-month YYYY-MM`, то отчёт пытается автоматически подхватить `data/interim/month_YYYY-MM.parquet` (результат `infer-month`) и добавить его в витрину для расчёта delta и отдельной секции интерпретации.

В `reports/viz_<tag>/report.md` появится блок **"Интерпретация infer-month"**:
- был ли реально подключён parquet из `infer-month`,
- сколько строк оттуда использовано,
- топ категорий нового месяца с `count` и `share_of_complaints`.

Интерпретация:
- `count` — абсолютное число жалоб категории в новом месяце;
- `share_of_complaints` — доля категории среди всех жалоб нового месяца;
- `delta_bars` — насколько доля категории изменилась относительно baseline периода (в процентных пунктах).

## Novelty-hunt: поиск новых подтипов при тех же категориях

`novelty-hunt` — отдельный режим для поиска "непохожего на прошлое" внутри уже известных категорий.
Он не заменяет `compare/novelty`, а работает параллельно и ищет дрейф формулировок/контекста при том же label.

Пример запуска:

```bash
python -m complaints_trends.cli novelty-hunt \
  --config configs/project.yaml \
  --new-month 2025-12 \
  --baseline-range 2025-10..2025-11 \
  --tag 2025_12
```

Опционально можно включить LLM-описания кластеров (с кэшированием):

```bash
python -m complaints_trends.cli novelty-hunt \
  --config configs/project.yaml \
  --new-month 2025-12 \
  --baseline-range 2025-10..2025-11 \
  --tag 2025_12 \
  --use-llm-summary
```

Артефакты:
- `data/interim/novelty_hunt_state_<tag>.parquet`
- `data/interim/novelty_hunt_meta_<tag>.json`
- `data/interim/novelty_hunt_clusters_<tag>.json`
- `exports/novelty_hunt_<tag>.xlsx`
- `reports/novelty_hunt_<tag>.html`

## Pattern monitoring: ongoing special pattern внутри старых категорий

Новый режим `pattern-fit` + `pattern-monitor` ищет и отслеживает продолжающийся специальный подтип жалоб **внутри существующих категорий**.

### Кратко: принципы (тезисно)

- **Не заменяет** существующие режимы (`compare`, `novelty-hunt`), а работает как отдельный контур.
- Базовая логика двухслойная:
  1) сначала смотрим, **какие категории выросли** относительно baseline,
  2) затем внутри этих категорий ищем **event-like подпаттерн**.
- Сигнал строится не по "ключевым словам причины", а по **схожести формулировок/контекстов** внутри категории.
- Результат мониторинга учитывает:
  - row-level похожесть на event-профиль,
  - отклонение от normal-профиля,
  - day-level давление по категории и сглаженное состояние (инерция).
- Пайплайн воспроизводимый: все ключевые шаги сохраняются в parquet/json/joblib/xlsx/html для ручной проверки.

### Конфиг `analysis.pattern_monitoring`: что означает каждый параметр

Общая логика по коду:
- `pattern-fit` строит профиль паттерна на `normal_period` и `event_period`: baseline по дням/категориям, рост, seed pool, кластеры, fit bundle.
- `pattern-monitor` применяет fit bundle к новым строкам и считает row score + daily pressure/state + alerts.
- YAML (`analysis.pattern_monitoring`) — основной источник значений.
- CLI перекрывает YAML для части параметров:
  - `pattern-fit --label-source --normal-period --event-period` перекрывают `label_source`, `normal_period`, `event_period`.
  - `pattern-monitor --label-source` перекрывает `label_source`.
  - В `pattern-monitor` диапазон данных задается CLI-параметрами `--date-from/--date-to` или `--month` (это не поля `PatternMonitoringConfig`).

Ниже — параметры именно `PatternMonitoringConfig`, сгруппированные по смыслу.

#### `enabled`
- Где используется: **по текущему коду напрямую не используется**.
- Тип: `bool`
- Допустимые значения: `true/false`
- Значение по умолчанию: `true`
- Что делает: флаг присутствует в конфиге, но в `pattern-fit`/`pattern-monitor` не проверяется.
- На что влияет: напрямую не влияет.
- Когда увеличивать: н/д.
- Когда уменьшать: н/д.
- Риски: ложное ощущение, что выключает режим.
- Связанные параметры: нет.

## Источник меток и периоды

#### `label_source`
- Где используется: `оба` (через CLI-команды и `get_label_columns`).
- Тип: `Literal["llm", "pred"]`
- Допустимые значения: `llm`, `pred`
- Значение по умолчанию: `llm`
- Что делает: выбирает поля меток (`is_complaint_llm/complaint_category_llm` или `is_complaint_pred/category_pred`).
- На что влияет: candidate volume, recall/precision (через качество источника меток).
- Когда увеличивать: н/д (это не числовой параметр); переключать на `llm`, если pred-модель слабая на нужном кейсе.
- Когда уменьшать: переключать на `pred`, если LLM-метки шумные/дорогие.
- Риски: неверный источник меток может дать пустой fit или шумный monitor.
- Связанные параметры: `complaints_only`, `include_other_category`, CLI `--label-source`.

#### `normal_period`
- Где используется: `pattern-fit` (как дефолт, если CLI не передал `--normal-period`).
- Тип: `str | None` (формат `YYYY-MM..YYYY-MM`)
- Допустимые значения: валидный month-range или `null`
- Значение по умолчанию: `null`
- Что делает: задает baseline-период для fit.
- На что влияет: стабильность baseline и чувствительность к росту.
- Когда увеличивать: делать период длиннее, если baseline шумный.
- Когда уменьшать: если в длинном периоде есть структурный дрейф.
- Риски: неверный период искажает z-score и список candidate-категорий.
- Связанные параметры: `event_period`, `baseline_weekday_shrink_k`, `baseline_min_std`.

#### `event_period`
- Где используется: `pattern-fit` (как дефолт, если CLI не передал `--event-period`).
- Тип: `str | None` (формат `YYYY-MM..YYYY-MM`)
- Допустимые значения: валидный month-range или `null`
- Значение по умолчанию: `null`
- Что делает: задает период «аномалии» для поиска роста и seed pool.
- На что влияет: recall и количество seed-кандидатов.
- Когда увеличивать: если в event мало строк.
- Когда уменьшать: если смешиваются разные волны событий.
- Риски: пустой/неверный период => 0 кандидатов.
- Связанные параметры: `normal_period`, `anomaly_*`, `top_growth_categories`.

#### `freq`
- Где используется: **по текущему коду напрямую не используется** (фиксированная дневная агрегация).
- Тип: `Literal["D"]`
- Допустимые значения: только `D`
- Значение по умолчанию: `D`
- Что делает: декларативное поле; в расчетах fit/monitor сейчас не переключает частоту.
- На что влияет: напрямую не влияет.
- Когда увеличивать/уменьшать: н/д.
- Риски: ожидание, что можно переключить W/M здесь.
- Связанные параметры: нет.

## Фильтрация строк и категорий

#### `complaints_only`
- Где используется: `оба`.
- Тип: `bool`
- Допустимые значения: `true/false`
- Значение по умолчанию: `true`
- Что делает: оставляет только строки с `is_complaint=True`.
- На что влияет: precision ↑, recall ↓, candidate volume ↓.
- Когда увеличивать (включать): когда много нерелевантного шума.
- Когда уменьшать (выключать): для debug-recall при 0 попаданий.
- Риски: можно отрезать «пограничные» кейсы до построения профиля.
- Связанные параметры: `label_source`, `include_other_category`.

#### `include_other_category`
- Где используется: `оба`.
- Тип: `bool`
- Допустимые значения: `true/false`
- Значение по умолчанию: `false`
- Что делает: включает/исключает категорию `OTHER`.
- На что влияет: recall ↑ при `true`, но precision и интерпретируемость ↓.
- Когда увеличивать: если подозрение, что нужный паттерн тонет в `OTHER`.
- Когда уменьшать: для production-стабильности и меньшего шума.
- Риски: `OTHER` часто смешивает разнотипные тексты.
- Связанные параметры: `complaints_only`, `top_growth_categories`.

#### `min_rows_per_category`
- Где используется: **по текущему коду напрямую не используется**.
- Тип: `int`
- Допустимые значения: целое `>=0`
- Значение по умолчанию: `80`
- Что делает: поле есть в конфиге, но в fit/monitor нет прямой проверки.
- На что влияет: напрямую не влияет.
- Когда увеличивать/уменьшать: н/д.
- Риски: можно ошибочно тюнить «мертвый» параметр.
- Связанные параметры: `min_normal_rows_per_category`, `min_event_rows_per_category`.

#### `min_normal_rows_per_category`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: целое `>=1`
- Значение по умолчанию: `120`
- Что делает: минимальный размер normal-части категории для построения профиля.
- На что влияет: стабильность профиля vs recall по редким категориям.
- Когда увеличивать: если кластеры нестабильны/шумные.
- Когда уменьшать: первый тюнинг при 0 попаданий.
- Риски: слишком высоко => почти все категории отсекаются.
- Связанные параметры: `min_event_rows_per_category`, `vectorizer_source`, `cluster_method`.

#### `min_event_rows_per_category`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: целое `>=1`
- Значение по умолчанию: `60`
- Что делает: минимальный размер event-части категории для fit.
- На что влияет: recall по редким всплескам.
- Когда увеличивать: когда много случайных/мелких категорий.
- Когда уменьшать: один из первых шагов при пустом fit.
- Риски: слишком низко => слабый/нестабильный профиль.
- Связанные параметры: `min_normal_rows_per_category`, `top_growth_categories`.

## Текст и preprocessing

#### `text_field`
- Где используется: `оба`.
- Тип: `str`
- Допустимые значения: имя текстовой колонки в parquet.
- Значение по умолчанию: `client_first_message`
- Что делает: базовый источник текста (если не включен `use_first_message_only`).
- На что влияет: качество эмбеддингов и разделимость паттерна.
- Когда увеличивать: переключить на более информативное поле (напр. `dialog_text`).
- Когда уменьшать: вернуться к короткому полю, если длинный текст сильно шумит.
- Риски: несуществующее поле => fallback, но сигнал может ухудшиться.
- Связанные параметры: `use_first_message_only`, `strip_system_speakers`.

#### `use_first_message_only`
- Где используется: `оба` (через `build_text_clean`).
- Тип: `bool`
- Допустимые значения: `true/false`
- Значение по умолчанию: `true`
- Что делает: принудительно берет `client_first_message` при наличии.
- На что влияет: precision может вырасти (меньше шума), recall может упасть (теряется контекст).
- Когда увеличивать (включать): для стабильного, короткого, чистого сигнала.
- Когда уменьшать (выключать): early-debug при 0 попаданий, чтобы вернуть полный контекст.
- Риски: потеря критичных деталей, которые появляются позже в диалоге.
- Связанные параметры: `text_field`, `strip_system_speakers`.

#### `strip_system_speakers`
- Где используется: `оба`.
- Тип: `bool`
- Допустимые значения: `true/false`
- Значение по умолчанию: `true`
- Что делает: удаляет маркеры `CLIENT:/OPERATOR:/CHATBOT:`.
- На что влияет: чистота текста, снижает технический шум.
- Когда увеличивать: почти всегда полезно держать `true`.
- Когда уменьшать: если сами маркеры несут полезный signal в конкретной разметке.
- Риски: редкий кейс потери служебного паттерна роли.
- Связанные параметры: `use_first_message_only`, `text_field`.

## Baseline / growth detection

#### `baseline_weekday_shrink_k`
- Где используется: `pattern-fit` (и косвенно `pattern-monitor` через fit bundle baseline params).
- Тип: `float`
- Допустимые значения: `>0`
- Значение по умолчанию: `5.0`
- Что делает: силу shrinkage weekday mean к global mean.
- На что влияет: стабильность baseline, чувствительность к всплескам.
- Когда увеличивать: если weekday-оценки шумные/редкие.
- Когда уменьшать: если есть много истории и нужен более «острый» weekday baseline.
- Риски: слишком большое значение сглаживает реальные weekday-эффекты.
- Связанные параметры: `baseline_min_std`, `anomaly_z_threshold`.

#### `baseline_min_std`
- Где используется: `оба` (fit при построении baseline и monitor при day anomaly из fit bundle).
- Тип: `float`
- Допустимые значения: `>0`
- Значение по умолчанию: `1.0`
- Что делает: нижняя граница std в z-score.
- На что влияет: alert sensitivity, стабильность z-score.
- Когда увеличивать: если ложных аномалий слишком много.
- Когда уменьшать: если z-score «глухой» и не видит рост.
- Риски: слишком низко => взрыв ложных пиков.
- Связанные параметры: `anomaly_z_threshold`, `day_alert_threshold`.

#### `anomaly_z_threshold`
- Где используется: `pattern-fit`.
- Тип: `float`
- Допустимые значения: обычно `>=0`
- Значение по умолчанию: `2.5`
- Что делает: порог дней over-threshold при расчете growth summary.
- На что влияет: консервативность отбора candidate-категорий.
- Когда увеличивать: если fit захватывает слишком много случайных категорий.
- Когда уменьшать: один из первых рычагов при 0 кандидатов.
- Риски: слишком низко => шумные категории в кандидаты.
- Связанные параметры: `anomaly_min_event_days`, `top_growth_categories`.

#### `anomaly_min_excess_total`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: `>=0`
- Значение по умолчанию: `20`
- Что делает: минимум суммарного positive excess для candidate.
- На что влияет: candidate volume и recall.
- Когда увеличивать: при избыточном числе слабых кандидатов.
- Когда уменьшать: ранний шаг при пустом fit.
- Риски: слишком высокий порог убивает редкие паттерны.
- Связанные параметры: `anomaly_z_threshold`, `anomaly_min_event_days`.

#### `anomaly_min_event_days`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: `>=0`
- Значение по умолчанию: `3`
- Что делает: минимум дней, где zscore >= threshold.
- На что влияет: устойчивость сигнала (одноразовый всплеск vs продолжающийся паттерн).
- Когда увеличивать: если нужно исключить одноразовые шумы.
- Когда уменьшать: при коротком event периоде или 0 попаданий.
- Риски: большой порог пропускает короткие, но реальные инциденты.
- Связанные параметры: `anomaly_z_threshold`, `event_period`.

#### `top_growth_categories`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: `>=1`
- Значение по умолчанию: `12`
- Что делает: top-N категорий по growth_score включаются как кандидаты даже если rule-threshold не пройден.
- На что влияет: recall (safety-net) и число категорий в fit.
- Когда увеличивать: при 0/мало кандидатов.
- Когда уменьшать: если fit слишком широкий и шумный.
- Риски: большое значение снижает селективность.
- Связанные параметры: `anomaly_*`, `include_other_category`.

## Векторизация и пространство признаков

#### `vectorizer_source`
- Где используется: `pattern-fit`.
- Тип: `Literal["trained", "fit_normal"]`
- Допустимые значения: `trained`, `fit_normal`
- Значение по умолчанию: `trained`
- Что делает: источник векторизатора для category space.
- На что влияет: переносимость signal и чувствительность к domain shift.
- Когда увеличивать: переключать на `fit_normal` при явном mismatch обученного vectorizer и текущих данных.
- Когда уменьшать: возвращать `trained` для консистентности со stable training pipeline.
- Риски: `fit_normal` на малом наборе может переобучиться на локальный шум.
- Связанные параметры: `svd_components`, `random_state`, `within_category_method`.

#### `svd_components`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: `>=2` (дальше клиппинг по размерам матрицы).
- Значение по умолчанию: `200`
- Что делает: размер латентного пространства SVD.
- На что влияет: разделимость кластеров и стабильность distance/similarity.
- Когда увеличивать: если теряется семантика в слишком грубой проекции.
- Когда уменьшать: если данных мало и пространство шумное.
- Риски: слишком много компонент на малых данных => нестабильные кластеры.
- Связанные параметры: `vectorizer_source`, `cluster_method`, `min_cluster_size`.

#### `random_state`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: целое.
- Значение по умолчанию: `42`
- Что делает: фиксирует stochastic-части (SVD).
- На что влияет: воспроизводимость.
- Когда увеличивать/уменьшать: менять только для диагностики чувствительности.
- Риски: без фиксации сложнее сравнивать итерации fit.
- Связанные параметры: `svd_components`.

#### `within_category_method`
- Где используется: `pattern-fit`.
- Тип: `Literal["knn_cosine", "centroid_delta"]`
- Допустимые значения: `knn_cosine`, `centroid_delta`
- Значение по умолчанию: `knn_cosine`
- Что делает: способ расчета novelty_to_normal в event-строках.
- На что влияет: seed ranking, cluster density, recall.
- Когда увеличивать: переключать на `centroid_delta` для более «глобального» сигнала.
- Когда уменьшать: использовать `knn_cosine` для локальной чувствительности.
- Риски: метод может быть чувствителен к структуре категории.
- Связанные параметры: `knn_k`, `vectorizer_source`.

#### `knn_k`
- Где используется: `pattern-fit` (для `within_category_method=knn_cosine`).
- Тип: `int`
- Допустимые значения: `>=1`
- Значение по умолчанию: `15`
- Что делает: число соседей normal-space при novelty.
- На что влияет: гладкость novelty (малый k — более резкий).
- Когда увеличивать: если novelty слишком шумная.
- Когда уменьшать: если теряются локальные аномальные хвосты.
- Риски: чрезмерно большой k размывает редкий паттерн.
- Связанные параметры: `within_category_method`, `min_total_seeds_per_category`.

## Seed pool

#### `per_day_seed_quota_mode`
- Где используется: `pattern-fit`.
- Тип: `Literal["residual", "percent", "fixed"]`
- Допустимые значения: `residual`, `percent`, `fixed`
- Значение по умолчанию: `residual`
- Что делает: правило дневной квоты seed внутри категории.
- На что влияет: candidate volume и распределение seed по дням.
- Когда увеличивать: перейти на `percent`/`fixed`, если residual слишком жесткий и даёт мало seed.
- Когда уменьшать: вернуться к `residual`, если seed перегружен шумом.
- Риски: неудачный режим может перекосить seed в несколько дней.
- Связанные параметры: `per_day_seed_percent`, `per_day_seed_fixed`, `min_total_seeds_per_category`.

#### `per_day_seed_percent`
- Где используется: `pattern-fit` (только при `per_day_seed_quota_mode=percent`).
- Тип: `float`
- Допустимые значения: обычно `0..1+` (по коду не зажат, но практично `0..1`).
- Значение по умолчанию: `0.3`
- Что делает: доля top novelty строк на день.
- На что влияет: объём seed и recall.
- Когда увеличивать: если seed pool слишком мал.
- Когда уменьшать: если много шумных seed.
- Риски: >1 фактически превращается в почти полный дневной отбор.
- Связанные параметры: `per_day_seed_quota_mode`, `max_total_seeds_per_category`.

#### `per_day_seed_fixed`
- Где используется: `pattern-fit` (только при `per_day_seed_quota_mode=fixed`).
- Тип: `int`
- Допустимые значения: `>=0`
- Значение по умолчанию: `10`
- Что делает: фиксированное число seed на день.
- На что влияет: равномерность/объём seed.
- Когда увеличивать: при недоборе seed.
- Когда уменьшать: при переизбытке шума.
- Риски: игнорирует реальную силу day-level аномалии.
- Связанные параметры: `per_day_seed_quota_mode`, `min_total_seeds_per_category`.

#### `min_total_seeds_per_category`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: `>=0`
- Значение по умолчанию: `15`
- Что делает: нижняя граница seed pool после дневных квот (fallback top novelty).
- На что влияет: recall и шанс получить кластеры.
- Когда увеличивать: при нестабильной кластеризации из-за малого seed.
- Когда уменьшать: когда seed заполняется сомнительными строками.
- Риски: большое значение может принудительно тянуть шумные строки.
- Связанные параметры: `per_day_seed_quota_mode`, `min_cluster_size`.

#### `max_total_seeds_per_category`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: `>=1`
- Значение по умолчанию: `500`
- Что делает: верхняя граница seed pool.
- На что влияет: compute cost и шум в кластерах.
- Когда увеличивать: если категория очень крупная и обрезка вредна.
- Когда уменьшать: если fit слишком тяжелый/шумный.
- Риски: слишком низко — потеря редких подпаттернов.
- Связанные параметры: `min_total_seeds_per_category`, `cluster_method`.

## Кластеризация

#### `cluster_method`
- Где используется: `pattern-fit`.
- Тип: `Literal["optics", "dbscan", "agglomerative"]`
- Допустимые значения: `optics`, `dbscan`, `agglomerative`
- Значение по умолчанию: `optics`
- Что делает: алгоритм кластеризации seed.
- На что влияет: cluster density, число outlier, стабильность профилей.
- Когда увеличивать: переключить на `agglomerative`, если нужно фиксированно компактные группы.
- Когда уменьшать: `optics/dbscan`, если важна форма плотностей и выбросы.
- Риски: неподходящий метод может свалить всё в `-1`/один кластер.
- Связанные параметры: `min_cluster_size`, `max_clusters_per_category`, `svd_components`.

#### `min_cluster_size`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: `>=1`
- Значение по умолчанию: `10`
- Что делает: минимальный размер валидного кластера и порог fallback (если seed < min_cluster_size, все в cluster 0).
- На что влияет: cluster density, recall редких групп.
- Когда увеличивать: если много мелких шумных кластеров.
- Когда уменьшать: один из первых рычагов при 0/очень мало кластеров.
- Риски: слишком большой порог убивает редкие важные группы.
- Связанные параметры: `min_total_seeds_per_category`, `cluster_method`.

#### `max_clusters_per_category`
- Где используется: `pattern-fit`.
- Тип: `int`
- Допустимые значения: `>=1`
- Значение по умолчанию: `5`
- Что делает: обрезает число сохраняемых кластеров на категорию.
- На что влияет: детализация профиля vs стабильность.
- Когда увеличивать: если категория многомодальная.
- Когда уменьшать: если профили слишком раздроблены.
- Риски: слишком низко скрывает вторичные паттерны.
- Связанные параметры: `cluster_method`, `min_cluster_size`.

## Scoring row-level (monitor)

#### `score_w_event_similarity`
- Где используется: `pattern-monitor` (в `pattern_like_raw`).
- Тип: `float`
- Допустимые значения: числовые (обычно `0..1`)
- Значение по умолчанию: `0.45`
- Что делает: вес сходства с event centroids.
- На что влияет: чувствительность к «похожести на event».
- Когда увеличивать: если хотите сильнее ловить event-like формулировки.
- Когда уменьшать: если много ложных alert по похожим, но нормальным кейсам.
- Риски: перекос в сторону lexical similarity.
- Связанные параметры: `score_topk_event_neighbors`, `row_alert_threshold`.

#### `score_w_normal_distance`
- Где используется: `pattern-monitor`.
- Тип: `float`
- Допустимые значения: числовые
- Значение по умолчанию: `0.25`
- Что делает: вес «дальности от normal centroid».
- На что влияет: novelty-чувствительность.
- Когда увеличивать: если паттерн проявляется как отклонение от нормы.
- Когда уменьшать: если новизна слишком шумная.
- Риски: может усиливать outlier-шум.
- Связанные параметры: `score_topk_normal_neighbors`, `row_alert_threshold`.

#### `score_w_category_anomaly`
- Где используется: `pattern-monitor`.
- Тип: `float`
- Допустимые значения: числовые
- Значение по умолчанию: `0.30`
- Что делает: вес day-level category anomaly.
- На что влияет: связь row-score с дневным давлением категории.
- Когда увеличивать: если важно «подтягивать» ряды в реально аномальные дни.
- Когда уменьшать: если нужно больше row-level, меньше day-level влияния.
- Риски: может завышать score в дни общего всплеска без специального подпаттерна.
- Связанные параметры: `day_alert_threshold`, `daily_pressure_mode`.

#### `score_topk_event_neighbors`
- Где используется: `pattern-monitor`.
- Тип: `int`
- Допустимые значения: `>=1`
- Значение по умолчанию: `10`
- Что делает: top-k event centroid similarities для усреднения.
- На что влияет: гладкость event_similarity.
- Когда увеличивать: если centroids много и нужны более сглаженные оценки.
- Когда уменьшать: если важны «острые» близости к лучшим центроидам.
- Риски: слишком большой k размывает сигнал.
- Связанные параметры: `score_w_event_similarity`.

#### `score_topk_normal_neighbors`
- Где используется: `pattern-monitor`.
- Тип: `int`
- Допустимые значения: `>=1`
- Значение по умолчанию: `20`
- Что делает: kNN размер для novelty_to_normal.
- На что влияет: устойчивость normal-distance оценки.
- Когда увеличивать: чтобы уменьшить шум distance.
- Когда уменьшать: чтобы усилить локальные отклонения.
- Риски: крайние значения дают или шум, или излишнюю инерцию.
- Связанные параметры: `score_w_normal_distance`.

#### `row_alert_threshold`
- Где используется: `pattern-monitor`.
- Тип: `float`
- Допустимые значения: обычно `0..1`
- Значение по умолчанию: `0.65`
- Что делает: порог `is_pattern_alert` на строке.
- На что влияет: alert sensitivity и объём алертов.
- Когда увеличивать: если слишком много false positives.
- Когда уменьшать: если алертов слишком мало.
- Риски: слишком низко -> шум; слишком высоко -> пропуски.
- Связанные параметры: `score_w_*`, `daily_top_k_rows`.

> Примечание по коду: `pattern-fit` сохраняет `score_w_*` и `row_alert_threshold` в fit bundle как метаданные, но monitor считает score по **текущему cfg_pm**, то есть это monitor-тюнинг.

## Daily pressure / state (monitor)

#### `daily_top_k_rows`
- Где используется: `pattern-monitor`.
- Тип: `int`
- Допустимые значения: `>=1`
- Значение по умолчанию: `10`
- Что делает: сколько top row-score строк на категорию/день участвует в давлении.
- На что влияет: day-level sensitivity.
- Когда увеличивать: если паттерн размазан по многим строкам.
- Когда уменьшать: если хотите акцент только на самых сильных кейсах.
- Риски: высокий k может «раздувать» daily pressure.
- Связанные параметры: `daily_pressure_mode`, `day_alert_threshold`.

#### `daily_pressure_mode`
- Где используется: `pattern-monitor`.
- Тип: `Literal["sum_topk", "one_minus_prod"]`
- Допустимые значения: `sum_topk`, `one_minus_prod`
- Значение по умолчанию: `sum_topk`
- Что делает: формулу агрегации row-score в category_pressure.
- На что влияет: масштаб и нелинейность day pressure.
- Когда увеличивать: `sum_topk` — когда нужен более линейный рост давления.
- Когда уменьшать: `one_minus_prod` — когда нужен saturating-эффект и меньше взрывов при множестве средних score.
- Риски: выбор формулы сильно меняет day-level поведение.
- Связанные параметры: `daily_top_k_rows`, `day_alert_threshold`.

#### `state_alpha`
- Где используется: `pattern-monitor`.
- Тип: `float`
- Допустимые значения: обычно `0..1`
- Значение по умолчанию: `0.35`
- Что делает: сглаживание `smoothed_state = alpha*overall + (1-alpha)*prev`.
- На что влияет: стабильность state и реакция на всплески.
- Когда увеличивать: если нужно быстрее реагировать.
- Когда уменьшать: если нужно сильнее подавлять краткосрочный шум.
- Риски: высокий alpha даёт «дерганый» state.
- Связанные параметры: `day_alert_threshold`, `daily_pressure_mode`.

#### `day_alert_threshold`
- Где используется: `pattern-monitor`.
- Тип: `float`
- Допустимые значения: обычно `0..1`
- Значение по умолчанию: `0.50`
- Что делает: порог категории по `category_pressure` для дневного счетчика.
- На что влияет: day-level alert sensitivity.
- Когда увеличивать: если много ложных «тревожных» дней.
- Когда уменьшать: если мониторинг слишком консервативен.
- Риски: слишком низкий порог «краснит» почти каждый день.
- Связанные параметры: `daily_top_k_rows`, `daily_pressure_mode`, `state_alpha`.

## Пути к артефактам

#### `interim_dir`
- Где используется: `оба`.
- Тип: `str`
- Допустимые значения: путь к директории.
- Значение по умолчанию: `data/interim`
- Что делает: корень parquet/joblib артефактов fit/monitor.
- На что влияет: воспроизводимость/доступность артефактов.
- Когда увеличивать/уменьшать: н/д (это path).
- Риски: смена пути без миграции => monitor не находит fit bundle.
- Связанные параметры: `exports_dir`, `reports_dir`, `tag`.

#### `exports_dir`
- Где используется: `оба`.
- Тип: `str`
- Допустимые значения: путь.
- Значение по умолчанию: `exports`
- Что делает: xlsx-экспорты fit/monitor.
- На что влияет: удобство ручной проверки.
- Когда увеличивать/уменьшать: н/д.
- Риски: неверный путь -> нет ожидаемых выгрузок.
- Связанные параметры: `reports_dir`, `interim_dir`.

#### `reports_dir`
- Где используется: `оба`.
- Тип: `str`
- Допустимые значения: путь.
- Значение по умолчанию: `reports`
- Что делает: html-отчеты fit/monitor.
- На что влияет: наблюдаемость процесса.
- Когда увеличивать/уменьшать: н/д.
- Риски: неверный путь усложняет отладку.
- Связанные параметры: `exports_dir`, `interim_dir`.

### Какие параметры менять первыми, если `pattern-fit` дает 0 попаданий

Рекомендуемая последовательность (от «мягких» к более структурным):
1. Проверить источник/периоды:
   - `label_source` (и CLI override `--label-source`),
   - корректность `normal_period`/`event_period` и что в event реально есть строки.
2. Ослабить жесткие фильтры:
   - `complaints_only=false` (в дебаге),
   - `include_other_category=true` (в дебаге),
   - `use_first_message_only=false`.
3. Ослабить пороги отбора категорий:
   - уменьшить `min_normal_rows_per_category`, `min_event_rows_per_category`,
   - уменьшить `anomaly_z_threshold`, `anomaly_min_excess_total`, `anomaly_min_event_days`,
   - увеличить `top_growth_categories`.
4. Ослабить seed/cluster-консервативность:
   - `vectorizer_source=fit_normal` (диагностически),
   - сменить `within_category_method` (`knn_cosine` <-> `centroid_delta`),
   - `per_day_seed_quota_mode=percent` и поднять `per_day_seed_percent`,
   - увеличить `min_total_seeds_per_category`,
   - для дебага попробовать `cluster_method=agglomerative`, уменьшить `min_cluster_size`.

Практический смысл этих рычагов:
- recall повышают: `complaints_only=false`, `include_other_category=true`, `use_first_message_only=false`, снижение `min_*rows`, снижение `anomaly_*`, увеличение `top_growth_categories`, мягкие seed/cluster настройки;
- слишком консервативные: `complaints_only=true` + `include_other_category=false` + высокие `min_*rows` + высокие `anomaly_*` + маленький `top_growth_categories` + большой `min_cluster_size`.

### Какие параметры относятся уже к monitor/scoring, а не к fit

В первую очередь monitor-stage (row/day alerts), а не построение fit-профиля:
- `score_w_event_similarity`
- `score_w_normal_distance`
- `score_w_category_anomaly`
- `score_topk_event_neighbors`
- `score_topk_normal_neighbors`
- `daily_top_k_rows`
- `daily_pressure_mode`
- `state_alpha`
- `row_alert_threshold`
- `day_alert_threshold`

Если проблема: «fit построился, но monitor слишком шумный/слишком тихий» — крутите именно эти параметры.
Если проблема: «fit пустой / почти нет категорий / seed не формируется» — крутите fit-параметры из предыдущего раздела.

### Пример YAML для режима «мягкий fit / debug recall» (не production-default)

```yaml
analysis:
  pattern_monitoring:
    label_source: llm
    complaints_only: false
    include_other_category: true
    use_first_message_only: false

    min_normal_rows_per_category: 40
    min_event_rows_per_category: 20

    anomaly_z_threshold: 1.5
    anomaly_min_excess_total: 5
    anomaly_min_event_days: 1
    top_growth_categories: 25

    vectorizer_source: fit_normal
    within_category_method: knn_cosine
    knn_k: 10

    per_day_seed_quota_mode: percent
    per_day_seed_percent: 0.5
    min_total_seeds_per_category: 20
    max_total_seeds_per_category: 800

    cluster_method: agglomerative
    min_cluster_size: 5
    max_clusters_per_category: 8
```

Это диагностический профиль для поиска причин «пустого fit». После нахождения сигнала параметры обычно ужесточают обратно.

### Быстрый чеклист перед повторным запуском `pattern-fit`

- Выбран ли правильный `label_source` (и не перекрыт ли он CLI-опцией)?
- Точно ли в `event_period` есть достаточное число строк нужной категории?
- Не слишком ли жесткие фильтры (`complaints_only`, `include_other_category`, `min_*rows`)?
- Не отрезали ли сигнал опцией `use_first_message_only=true`?
- Не переужаты ли пороги роста (`anomaly_*`, `top_growth_categories`)?
- Не «съедает» ли сигнал preprocessing (очистка/stopwords/strip speaker markers)?

### Что делает каждый режим (тезисно)

#### `pattern-fit`
- Берёт historical период и делит его на `normal-period` и `event-period`.
- Строит baseline по дневным count категорий и выделяет категории с аномальным ростом.
- Для выросших категорий выделяет event-like seed-строки внутри категории.
- Кластеризует seed-строки и строит event-профили (centroid/top terms/examples).
- Сохраняет fit bundle и артефакты, которые затем использует `pattern-monitor`.

#### `pattern-monitor`
- Загружает ранее построенный fit bundle по `tag`.
- Скорит новые жалобы по похожести на event-профили vs normal-профили категорий.
- Считает `category_daily_pressure` и `overall_daily_state` со сглаживанием.
- Формирует row/day alerts и экспортирует результаты в parquet/xlsx/html.

### 1) Обучение профиля паттерна

```bash
python -m complaints_trends.cli pattern-fit \
  --config configs/project.yaml \
  --tag mig_2025_q4 \
  --normal-period 2025-01..2025-08 \
  --event-period 2025-11..2025-12 \
  --label-source llm
```

`pattern-fit` строит baseline по категориям на normal-period, находит выросшие категории в event-period, выделяет внутри них event-like seed-строки и кластеризует их.

### 2) Мониторинг новых дней/месяцев

По диапазону дат из historical витрины:

```bash
python -m complaints_trends.cli pattern-monitor \
  --config configs/project.yaml \
  --tag mig_2025_q4 \
  --date-from 2025-12-15 \
  --date-to 2025-12-15 \
  --label-source llm
```

По `month_YYYY-MM.parquet` после `infer-month`:

```bash
python -m complaints_trends.cli pattern-monitor \
  --config configs/project.yaml \
  --tag mig_2025_q4 \
  --month 2026-01 \
  --label-source pred \
  --force-materialize
```

### Артефакты и валидация

`pattern-fit` сохраняет в `data/interim/pattern_fit_<tag>/`:
- `category_growth_summary.parquet`
- `seed_pool.parquet`
- `cluster_members.parquet`
- `cluster_profiles.json`
- `fit_bundle.joblib`

`pattern-monitor` сохраняет в `data/interim/pattern_monitor_<tag>/`:
- `scored_rows.parquet`
- `category_daily_pressure.parquet`
- `overall_daily_state.parquet`

Также формируются:
- `exports/pattern_fit_<tag>.xlsx`, `reports/pattern_fit_<tag>.html`
- `exports/pattern_monitor_<tag>.xlsx`, `reports/pattern_monitor_<tag>.html`

## Interactive Dashboard Service (FastAPI + React)

Добавлен MVP интерактивного сервиса из двух частей:

1. **FastAPI backend** (`src/complaints_trends/api`) — тонкий API-слой поверх существующего engine/артефактов (`prepare`, `viz-build`, `pattern-fit`, `pattern-monitor`).
2. **React + TypeScript frontend** (`apps/dashboard`) — SPA для Overview / Categories / Timeseries / Pattern Fit / Pattern Monitor / Reports / Settings.

### Запуск backend

```bash
python -m complaints_trends.cli api-serve --config configs/project.yaml --host 0.0.0.0 --port 8000
```

### Запуск frontend

```bash
npm install --prefix apps/dashboard
npm run dev --prefix apps/dashboard
```

### Как это связано с существующим pipeline

- Существующие CLI-команды не удалены и не заменены.
- API-роуты читают существующие parquet/json/joblib артефакты и отдают JSON.
- Run endpoints (`/api/runs/*`) запускают существующие python-функции: `viz-build`, `pattern-fit`, `pattern-monitor`, `infer-month`.

### Основные endpoints

- `GET /api/health`, `GET /api/meta/config`, `GET /api/meta/datasets`, `GET /api/meta/tags`
- `GET /api/overview`
- `GET /api/categories`, `GET /api/categories/{category}/...`
- `GET /api/timeseries/...`
- `GET /api/pattern-fit/...`
- `GET /api/pattern-monitor/...`
- `POST /api/reports/executive|operations|pattern-monitoring`
- `POST /api/runs/viz-build|pattern-fit|pattern-monitor|infer-month`

### Страницы dashboard

- `/overview`
- `/categories`
- `/timeseries`
- `/pattern-fit`
- `/pattern-monitor`
- `/reports`
- `/settings`


## Executive report (/reports)

Вкладка `/reports` переработана в executive dashboard для руководства с фокусом на 30–60 секунд понимания ситуации:

- **Минимальные controls**: period, compare mode, custom baseline (только для `custom_range`), category scope, toggles `include examples` и `include ownership`, кнопки `Build`, `Reset`, `Download HTML`, `Download Markdown`, `Print view`.
- **KPI cards**: total complaints, delta vs baseline, categories above baseline, top growth category, pattern risk, primary area/owner (опционально).
- **4 ключевых блока**: `Actual vs Expected`, вклад категорий в рост, список категорий выше baseline с priority, top alert examples.
- **Executive summary**: детерминированный блок `headline + bullets + recommended actions` от backend (без LLM).
- **Glossary tooltips**: у ключевых терминов (baseline, expected, delta, anomaly, pattern risk, contribution, owner/area, priority и др.) доступны hover-пояснения на русском.

### Новый API контракт executive report

`POST /api/reports/executive`

Request:
- `date_from`, `date_to`
- `compare_mode`: `previous_period|same_weekday|seasonal|custom_range`
- `baseline_date_from`, `baseline_date_to` (для custom range)
- `categories` (optional)
- `include_examples`
- `include_ownership`
- `pattern_tag` (optional, default `latest`)

Response:
- `meta`
- `kpis`
- `charts` (`actual_expected`, `category_contribution`, `category_priority`, `alert_examples`)
- `summary`
- `definitions`
- `export` (`markdown`, `html`)

### Ownership mapping (optional)

Если есть файл:
- `configs/category_ownership.csv` или
- `data/reference/category_ownership.csv`

то backend рассчитывает `primary_area` (наиболее вероятный контур для первичной проверки).
Если файла нет — блок ownership корректно скрывается, ошибок нет.

### Pattern risk в executive report

`Pattern risk` — это не метрика объема жалоб, а индикатор вероятности сохранения ранее выявленного нетипичного проблемного сценария.

High-level расчет (диапазон `0..1`):
- `state_component` из `overall_daily_state` (приоритет `smoothed_state`, fallback `overall_pressure`),
- `alert_component` как доля alert-строк в `scored_rows`,
- `pressure_component` как доля дней с pressure в `category_daily_pressure`.

Итоговая формула при полном наборе данных:
`0.60 * state + 0.25 * alerts + 0.15 * pressure`.

Если доступна только часть данных, используется degraded-режим (`state_only` или `alerts_only`).
Если pattern-monitor артефактов нет, возвращается `unavailable`.

## Interactive Dashboard: Timeseries tab and category scope filter

В дашборде обновлена вкладка `/timeseries` как основной экран динамики:
- `Actual vs Expected`,
- `Daily Delta/Excess`,
- `Cumulative`,
- структура категорий (stacked/lines + 100% share),
- `weekday x hour` heatmap,
- calendar heatmap,
- compare summary и вклад категорий (contribution table/chart).

### Единый category scope filter

В глобальный `FilterBar` вынесен общий фильтр категории с режимами:
- `Top N` (дефолт `10`),
- `Custom` (ручной multiselect),
- `All`.

Поддерживаются параметры:
- `categoryMode=top|custom|all`
- `topN=10`
- `categories=cat1,cat2`
- `includeOther=true|false`

Для режима `Top N` backend сам рассчитывает топ категорий за выбранный период.
`includeOther=true` агрегирует хвост в `OTHER` для графиков структуры.

### Сравнение с baseline

Во вкладке `/timeseries` сравнение поддерживает baseline mode:
- `previous_period`
- `same_weekday`
- `seasonal`
- `custom_range`

Также доступна таблица category compare:
`actual_count`, `expected_count`, `delta_abs`, `delta_pct`, `share`, `contribution_to_growth`, `anomaly_score`.

### Как читать графики Timeseries (объяснение для руководителя)

Ниже — практическая расшифровка каждого блока вкладки `/timeseries`: что он показывает, как его читать и какой управленческий вывод из него делать.

#### 1) Actual vs Expected
**Что показывает:**
- `Actual` — фактическое число жалоб по дням/неделям/месяцам.
- `Expected` — ожидаемый уровень (baseline) по выбранному режиму сравнения.

**Как смотреть:**
- Если `Actual` системно выше `Expected`, есть устойчивый негативный сдвиг.
- Разовые “шипы” — это инциденты, системный разрыв — это тренд.

**Что говорит начальству:**
- “Нагрузка на поддержку выше/ниже нормы”.
- “Ситуация разовая или структурная”.

#### 2) Daily Delta / Excess
**Что показывает:**
- Отклонение `Actual - Expected` в каждом периоде (столбики +/−).

**Как смотреть:**
- Положительные столбики: жалоб больше нормы.
- Отрицательные: меньше нормы.
- Серия положительных столбиков подряд — признак длительной проблемы.

**Что говорит начальству:**
- “Когда именно началось ухудшение и как долго длится”.

#### 3) Cumulative complaints
**Что показывает:**
- Накопленную сумму жалоб и ожидаемого уровня.

**Как смотреть:**
- Расхождение кривых = накопленный “перерасход” по жалобам.
- Чем быстрее расходятся линии, тем выше темп ухудшения.

**Что говорит начальству:**
- “Общий масштаб проблемы за период, а не только ежедневный шум”.

#### 4) Category structure over time (stacked/lines)
**Что показывает:**
- Вклад категорий в общий поток жалоб по времени.
- В режиме `stacked` — структура потока, в `lines` — динамика каждой категории отдельно.

**Как смотреть:**
- Рост “толщины” одной категории в stacked — её вклад в общий рост.
- В режиме lines удобно смотреть 1–5 приоритетных категорий.

**Что говорит начальству:**
- “Какие направления болят сильнее всего и кто формирует общий рост”.

#### 5) 100% category shares
**Что показывает:**
- Не абсолютный объем, а долю категорий внутри общего потока (сумма = 100%).

**Как смотреть:**
- Если доля категории растет при стабильном общем объеме — структура ухудшается.
- Полезно, когда общий объем «ровный», но состав жалоб меняется.

**Что говорит начальству:**
- “Меняется ли природа проблем, даже если общий уровень похож на прошлый”.

#### 6) Weekday × Hour heatmap
**Что показывает:**
- В какие дни недели и часы чаще возникает поток жалоб.

**Как смотреть:**
- Самые “горячие” клетки — пики нагрузки.
- Используется для планирования смен, SLA и коммуникаций.

**Что говорит начальству:**
- “Когда нужна усиленная операционная готовность”.

#### 7) Calendar heatmap
**Что показывает:**
- “Тепловую карту” по календарным датам.

**Как смотреть:**
- Видно сезонность, периоды всплесков, предпраздничные/послепраздничные эффекты.
- Легко сопоставлять с релизами, маркетингом, внешними событиями.

**Что говорит начальству:**
- “Какие даты/недели исторически рискованные”.

#### 8) Compare to baseline (summary)
**Что показывает:**
- Сводные метрики по выбранному периоду против baseline:
  - `actual_total`
  - `baseline_total`
  - `delta_abs`
  - `delta_pct`

**Как смотреть:**
- Это короткий “executive snapshot”: насколько хуже/лучше текущий период.

**Что говорит начальству:**
- “На сколько процентов мы выше/ниже нормы и каков абсолютный эффект”.

#### 9) Contribution chart + таблица вкладов категорий
**Что показывает:**
- Какие категории дали основной вклад в рост/снижение относительно baseline.
- Таблица с `actual_count`, `expected_count`, `delta_abs`, `delta_pct`, `share`, `contribution_to_growth`, `anomaly_score`.

**Как смотреть:**
- Сначала смотрим топ по `delta_abs` и `contribution_to_growth`.
- Затем проверяем `delta_pct` (скорость роста) и `share` (масштаб влияния).

**Что говорит начальству:**
- “Где приоритетно запускать корректирующие действия, чтобы быстрее всего снизить общий поток жалоб”.

---

### Рекомендуемый порядок анализа для руководства (1–2 минуты)
1. **Compare summary**: общий итог по периоду против нормы.
2. **Actual vs Expected + Delta bars**: когда началось отклонение и устойчиво ли оно.
3. **Contribution / category table**: какие 3–5 категорий дают основной негатив.
4. **Heatmaps**: когда по времени суток/дням недели усилять операционный контур.
5. **100% shares**: меняется ли структура проблем (не только объем).

Такой порядок позволяет быстро перейти от “есть проблема” к “где именно и что делать в первую очередь”.


## Preparation flow (upload + GigaChat prepare)

Добавлена новая вкладка `/preparation` для интерактивной подготовки нового Excel:

1. Загрузить файл (`POST /api/preparation/upload`)
2. Запустить разметку (`POST /api/preparation/{upload_id}/run`)
3. Дождаться статуса `succeeded`
4. Открыть Pattern Monitor по готовому файлу (`POST /api/preparation/jobs/{upload_id}/open-pattern-monitor`)

Что делает backend после `run`:
- прогоняет загруженный файл через существующий `prepare_dataset` (GigaChat/LLM flow не переписан),
- сохраняет per-job артефакты в `.../preparation_jobs/<upload_id>/`,
- добавляет `source_upload_id` и метаданные происхождения строк,
- безопасно merge-ит результат в основной `prepare.output_parquet`:
  - перед append удаляет старые строки этого же `source_upload_id`.

Пока job в `uploaded|queued|running`, file-scoped Pattern Monitor блокируется с причиной `preparation_not_finished`.
После `succeeded` UI автоматически может открыть `/pattern-monitor` с preset-фильтрами `uploadId` и `date range`.


## Analyst feedback loop (Pattern Monitor second layer)

Pattern Monitor now keeps architecture in **three layers**:

1. **Base candidate generator**: existing pattern-fit/pattern-monitor artifacts generate candidate rows.
2. **Analyst feedback layer**: analysts can label each row as `true`, `false`, `uncertain` and optionally add reason/comment.
3. **Optional calibrator/reranker layer**: logistic-regression model trains on analyst labels and reranks existing candidates to improve precision.

### Storage

Feedback and model registry are stored in a dedicated SQLite DB:

- `data/interim/feedback.db` (or `<pattern_monitoring.interim_dir>/feedback.db`)
- tables: `analyst_feedback`, `reranker_model_versions`, `review_sessions`

### API overview

- `POST /api/feedback`
- `POST /api/feedback/bulk`
- `GET /api/feedback`
- `GET /api/feedback/summary`
- `POST /api/pattern-monitor/calibrator/train`
- `GET /api/pattern-monitor/calibrator/versions`
- `POST /api/pattern-monitor/calibrator/{version_id}/activate`
- `POST /api/pattern-monitor/calibrator/{version_id}/deactivate`

Pattern monitor alerts endpoint supports scoring mode:

- `GET /api/pattern-monitor/alerts?scoring_mode=base|calibrated|reranked`

If no active reranker exists, API safely falls back to `base` mode and reports effective mode in response.

### UI flow

On `/pattern-monitor` page:

- switch between scoring modes (Base / Calibrated / Reranked)
- toggle review mode
- label rows inline
- view reviewed quality summary and model precision cards
- train calibrator and activate model versions

## Docker Compose quick start (one-command VM run)

You can run API + Dashboard in one command and bind your **entire local `data/` folder** from the host.

### Files added

- `docker-compose.yml`
- `docker/Dockerfile.api`
- `docker/Dockerfile.dashboard`
- `scripts/docker_up_rebuild.sh`
- `scripts/docker_down_wipe.sh`

### Start / rebuild (removes old containers first)

```bash
./scripts/docker_up_rebuild.sh /absolute/or/relative/path/to/data processed/all_prepared.parquet
```

What this does:
- bind-mounts host `data/` into container `/app/data`
- uses parquet path relative to data root (default `processed/all_prepared.parquet`)
- creates missing runtime folders (`data`, `reports`, `exports`, `models`)
- runs `docker compose up -d --build`

Mounted local paths used by app:
- `./data -> /app/data` (prepared parquet, interim artifacts, uploads/raw/processed)
- `./configs -> /app/configs` (project config)
- `./certs -> /app/certs` (TLS/mTLS certs if used)
- `./reports -> /app/reports`
- `./exports -> /app/exports`
- `./models -> /app/models`

Endpoints:
- API: `http://localhost:8000`
- Dashboard: `http://localhost:4173`

### Stop and wipe everything

```bash
./scripts/docker_down_wipe.sh /absolute/or/relative/path/to/data processed/all_prepared.parquet
```

What this does:
- `docker compose down --volumes --remove-orphans`
- removes generated runtime files in mounted data folders (`interim`, `processed`, `raw`, `uploads`) plus `reports`, `exports`, `models`
- removes parquet file resolved as `<data_dir>/<parquet_relative_path>`

### External parquet behavior

The API runtime config is generated on container start and points `prepare.output_parquet` to `"/app/data/<parquet_relative_path>"`, so all new writes go back to your mounted host `data/` folder.

### Custom download mirrors / registries

If you need corporate mirrors, you can override build-time sources via environment variables before `docker compose up`:

- `ALPINE_MIRROR` (default: `dl-cdn.alpinelinux.org`)
- `PIP_INDEX_URL`
- `PIP_TRUSTED_HOST`
- `NPM_REGISTRY`

Example:

```bash
export ALPINE_MIRROR=my.alpine.mirror.local
export PIP_INDEX_URL=https://my.pypi.mirror/simple
export PIP_TRUSTED_HOST=my.pypi.mirror
export NPM_REGISTRY=https://my.npm.mirror/
./scripts/docker_up_rebuild.sh ./data processed/all_prepared.parquet
```

## Review Dataset

Новая вкладка `/review-dataset` показывает накопленный analyst feedback dataset из SQLite.

- Хранилище: `data/interim/feedback.db`.
- Основной API:
  - `GET /api/feedback/dataset`
  - `GET /api/feedback/{row_id}`
  - `GET /api/feedback/export?output_format=csv|json`
- Доступны фильтры по verdict/pattern_tag/reviewer/date/category/subcategory/model_version/reason_code, поиск `q`, пагинация, сортировка и экспорт.

Это рабочий data-review экран: KPI, фильтры, таблица, детали строки и кнопка открытия в Pattern Monitor.

### Русские labels категорий/подкатегорий

Источник русских названий: `files.categories_seed_path` (по умолчанию `configs/categories_seed.yaml`, поля `label_ru`).

- В `analyst_feedback` хранятся и отдаются сразу обе пары полей:
  - `category` / `category_label_ru`
  - `subcategory` / `subcategory_label_ru`
- При старте API выполняется backfill: если в существующих строках `*_label_ru` пустые, они дозаполняются из taxonomy mapping (или fallback в исходный код категории/подкатегории).
- В API и UI используется `label_ru` (с fallback на код), чтобы на дашбордах отображались русские названия.

## Model Quality

Новая вкладка `/model-quality` показывает качество двухслойной схемы:

- **Layer 1 (Base):** candidate generation.
- **Layer 2 (Reranker):** prioritization of reviewed candidates.
- **Goal:** improve precision@K without losing candidate coverage.

Страница включает compare-метрики `base vs calibrated vs reranked`, реальные `precision@10/20/50/100`, разрезы по bucket/category/cluster и таблицу версий reranker.

Артефакты второго слоя:

- модели и сериализованные калибраторы: `models/rerankers/`
- версии моделей в `feedback.db`: таблица `reranker_model_versions`

## Unflagged Audit

Для оценки hidden positives за пределами candidate generator добавлен random audit flow:

- `POST /api/audit/unflagged/create`
- `GET /api/audit/unflagged/samples`
- `GET /api/audit/unflagged/{sample_id}`
- `POST /api/audit/unflagged/{sample_id}/review`

Новые таблицы в `feedback.db`:

- `unflagged_audit_samples`
- `unflagged_audit_rows`

Оценка hidden positives считается как:

- `p_hat = true_in_sample / reviewed_in_sample`
- `estimated_hidden_positives = p_hat * source_pool_size`

Это оценка по случайной выборке (не абсолютная истина).

## Dashboard on VM / external host access

Для запуска на ВМ используйте скрипты из корня проекта. `start_vm.sh` и
`restart_vm.sh` по умолчанию включают **`HTTPS_ENABLED=1`**: API обслуживается по
HTTPS на порту `18000`, dashboard — по HTTPS на `15173`, HMR — по WSS.
TLS подключён через параметры [Uvicorn](https://www.uvicorn.org/settings/#https)
и [Vite](https://vite.dev/config/server-options#server-https).

Подготовьте зависимости через `scripts/setup_vm.sh` (Ubuntu/Debian) или
`scripts/install_deps.sh`, если системные зависимости уже установлены.
Разместите выданные для вашего домена серверные сертификаты:

- `certs/server/fullchain.pem`: PEM-сертификат сервера, затем все промежуточные
  сертификаты CA в порядке от издателя сервера к корню. Корневой сертификат
  обычно не включают в отправляемую сервером цепочку: он должен быть доверен клиенту.
- `certs/server/privkey.pem`: соответствующий приватный ключ, доступный пользователю,
  от которого запускается приложение (`chmod 600 certs/server/privkey.pem`).

`PUBLIC_HOST` должен совпадать с DNS-именем в SAN сертификата и разрешаться в адрес ВМ.
При открытии по IP сертификат должен содержать этот IP в SAN. Если заданы отдельные
`API_PUBLIC_HOST` и `DASHBOARD_PUBLIC_HOST`, сертификат должен покрывать оба имени.
Серверные сертификаты хранятся отдельно от клиентских сертификатов GigaChat;
для подключения к GigaChat по-прежнему нужен его CA bundle (`certs/ca.pem` или
`GIGACHAT_CA_BUNDLE_FILE`).

Один раз скопируйте шаблон в `.env.vm` в корне проекта:

```bash
cp .env.vm.example .env.vm
chmod 600 .env.vm
```

В `.env.vm` укажите свои значения и сохраните файл:

```dotenv
PUBLIC_HOST=dashboard.example.com
HTTPS_ENABLED=1
TLS_CERT_FILE=/path/to/fullchain.pem
TLS_KEY_FILE=/path/to/privkey.pem
```

После этого запускайте без повторного указания домена и сертификатов:

```bash
scripts/start_vm.sh
# Отдельная точка входа, принудительно включающая HTTPS:
scripts/start_vm_https.sh
```

Все три скрипта (`start_vm.sh`, `start_vm_https.sh`, `restart_vm.sh`) автоматически
читают `.env.vm` из корня репозитория независимо от текущего каталога. Файл исключён
из Git; обновление кода сохраняет настройки ВМ. При обновлении проекта не копируйте
шаблон поверх уже заполненного `.env.vm`.

Адреса после запуска:

- UI: `https://dashboard.example.com:15173`
- API health: `https://dashboard.example.com:18000/api/health`

Например, для сертификатов Let's Encrypt сохраните в `.env.vm`:

```dotenv
PUBLIC_HOST=dashboard.example.com
TLS_CERT_FILE=/etc/letsencrypt/live/dashboard.example.com/fullchain.pem
TLS_KEY_FILE=/etc/letsencrypt/live/dashboard.example.com/privkey.pem
```

Если серверный сертификат и промежуточная цепочка выданы отдельными файлами:

```dotenv
TLS_CERT_FILE=/path/to/server.pem
TLS_CHAIN_FILE=/path/to/intermediates.pem
TLS_KEY_FILE=/path/to/server.key
```

Пути относительно проекта разрешаются от корня репозитория. Скрипт собирает цепочку
в `.run/tls/fullchain.pem`, проверяет формат PEM и соответствие сертификата ключу,
затем передаёт весь файл обоим серверам. При ошибке запуск прекращается без перехода
на HTTP. Для зашифрованного ключа задайте `TLS_KEY_PASSWORD` в `.env.vm` или окружении.
Для корпоративного CA можно задать `TLS_CA_FILE=/path/to/root-ca.pem`, чтобы HTTPS-прокси
dashboard доверял API; проверка TLS остаётся включённой. Доверие к этому CA в браузере
нужно настроить на клиентской машине.

После обновления сертификатов выполните:

```bash
scripts/restart_vm.sh
# Остановка:
scripts/stop_vm.sh
```

`restart_vm.sh` заново читает `.env.vm` и проверяет новые файлы до остановки работающих
процессов. Параметры из окружения имеют приоритет над файлом, например:
`API_PORT=18443 scripts/start_vm.sh`. Для другого файла настроек задайте
`VM_ENV_FILE=/path/to/vm.env`; относительный путь считается от корня репозитория.
Если явно указанный файл недоступен или содержит ошибку, запуск/перезапуск остановится.
При отсутствии обычного `.env.vm` используются прежние значения из окружения и defaults.

Формат `.env.vm` — строки `NAME=value`, комментарии с `#`, при необходимости одинарные
или двойные кавычки вокруг значения. Значения читаются буквально: `$VAR`, `~`, обратные
кавычки и shell-команды не разворачиваются, escape-последовательности не обрабатываются.
Обычный `.env` остаётся отдельным файлом и этими скриптами автоматически не читается.

Для явного запуска ВМ по HTTP в тестовом окружении есть `HTTPS_ENABLED=0`.
`PUBLIC_SCHEME` должен соответствовать этому флагу; старые HTTP URL overrides
и `VITE_HMR_PROTOCOL=ws` при включённом HTTPS отклоняются. Порты можно изменить
через `API_PORT` и `DASHBOARD_PORT`. Откройте соответствующие входящие TCP-порты
в firewall/security group ВМ. На этих портах в режиме HTTPS принимается TLS;
отдельный HTTP-редирект не запускается.

Для обычного локального запуска остаётся `scripts/start_local_stack.sh`
(`http://127.0.0.1:5173`, API `http://127.0.0.1:8000`).
