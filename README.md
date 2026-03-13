# complaints-trends

Максимально подробное руководство по проекту анализа жалоб из Excel:
- подготовка данных с **GigaChat только для нормализации/weak labels**,
- обучение и аналитика только на локальных моделях `scikit-learn`,
- инференс нового месяца, тренды MoM и поиск новых тем (novelty).

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
