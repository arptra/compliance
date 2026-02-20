# complaints-trends

Максимально подробное руководство по проекту анализа жалоб из Excel:
- подготовка данных с **GigaChat только для нормализации/weak labels**,
- обучение и аналитика только на локальных моделях `scikit-learn`,
- инференс нового месяца, тренды MoM и поиск новых тем (novelty).

---

## 1. Что решает проект

Проект предназначен для обработки ежемесячных Excel-файлов контакт-центра/поддержки (десятки колонок, длинные диалоги, шумные роли вроде `CLIENT/OPERATOR/CHATBOT`).

Ключевая задача: из полного диалога извлекать **только первый осмысленный запрос клиента** (`client_first_message`) и использовать именно его для:
1. нормализации и слабой разметки через GigaChat;
2. обучения локальных моделей;
3. инференса, трендов и поиска новых тем.

Это снижает шум от операторских реплик и служебных вставок, повышает стабильность категорий и улучшает интерпретируемость результатов.

---

## 2. Архитектура и принципы

### 2.1 Принцип разделения этапов

- **Этап подготовки (`prepare`)**: можно использовать GigaChat для нормализации и weak labels.
- **Этап обучения/инференса/сравнения (`train`, `infer-month`, `trends`, `compare`)**: только локальные ML-модели (`scikit-learn`), без LLM.

### 2.2 Почему так

- GigaChat хорошо подходит для первичной псевдо-разметки и приведения текстов к единообразной структуре.
- Локальные модели дают предсказуемую скорость, повторяемость и отсутствие внешних зависимостей в боевой аналитике.

### 2.3 mTLS только для GigaChat

Подключение GigaChat реализовано через mTLS (сертификат клиента + ключ + CA bundle), без API-ключа.

Используемая инициализация:

```python
from gigachat import GigaChat

client = GigaChat(
  base_url="https://gigachat.devices.sberbank.ru/api/v1",
  ca_bundle_file="certs/ca.pem",
  cert_file="certs/client.pem",
  key_file="certs/client.key",
  key_file_password="<optional>",
  verify_ssl_certs=True,
  timeout=60.0,
  max_retries=3,
  retry_backoff_factor=0.5
)
```

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
  month_source: "filename"
  month_regex: "(\d{4})[-_](\d{2})"
  month_column: null
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

prepare:
  pilot_month: null
  pilot_limit: 1000
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
    val_month: null
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
python -m complaints_trends.cli prepare --config configs/project.yaml --pilot --month 2025-10 --limit 1000
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

Главный файл: `configs/project.yaml`.

Ниже — ключевые блоки и их влияние.

## 5.1 `input`
- `input_dir`, `file_glob`: где искать Excel.
- `month_source`: `filename` или `column`.
- `month_regex`: как извлекать `YYYY-MM` из имени файла.
- `id_column`: если нет стабильного ID, `row_id` будет сгенерирован.
- `signal_columns`: дополнительные поля для LLM/аналитики (например `subject/channel/product/status`). **Диалоговые поля (`dialog_columns`) сюда включать не обязательно** — они и так обрабатываются отдельно.
- `dialog_column`: legacy-колонка с полным диалогом (fallback).
- `dialog_columns`: список нескольких текстовых полей (например чат/звонок/комментарий/суммаризация). Пайплайн автоматически выберет наиболее содержательное непустое поле как `raw_dialog`, а также передаст все непустые поля в `dialog_context` для GigaChat.
- `signal_columns` и `dialog_columns` логически разделены: из `signal_columns` в prompt уходят только недиалоговые поля (`signal_fields`).

**Практика:**
- Если именование файлов нестабильно, лучше `month_source: column`.
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
- `mode: mtls`: обязательный режим.
- `ca_bundle_file/cert_file/key_file`: mTLS-пути.
- `key_file_password_env`: имя env-переменной для пароля ключа.
- `max_workers`, `batch_size`: скорость/нагрузка.
- `max_text_chars`: ограничение длины входа в LLM.
- `cache_db`: sqlite-кэш ответов.
- `prompt_version`: меняйте при изменении промпта, чтобы не смешивать старые кэши.

## 5.5 `prepare`
- `pilot_month`, `pilot_limit`: пилотный режим.
- `output_parquet`, `pilot_parquet`, `pilot_review_xlsx`: куда сохранять артефакты.

## 5.6 `training`
- `text_field`: обычно `client_first_message`.
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
python -m complaints_trends.cli prepare --config configs/project.yaml --pilot --month 2025-09 --limit 5000
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

## 6.2 `train`
Команда:

```bash
python -m complaints_trends.cli train --config configs/project.yaml
```

Что делает:
1. Загружает prepared parquet.
2. Подмешивает gold-разметку из `pilot_review.xlsx` (если есть).
3. Очищает тексты от мусора/плейсхолдеров.
4. Строит TF-IDF (word + char).
5. Обучает:
   - бинарную модель жалоб,
   - модель категорий для жалоб.
6. Валидирует (time split/random split).
7. Сохраняет артефакты и training report.

Выход:
- `models/vectorizers.joblib`
- `models/complaint_model.joblib`
- `models/category_model.joblib`
- `models/label_encoder.joblib`
- `models/training_metadata.json`
- `reports/training_report.html`, `reports/training_report.md`

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
2. Извлекает `client_first_message` тем же кодом, что в train.
3. Применяет локальные модели.
4. Сохраняет разметку и отчет.

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
python -m complaints_trends.cli prepare --config configs/project.yaml --pilot --month 2025-09 --limit 5000
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

2. **Месяц не извлекается**
   - Проверьте `month_regex` и формат имен.

3. **Много мусора в summary/keywords**
   - Обновите deny tokens, prompt и логику repair.

4. **Плохая точность категории**
   - Увеличьте объем gold-меток из pilot-review,
   - пересоберите categories seed,
   - подберите `min_df/max_features`.

5. **Слишком много/мало novel topics**
   - Подстройте `threshold_percentile`, `kmeans_k`, `svd_components`.

---

## 14. Что улучшать дальше

- Добавить детальные графики (PR curve, confusion matrix heatmap, MoM stacked plots).
- Добавить отдельную команду `label-new-topics` (ручная доразметка новых кластеров).
- Добавить контроль дрейфа по токенам и каналам.
- Добавить расширенные unit/integration тесты на реальные форматы диалогов.

