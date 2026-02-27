# Staged non-LLM pipeline: Excel → clusters/complaints → December novelty

Проект работает **по этапам**, чтобы вы сначала проверили качество на малом объёме, и только потом запускали полный расчёт.

## 0) Что важно теперь
- В анализ идёт **только первое сообщение клиента** (`message_client_first`), а не весь диалог с оператором/ботом.
- Полный исходный текст сохраняется в `message_raw` (без обрезки) для контроля.
- Для ML используется `message_clean` (очищенная версия, без мусора и placeholder’ов).

---

## 1) Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2) Новый staged workflow

### Stage 1 — PILOT (один месяц)
```bash
python -m app.pilot --config configs/config.yaml --month 2025-10
```

Создаёт:
- `outputs/pilot_labeled.xlsx`
- `reports/pilot_report.md`
- `reports/pilot_cluster_summaries.csv`
- `reports/pilot_complaint_examples.md`
- `reports/pilot_approval.txt`

Что проверять в pilot:
1. Корректно ли извлекается первое сообщение клиента.
2. Понятны ли кластеры по примерам.
3. Адекватны ли complaint-примеры (top / borderline / non-complaints).

### Stage 2 — FULL TRAIN (все baseline месяцы)
```bash
python -m app.train_full --config configs/config.yaml
```

Если включено `stages.require_pilot_approval: true`, перед запуском нужно вручную записать `APPROVED` в файл:
`reports/pilot_approval.txt`.

Создаёт:
- `outputs/labeled_baseline.xlsx`
- `reports/train_report.md`
- `reports/cluster_summaries.csv`
- `reports/complaint_seed_metrics.json`
- `models/*`

### Stage 3 — December compare (сравнение с baseline)
```bash
python -m app.compare_december --config configs/config.yaml
```

Создаёт:
- `outputs/labeled_december.xlsx`
- `outputs/combined_labeled.xlsx`
- `reports/december_report.md`
- `reports/december_novel_complaints_clusters.csv`

Если Stage 2 не запускался и нет моделей — команда завершится с понятной ошибкой.

---

## 3) Backward compatibility CLI
Старые команды сохранены как алиасы:
```bash
python -m app.train --config configs/config.yaml
python -m app.predict --config configs/config.yaml
```

---

## 4) Как передавать документы

### Один общий файл
```yaml
input:
  path: "data/input.xlsx"
```

### Baseline по месяцам + отдельный декабрь
```yaml
input:
  path: null
  baseline_paths:
    - "data/2025-06.xlsx"
    - "data/2025-07.xlsx"
    - "data/2025-08.xlsx"
  december_path: "data/2025-12.xlsx"
```

---

## 5) Как передавать колонку(и) сообщений

### Одна колонка
```yaml
input:
  message_col: "message"
  message_cols: null
```

### Несколько каналов в одной строке
```yaml
input:
  message_col: null
  message_cols:
    - "message_chat"
    - "message_email"
    - "message_callcenter"
```

Они объединяются в `message_raw`, затем извлекается `message_client_first`.

---

## 6) Извлечение только первого сообщения клиента

Настройки в `input.role_parsing`:
- `client_markers`
- `operator_markers`
- `chatbot_markers`
- `client_prefix_regexes`
- `stop_on_markers`
- fallback (`first_paragraph` или `first_n_chars`)

Логика:
1. найти первый client-marker;
2. взять текст после него;
3. остановиться на первом operator/chatbot marker;
4. если marker’ов нет — fallback.

---

## 7) Как устроены текстовые поля в output

В итоговых Excel есть 3 ключевых поля:
- `message_raw` — полный исходный текст, без изменения;
- `message_client_first` — извлечённое первое сообщение клиента;
- `message_clean` — очищенный текст для ML.

---

## 8) Как убирать мусор (`!num!`, `!fio!`, placeholders и т.д.)

### Где настраивать
1. `configs/deny_tokens.txt` — запрещённые токены (никогда не должны попадать в top terms).
2. `configs/extra_stopwords.txt` — дополнительные стоп-слова вашего домена.
3. `preprocess.placeholder_patterns` в `configs/config.yaml` — regex для шаблонов мусора.

> Сейчас `configs/extra_stopwords.txt` уже расширен **максимально широко для русских разговорных диалогов** (приветствия, междометия, паразиты, короткие реакции, опечатки).
> Если станет слишком агрессивно и начнут пропадать полезные слова, удалите лишние элементы из этого файла.

### Как добавить новый мусор
Пример: хотите убрать `!acc!` и `!passport!`.

1) Добавьте в `configs/deny_tokens.txt`:
```text
!acc!
!passport!
```

2) Если это шаблон, добавьте regex в `preprocess.placeholder_patterns`, например:
```yaml
- "(?i)![a-z_]{2,20}!"
```

3) Перезапустите Stage 1 pilot и проверьте, что в `pilot_cluster_summaries.csv` и `pilot_report.md` эти артефакты исчезли.

---

## 9) Как читать отчёты

### `reports/pilot_report.md`
- раздел “How to review” — чеклист проверки качества;
- кластеры: `cluster_id`, `top_terms`, `count`, `example_messages`;
- complaint-примеры: top / borderline / non-complaints.

### `reports/train_report.md`
- итог обучения baseline;
- метрики weak supervision;
- baseline cluster overview.

### `reports/december_report.md`
- сравнение baseline vs december;
- порог новизны;
- emerging terms;
- группы новых complaint-контекстов.

---

## 10) Пример полного процесса (рекомендуемый)

```bash
# 1) pilot на одном месяце
python -m app.pilot --config configs/config.yaml --month 2025-10

# 2) вручную открыть reports/pilot_approval.txt и написать APPROVED

# 3) полный train
python -m app.train_full --config configs/config.yaml

# 4) сравнение с декабрем
python -m app.compare_december --config configs/config.yaml
```


## Почему в декабре жалоб может быть в 2 раза больше, а кластеры "не меняются"

Это ожидаемо для текущей архитектуры:
- baseline-кластеры обучаются на Stage 2 и их количество фиксировано (`clustering.n_clusters`),
- в Stage 3 декабрь **не переобучает** baseline-кластеры, а только проектируется на них.

Что теперь добавлено:
- `reports/december_complaint_clusters.csv` — отдельные кластеры **всех** жалоб декабря,
- `reports/december_report.md` показывает:
  - все жалобы декабря,
  - novel жалобы,
  - non-novel жалобы,
  - отдельные complaint-группы декабря.

Если хотите больше чувствительности к декабрю:
- увеличьте `novelty.percentile` (например, с 2 до 5),
- настройте `novelty.december_complaint_subclusters`.


## 11) Как менять настройки кластеризации (подробно)

Ниже — практическая шпаргалка, **что именно крутить** и к чему это приводит.

### 11.1 Какие модели кластеризации доступны

В `configs/config.yaml` блок `clustering`:

```yaml
clustering:
  model: "minibatch_kmeans"   # minibatch_kmeans | kmeans | birch
  n_clusters: 30
  batch_size: 4096
  birch_threshold: 0.5
  birch_branching_factor: 50
```

Поддерживаемые варианты:
1. `minibatch_kmeans` (по умолчанию)
   - Быстро на больших данных.
   - Хороший стартовый выбор для продакшн.
2. `kmeans`
   - Классический KMeans, обычно немного стабильнее на маленьких/средних данных.
   - Медленнее на больших объёмах.
3. `birch`
   - Иерархическая компрессия, удобна при очень больших выборках/потоках.
   - Может давать более "крупные" и неоднородные кластеры, если threshold подобран плохо.

---

### 11.2 Про "дистанции" (очень важно)

В этой реализации:
- Для `kmeans` / `minibatch_kmeans` внутренняя метрика модели — **евклидова** (это ограничение алгоритма).
- Для `birch` также используется евклидова геометрия при построении дерева.
- Для novelty-сравнения (и `cluster_sim_to_centroid`) используется **cosine similarity** к центроидам baseline.

Итог: в проекте одновременно используются:
- евклидова логика для обучения кластеров,
- косинусная похожесть для интерпретации близости и novelty.

Это нормально и часто используется в TF-IDF пайплайнах.

---

### 11.3 Что меняет каждый параметр

#### `clustering.model`
- Меняет сам алгоритм группировки.
- Если кластеры слишком "рвутся" или наоборот слипаются — попробуйте другой алгоритм.

#### `clustering.n_clusters`
- Больше `n_clusters` → темы детальнее, но сложнее читать и больше мелких кластеров.
- Меньше `n_clusters` → темы грубее, но понятнее и стабильнее.

#### `clustering.batch_size` (только для minibatch)
- Больше → обычно быстрее и стабильнее, но больше RAM.
- Меньше → экономнее RAM, но может шуметь в границах кластеров.

#### `clustering.birch_threshold` (только для birch)
- Меньше threshold → больше мелких кластеров.
- Больше threshold → меньше, но более широкие кластеры.

#### `clustering.birch_branching_factor` (только для birch)
- Влияет на структуру дерева и производительность.
- Обычно оставляют 30–100 и подбирают только если есть проблемы по скорости/качеству.

---

### 11.4 Готовые рабочие пресеты (копируйте в config)

#### Пример A: Большой датасет, быстрый и стабильный старт
```yaml
clustering:
  model: "minibatch_kmeans"
  n_clusters: 40
  batch_size: 8192
```

Когда использовать:
- много строк,
- нужно быстро,
- хотите 30–60 тем без долгого тюнинга.

#### Пример B: Малый/средний датасет, максимум качества по центроидам
```yaml
clustering:
  model: "kmeans"
  n_clusters: 25
```

Когда использовать:
- данных не очень много,
- важна более аккуратная кластеризация,
- время обучения не критично.

#### Пример C: Очень большие объёмы, дерево BIRCH
```yaml
clustering:
  model: "birch"
  n_clusters: 30
  birch_threshold: 0.4
  birch_branching_factor: 50
```

Когда использовать:
- много данных и KMeans тяжёл,
- нужно быстро получить первичную структуру тем.

#### Пример D: Слишком доминирует одна тема (например payment_not_seen)
```yaml
clustering:
  model: "minibatch_kmeans"
  n_clusters: 60
  batch_size: 8192

vectorizer:
  min_df: 8
  max_df: 0.4
```

Зачем:
- увеличить детализацию,
- ослабить влияние слишком частых общих фраз.

#### Пример E: Слишком шумные/случайные кластеры
```yaml
clustering:
  model: "kmeans"
  n_clusters: 20

vectorizer:
  min_df: 10
  max_df: 0.5
```

Зачем:
- сделать кластеры крупнее и стабильнее,
- убрать редкий шум.

---

### 11.5 Как применять изменения

1. Меняете параметры в `configs/config.yaml`.
2. Перезапускаете минимум Stage 2:
```bash
python -m app.train_full --config configs/config.yaml
```
3. Затем Stage 3:
```bash
python -m app.compare_december --config configs/config.yaml
```
4. Смотрите:
- `reports/cluster_summaries.csv`
- `reports/december_report.md`
- `reports/december_complaint_clusters.csv`

---

### 11.6 Почему при инференсе может быть "другая картинка"

- Stage 2 обучает baseline-кластеры (их число фиксируется в модели).
- Stage 3 не переобучает baseline, а только назначает декабрьские сообщения в уже существующие кластеры.
- Отдельно строятся complaint-кластеры декабря (`december_complaint_clusters.csv`), поэтому там может быть другой топ структуры.

Если хотите выше чувствительность к декабрю:
- увеличьте `novelty.percentile` (например 2 → 5),
- подберите `novelty.december_complaint_subclusters`.
