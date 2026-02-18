# Пайплайн анализа обращений (без LLM): темы, жалобы и «новый контекст»

Проект читает Excel с обращениями пользователей и делает:
1) кластеризацию тем (без ручных меток),
2) бинарную детекцию жалоб,
3) поиск **новых контекстов жалоб** в декабре относительно baseline.

Все построено на классическом ML (`scikit-learn`), без LLM/трансформеров/внешних API.

---

## 1. Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Быстрый запуск на синтетике

```bash
python -m app.make_synth_data --out data/sample.xlsx
python -m app.train --config configs/config.yaml
python -m app.predict --config configs/config.yaml
```

После запуска появятся:
- `outputs/labeled_baseline.xlsx`
- `outputs/labeled_december.xlsx`
- `outputs/combined_labeled.xlsx`
- `reports/report.md`
- `models/*`

---

## 3. Как передавать документы (Excel)

Есть **два режима входа**.

### Режим A: один общий файл

В `configs/config.yaml`:

```yaml
input:
  path: "data/my_all_data.xlsx"
```

Тогда baseline/december отфильтруются по `input.baseline` и `input.december` (см. раздел 5).

### Режим B: baseline — много файлов по месяцам + декабрь отдельным файлом

```yaml
input:
  path: null
  baseline_paths:
    - "data/2025-06.xlsx"
    - "data/2025-07.xlsx"
    - "data/2025-08.xlsx"
    - "data/2025-09.xlsx"
    - "data/2025-10.xlsx"
    - "data/2025-11.xlsx"
  december_path: "data/2025-12.xlsx"
  december_paths: []   # можно оставить пустым
```

Можно также передать несколько декабрьских файлов через `december_paths`.

---

## 4. Самое важное: если колонок сообщений несколько

Теперь поддерживается **несколько колонок текста в одной строке** (например, из разных каналов: чат, почта, колл-центр).

### Вариант 1: одна колонка

```yaml
input:
  message_col: "message"
  message_cols: null
```

### Вариант 2: несколько колонок

```yaml
input:
  message_col: null
  message_cols:
    - "message_chat"
    - "message_email"
    - "message_callcenter"
```

Пайплайн объединит их в служебную колонку `message_joined` и будет работать с ней.

> Имена колонок должны совпадать **точно** с заголовками в Excel.

---

## 5. Как разделять baseline и декабрь (когда файл один)

### По датам (частый вариант)

```yaml
input:
  date_col: "date"
  baseline:
    mode: "last_n_months"
    n_months: 6
  december:
    mode: "month_value"
    month_value: "2025-12"
```

### Если есть колонка month (`2025-12`)

```yaml
input:
  date_col: null
  month_col: "month"
  baseline:
    mode: "last_n_months"
    n_months: 6
  december:
    mode: "month_value"
    month_value: "2025-12"
```

### Через фильтры (ручной флаг)

```yaml
input:
  baseline:
    mode: "explicit_filter"
    filter_col: "period"
    filter_value: "baseline"
  december:
    mode: "explicit_filter"
    filter_col: "period"
    filter_value: "december"
```

---

## 6. Команды запуска

```bash
python -m app.train --config configs/config.yaml
python -m app.predict --config configs/config.yaml
```

Для быстрого чернового обучения:

```bash
python -m app.train --config configs/config.yaml --sample 50000
```

---

## 7. Как понимать итоговый отчет `reports/report.md`

Отчет состоит из блоков:

1. **Dataset sizes**  
   Сколько строк в baseline и декабре.

2. **Topic clustering overview**  
   Сколько кластеров тем и какие самые большие (по объему).

3. **Complaint detection overview**  
   Метрики на seed-разметке:
   - `precision` — доля правильно найденных жалоб среди предсказанных жалоб,
   - `recall` — доля найденных жалоб среди всех жалоб seed-валидации,
   - `f1` — баланс precision/recall.  
   И сравнение доли жалоб: baseline vs december.

4. **Novelty detection**  
   - порог новизны (`novelty threshold`) и percentile,
   - сколько декабрьских жалоб признано novel,
   - emerging terms (термины, усилившиеся в декабре),
   - группы novel-жалоб с примерами.

5. **Artifact paths**  
   Пути к Excel/CSV/JSON/моделям.

### Как интерпретировать быстро
- Если `december complaint rate` сильно выше baseline — рост жалоб.
- Если много `novel complaints` — вероятно, появились новые типы проблем.
- Смотрите `emerging terms` + `novel complaint clusters` для приоритизации расследования.

---

## 8. Параметры и на что они влияют

### `preprocess.*`
- `lemmatize`: `true` медленнее, но иногда лучше обобщение форм слов.
- `use_razdel`: токенизация для RU; при недоступности есть fallback.
- `use_char_ngrams`: устойчивее к опечаткам, но может увеличить размер признаков.
- `max_token_len`: отсекает длинные шумовые токены.

### `vectorizer.*`
- `ngram_range`: `[1,2]` добавляет биграммы (больше контекста, больше размерность).
- `min_df`: увеличивайте для отсева редкого шума.
- `max_df`: снижайте, чтобы убрать слишком частые «общие» термины.
- `max_features`: главный рычаг по памяти/скорости.

### `clustering.*`
- `n_clusters`: больше кластеров = более детальные темы, но сложнее интерпретация.
- `batch_size`: влияет на скорость обучения MiniBatchKMeans.
- `top_terms`, `examples_per_cluster`: насколько подробно показывать темы в отчете.

### `complaint.*`
- `model`: `logreg` (вероятности) или `linearsvc` (margin).
- `threshold`: порог для `is_complaint`:
  - выше => меньше ложноположительных, но риск пропустить жалобы,
  - ниже => больше полнота, но больше шума.

### `novelty.*`
- `percentile`: ключевой порог новизны:
  - меньше (напр. 1) => строже, novel будет меньше,
  - больше (напр. 5) => мягче, novel будет больше.
- `novel_december_subclusters`: детализация групп «новых жалоб».
- `examples_per_novel_cluster`: сколько примеров показывать.

### `input.*`
- `path` — единый файл.
- `baseline_paths` + `december_path/december_paths` — много baseline-файлов и отдельный декабрь.
- `message_col` / `message_cols` — одна или несколько колонок текста.

---

## 9. Частые проблемы

### «Не найдена колонка»
Проверьте точное имя в Excel и в `message_col/message_cols`, `date_col`, `month_col`.

### «Дата не парсится»
Укажите `date_format` или переходите на `month_col` / `explicit_filter`.

### Медленно
- `lemmatize: false`
- уменьшить `max_features`
- уменьшить `n_clusters`
- использовать `--sample` для итераций

---

## 10. Мини-чеклист перед боевым прогоном

1. Проверили путь к файлам (`path` или `baseline_paths + december_path`).
2. Проверили колонку(и) сообщений (`message_col`/`message_cols`).
3. Проверили логику split baseline/december.
4. Запустили train + predict.
5. Проверили `reports/report.md` и `outputs/*.xlsx`.
