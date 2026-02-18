# Пайплайн анализа обращений (без LLM): темы, жалобы и «новый контекст»

Этот проект обрабатывает Excel-файл с сообщениями клиентов (в т.ч. на русском) и делает 3 вещи:
1. **Группирует сообщения по темам** (кластеризация, без ручной разметки).
2. **Определяет, является ли сообщение жалобой** (слабая разметка + классический ML).
3. Для декабря ищет **«жалобы в новом контексте»** — то, что похоже на новые проблемы относительно базового периода (обычно прошлые 6 месяцев).

Важно:
- ❌ Никаких LLM, трансформеров и внешних API.
- ✅ Только классический стек (`scikit-learn`, `pandas`, `numpy`, `scipy`).
- ✅ Можно запускать офлайн.

---

## 1. Что нужно на вход

Нужен Excel-файл (`.xlsx`) минимум с одной колонкой текста сообщений.

Обычно:
- колонка с текстом: `message`
- колонка с датой: `date` (например `2025-12-15`)

Если даты нет — можно использовать:
- колонку месяца (`month`, например `2025-12`),
- или любую служебную колонку для фильтра (`is_baseline`, `is_december` и т.п.).

---

## 2. Самый важный вопрос: «Жалобы в одной колонке — как её передать?»

Передача делается **через конфиг** `configs/config.yaml`.

Ключ:
```yaml
input:
  message_col: "message"
```

Где `"message"` — это **точное имя колонки в вашем Excel**, в которой лежит текст обращений/жалоб.

### Пример 1
В вашем файле колонка называется `Текст обращения`.
Тогда в конфиге ставите:
```yaml
input:
  message_col: "Текст обращения"
```

### Пример 2
Колонка называется `Жалоба клиента`.
Тогда:
```yaml
input:
  message_col: "Жалоба клиента"
```

> Главное: имя должно совпадать символ в символ (с пробелами, регистром, кириллицей).

---

## 3. Установка (пошагово, для начинающих)

Откройте терминал в папке проекта и выполните:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Быстрая проверка на тестовых данных

Чтобы убедиться, что всё работает:

```bash
python -m app.make_synth_data --out data/sample.xlsx
python -m app.train --config configs/config.yaml
python -m app.predict --config configs/config.yaml
```

После этого появятся файлы:
- `outputs/labeled_baseline.xlsx`
- `outputs/labeled_december.xlsx`
- `outputs/combined_labeled.xlsx`
- `reports/report.md`
- модели в `models/`

---

## 5. Как запустить на вашем Excel (реальный сценарий)

### Шаг 1: положите файл
Например:
`data/my_messages.xlsx`

### Шаг 2: настройте `configs/config.yaml`

Минимально проверьте эти поля:

```yaml
input:
  path: "data/my_messages.xlsx"      # путь к вашему Excel
  message_col: "message"             # имя колонки с текстом (замените на своё)
  date_col: "date"                   # имя колонки с датой (если есть)
  month_col: null                      # если даты нет, можно использовать month
```

### Шаг 3: выберите, как делить baseline и декабрь

#### Вариант A (самый частый): по дате
```yaml
input:
  baseline:
    mode: "last_n_months"
    n_months: 6
  december:
    mode: "month_value"
    month_value: "2025-12"
```

#### Вариант B: если есть колонка `month` (`2025-06`, `2025-07`, ...)
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

#### Вариант C: вручную через флаги/фильтр
Например, у вас есть колонка `period` со значениями `baseline` и `december`:
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

### Шаг 4: запуск

```bash
python -m app.train --config configs/config.yaml
python -m app.predict --config configs/config.yaml
```

---

## 6. Что происходит внутри пайплайна (простыми словами)

### Этап Train (`app.train`)
1. Читает Excel и делит строки на:
   - baseline (обычно прошлые 6 мес.),
   - december.
2. Чистит текст:
   - в нижний регистр,
   - удаляет URL/email/телефоны,
   - заменяет цифры на `<num>`.
3. Строит TF-IDF признаки.
4. Учит кластеризацию (MiniBatchKMeans) на baseline.
5. По словарю правил ставит seed-метки «жалоба/не жалоба/неизвестно».
6. Учит бинарный классификатор жалоб на seed-подмножестве.
7. Считает порог новизны по baseline (по процентилю сходства с центроидами).
8. Сохраняет модели и baseline-выгрузки.

### Этап Predict (`app.predict`)
1. Загружает модели.
2. Преобразует baseline+december тем же препроцессингом и TF-IDF.
3. Для каждой строки считает:
   - `cluster_id` (тема),
   - `complaint_score`, `is_complaint`,
   - `max_sim_to_baseline_centroid`, `is_novel`.
4. Выделяет **new-context complaints**:
   - `is_complaint == 1` и `is_novel == 1`.
5. Строит отчёт `reports/report.md` и Excel-выгрузки.

---

## 7. Как читать итоговые файлы

### `outputs/labeled_baseline.xlsx`
Строки baseline с тех. полями:
- `cluster_id`
- `complaint_score`
- `is_complaint`
- `max_sim_to_baseline_centroid`
- `is_novel`

### `outputs/labeled_december.xlsx`
То же для декабря.

### `outputs/combined_labeled.xlsx`
Обе выборки в одном файле.

### `reports/report.md`
Краткая аналитика:
- размеры выборок,
- топ кластеров,
- метрики детектора жалоб,
- доля жалоб,
- порог новизны,
- топ новых терминов,
- группы «новых» жалоб с примерами.

---

## 8. Частые ошибки и как исправить

### Ошибка: не найдена колонка
Проверьте `input.message_col`, `input.date_col`, `input.month_col` в `config.yaml`.
Имена должны точно совпадать с Excel.

### Дата не парсится
- Укажите `date_format` в конфиге,
- или переходите на `month_col`,
- или используйте `explicit_filter`.

### Медленно работает
- Оставьте `lemmatize: false`.
- Уменьшите `vectorizer.max_features`.
- Уменьшите `clustering.n_clusters`.
- Для чернового прогона используйте:
  ```bash
  python -m app.train --config configs/config.yaml --sample 50000
  ```

---

## 9. Основные команды

```bash
# 1) (опционально) сгенерировать тестовые данные
python -m app.make_synth_data --out data/sample.xlsx

# 2) обучение
python -m app.train --config configs/config.yaml

# 3) инференс + отчёт
python -m app.predict --config configs/config.yaml

# 4) тесты
pytest -q
```

---

## 10. Кратко: что менять в первую очередь под ваш Excel

1. `input.path` — путь к вашему файлу.
2. `input.message_col` — **имя колонки с жалобами/сообщениями**.
3. Настройка split (по дате/месяцу/фильтру).
4. Запуск train + predict.

Если хотите, можно завести отдельный `configs/config_prod.yaml` для боевого файла и не трогать примерный конфиг.
