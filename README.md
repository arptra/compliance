# Пайплайн анализа обращений (без LLM): темы, жалобы и «новый контекст»

Проект читает Excel с обращениями пользователей и делает:
1) кластеризацию тем (без ручных меток),
2) бинарную детекцию жалоб,
3) поиск новых контекстов жалоб в целевом периоде (исторически поле называется `december`).

Все на классическом ML (`scikit-learn`), без LLM/трансформеров/внешних API.

## Почему теперь отчет стал чище
Раньше в `top terms` попадали служебные артефакты анонимизации (`<phone>`, `xxxx`, `***`, ID/UUID).
Теперь перед векторизацией включена агрессивная очистка:
- удаляются placeholder-теги и маски (`<...>`, `***`, `xxxx`, `####`),
- удаляются URL/email/телефоны,
- удаляются/токенизируются числа (по конфигу),
- denylist и фильтры токенов блокируют мусор в словаре,
- top terms в кластерах фильтруются повторно и строятся через c-TF-IDF.

Итог: `cluster_summaries.csv` и `report.md` должны быть интерпретируемыми.

---

## 1) Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Быстрый запуск
```bash
python -m app.make_synth_data --out data/sample.xlsx
python -m app.train --config configs/config.yaml
python -m app.predict --config configs/config.yaml
```

---

## 3) Как передавать документы

### Режим A: один общий файл
```yaml
input:
  path: "data/my_all_data.xlsx"
```

### Режим B: baseline по месяцам + отдельный target-файл
```yaml
input:
  path: null
  baseline_paths:
    - "data/2025-06.xlsx"
    - "data/2025-07.xlsx"
    - "data/2025-08.xlsx"
  december_path: "data/2025-09.xlsx"
```

Также можно несколько target-файлов:
```yaml
input:
  december_paths:
    - "data/2025-09.xlsx"
    - "data/2025-10.xlsx"
```

---

## 4) Одна или несколько колонок сообщений

Одна колонка:
```yaml
input:
  message_col: "message"
  message_cols: null
```

Несколько колонок (из разных каналов):
```yaml
input:
  message_col: null
  message_cols:
    - "message_chat"
    - "message_email"
    - "message_callcenter"
```

Пайплайн объединит их в `message_joined`.

---

## 5) Как выбрать target-месяц(ы)

### Один месяц
```yaml
input:
  baseline:
    mode: "last_n_months"
    n_months: 6
  december:
    mode: "month_value"
    month_value: "2026-01"
```

### Диапазон
```yaml
input:
  december:
    mode: "date_range"
    start: "2026-01-01"
    end: "2026-02-28"
```

### Фиксированный baseline + target диапазон
```yaml
input:
  baseline:
    mode: "date_range"
    start: "2025-06-01"
    end: "2025-11-30"
  december:
    mode: "date_range"
    start: "2026-03-01"
    end: "2026-03-31"
```

---

## 6) Как читать отчет `reports/report.md`

Разделы:
1. **How to read this report** — базовые определения (category/complaint/novel).
2. **Dataset sizes** — объем baseline и target.
3. **Topic clustering overview** — крупные кластеры, top terms и примеры сообщений.
4. **Complaint detection overview** — precision/recall/f1 на seed-валидации.
5. **Novelty detection** — порог, доля novel-жалоб, emerging terms, группы новых жалоб.

Главное правило интерпретации:
- сначала смотрите **examples**,
- затем `top_terms` как краткое описание темы,
- `novel complaints` используйте как сигнал новых сценариев проблем.

---

## 7) Куда добавлять свои стоп-слова и deny-токены

- Файл пользовательских стоп-слов: `configs/extra_stopwords.txt`
- Файл запрещенных токенов/артефактов: `configs/deny_tokens.txt`

Добавляйте туда служебные слова и маски, которые не должны попадать в словарь и в `top terms`.

Примеры для denylist:
- `<num>`, `<phone>`, `<email>`, `<name>`
- `xxxx`, `хххх`, `***`, `####`
- технические маркеры/шаблоны вашей системы

---

## 8) Основные команды
```bash
python -m app.train --config configs/config.yaml
python -m app.predict --config configs/config.yaml
pytest -q
```

Для быстрых итераций:
```bash
python -m app.train --config configs/config.yaml --sample 50000
```
