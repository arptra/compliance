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
