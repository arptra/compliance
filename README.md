# complaints-trends

Python-приложение для подготовки, weak-supervision, обучения локальных моделей и аналитики жалоб по Excel-файлам.

## Стек
- Python 3.11+
- Typer + Rich
- pandas/openpyxl/pyarrow
- scikit-learn/scipy/joblib
- pydantic v2/pyyaml
- jinja2/matplotlib
- gigachat SDK (mTLS)

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## mTLS для GigaChat
Используется только mTLS-подключение:
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

1. Положите клиентский сертификат и ключ в `certs/`.
2. Укажите CA bundle (например Russian Trusted Root CA) через `.env`/config.
3. `verify_ssl_certs` по умолчанию `true`.

## Команды
```bash
python -m complaints_trends.cli prepare --config configs/project.yaml --pilot --month 2025-09 --limit 5000
python -m complaints_trends.cli prepare --config configs/project.yaml
python -m complaints_trends.cli train --config configs/project.yaml
python -m complaints_trends.cli trends --config configs/project.yaml
python -m complaints_trends.cli infer-month --config configs/project.yaml --excel data/raw/2025-12.xlsx --month 2025-12
python -m complaints_trends.cli compare --config configs/project.yaml --new-month 2025-12 --baseline-range 2025-06..2025-11
python -m complaints_trends.cli demo
```

## Артефакты
- `data/processed/*.parquet`
- `models/*.joblib`
- `reports/*.html`
- `exports/*.xlsx`
