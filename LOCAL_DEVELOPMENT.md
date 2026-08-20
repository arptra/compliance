# Локальная разработка

Backend и frontend запускаются отдельно, в двух терминалах. Перед выполнением команд откройте оба терминала в корне репозитория.

## Подготовка backend

При первом запуске создайте виртуальное окружение и установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

При необходимости создайте локальный файл окружения на основе `.env.example` и заполните параметры подключения к GigaChat.

## Запуск backend

```bash
source .venv/bin/activate
export PYTHONPATH=src
python -m complaints_trends.cli api-serve \
  --config configs/project.yaml \
  --host 0.0.0.0 \
  --port 8000
```

Backend будет доступен по адресу `http://127.0.0.1:8000`.

## Подготовка frontend

При первом запуске или после изменения зависимостей выполните:

```bash
npm install --prefix apps/dashboard
```

## Запуск frontend

```bash
npm run dev --prefix apps/dashboard
```

Frontend будет доступен по адресу `http://127.0.0.1:5173/gigachat`.

В режиме разработки Vite автоматически перенаправляет запросы `/api` на backend `http://127.0.0.1:8000`. Если backend запущен по другому адресу, укажите его перед запуском frontend:

```bash
export VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev --prefix apps/dashboard
```
