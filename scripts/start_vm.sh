#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/.run}"
LOG_DIR="${RUN_DIR}/logs"
DASHBOARD_DIR="${ROOT_DIR}/apps/dashboard"

API_PID_FILE="${RUN_DIR}/api.pid"
DASHBOARD_PID_FILE="${RUN_DIR}/dashboard.pid"
RUNTIME_CONFIG="${RUN_DIR}/project.runtime.yaml"

# НАСТРОЙКИ ДЛЯ ВИРТУАЛКИ - слушаем все интерфейсы
API_HOST="${API_HOST:-0.0.0.0}"              # ← изменено с 127.0.0.1
API_PORT="${API_PORT:-8000}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"  # ← изменено с 127.0.0.1
DASHBOARD_PORT="${DASHBOARD_PORT:-5173}"

# Пути к данным
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${ROOT_DIR}/configs/project.yaml}"
PREPARE_OUTPUT_PARQUET="${PREPARE_OUTPUT_PARQUET:-${ROOT_DIR}/data/processed/all_prepared.parquet}"
INTERIM_DIR="${INTERIM_DIR:-${ROOT_DIR}/data/interim}"
REPORTS_DIR="${REPORTS_DIR:-${ROOT_DIR}/reports}"
EXPORTS_DIR="${EXPORTS_DIR:-${ROOT_DIR}/exports}"
MODELS_DIR="${MODELS_DIR:-${ROOT_DIR}/models}"

# API URL для Dashboard (используем внешний IP или hostname)
# ВАЖНО: замени на реальный IP или hostname виртуалки
VM_IP="${VM_IP:-$(hostname -I | awk '{print $1}')}"
VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://${VM_IP}:${API_PORT}}"

# Виртуальное окружение Python (автоопределение)
DEFAULT_PYTHON311_ENV_DIR="${ROOT_DIR}/.venv311"
LEGACY_PYTHON_ENV_DIR="${ROOT_DIR}/.venv"
PYTHON_ENV_DIR=""
PYTHON_ACTIVATE=""
PYTHON_BIN=""

# Dashboard
DASHBOARD_VITE_BIN="${DASHBOARD_DIR}/node_modules/.bin/vite"
DASHBOARD_PACKAGE_LOCK="${DASHBOARD_DIR}/package-lock.json"
AUTO_INSTALL_DASHBOARD_DEPS="${AUTO_INSTALL_DASHBOARD_DEPS:-1}"
AUTO_BOOTSTRAP_PYTHON_ENV="${AUTO_BOOTSTRAP_PYTHON_ENV:-1}"

# Опционально: CA сертификаты
SETUP_CA_ENV_SCRIPT="${ROOT_DIR}/scripts/setup_gigachat_ca_env.sh"

# Функция проверки, не запущен ли уже процесс
ensure_pid_file_is_stale() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    return 1
  fi

  rm -f "${pid_file}"
  return 0
}

# Функция проверки, что сервер запустился
wait_until_alive() {
  local pid="$1"
  local name="$2"
  local log_file="$3"
  local max_attempts="${4:-30}"
  local attempt=0

  sleep 3

  while [[ $attempt -lt $max_attempts ]]; do
    if kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 1
  done

  echo "${name} did not start successfully. Check ${log_file}" >&2
  if [[ -f "${log_file}" ]]; then
    tail -n 40 "${log_file}" >&2 || true
  fi
  exit 1
}

# Проверка версии Python
is_python_ge_311() {
  local python_bin="$1"
  "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

# Создание Python окружения
bootstrap_python_env() {
  local bootstrap_log="${LOG_DIR}/python-bootstrap.log"

  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Python 3.11+ is required. Install python3.11 first." >&2
    exit 1
  fi

  echo "Bootstrapping Python 3.11 virtual environment in ${DEFAULT_PYTHON311_ENV_DIR}..."
  mkdir -p "${RUN_DIR}/pip-cache"

  if [[ ! -x "${DEFAULT_PYTHON311_ENV_DIR}/bin/python" ]]; then
    python3.11 -m venv "${DEFAULT_PYTHON311_ENV_DIR}" >"${bootstrap_log}" 2>&1
  fi

  PYTHON_ENV_DIR="${DEFAULT_PYTHON311_ENV_DIR}"
  PYTHON_ACTIVATE="${PYTHON_ENV_DIR}/bin/activate"
  PYTHON_BIN="${PYTHON_ENV_DIR}/bin/python"

  (
    cd "${ROOT_DIR}"
    PIP_CACHE_DIR="${RUN_DIR}/pip-cache" "${PYTHON_BIN}" -m pip install --upgrade pip >>"${bootstrap_log}" 2>&1
    PIP_CACHE_DIR="${RUN_DIR}/pip-cache" "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt" >>"${bootstrap_log}" 2>&1
  ) || {
    echo "Python environment bootstrap failed. Check ${bootstrap_log}" >&2
    tail -n 40 "${bootstrap_log}" >&2 || true
    exit 1
  }
}

# Подготовка Python окружения
ensure_python_environment() {
  local candidate_bin=""

  for candidate_bin in "${DEFAULT_PYTHON311_ENV_DIR}/bin/python" "${LEGACY_PYTHON_ENV_DIR}/bin/python"; do
    if [[ -x "${candidate_bin}" ]] && is_python_ge_311 "${candidate_bin}"; then
      PYTHON_ENV_DIR="$(cd "$(dirname "${candidate_bin}")/.." && pwd)"
      PYTHON_ACTIVATE="${PYTHON_ENV_DIR}/bin/activate"
      PYTHON_BIN="${PYTHON_ENV_DIR}/bin/python"
      return 0
    fi
  done

  if [[ "${AUTO_BOOTSTRAP_PYTHON_ENV}" != "1" ]]; then
    echo "No Python 3.11+ virtual environment found." >&2
    exit 1
  fi

  bootstrap_python_env
}

# Установка зависимостей Dashboard
ensure_dashboard_dependencies() {
  if [[ -x "${DASHBOARD_VITE_BIN}" ]]; then
    return 0
  fi

  if [[ "${AUTO_INSTALL_DASHBOARD_DEPS}" != "1" ]]; then
    echo "Dashboard dependencies are missing. Run: cd ${DASHBOARD_DIR} && npm ci" >&2
    exit 1
  fi

  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is not installed, install Node.js/npm first." >&2
    exit 1
  fi

  local install_log="${LOG_DIR}/dashboard-npm-install.log"
  echo "Installing dashboard dependencies..."

  if [[ -f "${DASHBOARD_PACKAGE_LOCK}" ]]; then
    (
      cd "${DASHBOARD_DIR}"
      NPM_CONFIG_CACHE="${RUN_DIR}/npm-cache" npm ci --no-fund --no-audit >"${install_log}" 2>&1
    )
  else
    (
      cd "${DASHBOARD_DIR}"
      NPM_CONFIG_CACHE="${RUN_DIR}/npm-cache" npm install --no-fund --no-audit >"${install_log}" 2>&1
    )
  fi

  if [[ ! -x "${DASHBOARD_VITE_BIN}" ]]; then
    echo "Failed to install dashboard dependencies. Check ${install_log}" >&2
    exit 1
  fi
}

# ============================================
# MAIN
# ============================================

# Создание необходимых директорий
mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${INTERIM_DIR}" "${REPORTS_DIR}" "${EXPORTS_DIR}" "${MODELS_DIR}"

# Проверка Python окружения
ensure_python_environment

# Проверка, не запущены ли процессы
if ! ensure_pid_file_is_stale "${API_PID_FILE}"; then
  echo "API is already running with pid $(cat "${API_PID_FILE}")" >&2
  exit 1
fi

if ! ensure_pid_file_is_stale "${DASHBOARD_PID_FILE}"; then
  echo "Dashboard is already running with pid $(cat "${DASHBOARD_PID_FILE}")" >&2
  exit 1
fi

# Активация Python окружения
# shellcheck source=/dev/null
source "${PYTHON_ACTIVATE}"
export PYTHONPATH="${ROOT_DIR}/src"

# Установка зависимостей Dashboard
ensure_dashboard_dependencies

# Настройка CA сертификатов (если есть)
if [[ -f "${SETUP_CA_ENV_SCRIPT}" ]]; then
  # shellcheck source=/dev/null
  source "${SETUP_CA_ENV_SCRIPT}" || true
  if declare -f setup_gigachat_ca_env >/dev/null; then
    setup_gigachat_ca_env "${ROOT_DIR}" || echo "Warning: CA setup failed" >&2
  fi
fi

# Генерация runtime конфига
if [[ -f "${ROOT_DIR}/scripts/generate_runtime_config.py" ]]; then
  echo "Generating runtime configuration..."
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_runtime_config.py" \
    --input "${CONFIG_TEMPLATE}" \
    --output "${RUNTIME_CONFIG}" \
    --prepare-output "${PREPARE_OUTPUT_PARQUET}" \
    --interim-dir "${INTERIM_DIR}" \
    --reports-dir "${REPORTS_DIR}" \
    --exports-dir "${EXPORTS_DIR}" \
    --models-dir "${MODELS_DIR}" || {
      echo "WARNING: Failed to generate runtime config" >&2
    }
fi

# Запуск API
API_LOG="${LOG_DIR}/api.log"
echo "Starting API on ${API_HOST}:${API_PORT}..."

(
  cd "${ROOT_DIR}"
  nohup "${PYTHON_BIN}" -m uvicorn src.api.main:app \
    --host "${API_HOST}" \
    --port "${API_PORT}" \
    --workers 1 \
    >"${API_LOG}" 2>&1 &
  echo $! > "${API_PID_FILE}"
)

# Запуск Dashboard
DASHBOARD_LOG="${LOG_DIR}/dashboard.log"
echo "Starting Dashboard on ${DASHBOARD_HOST}:${DASHBOARD_PORT}..."

(
  cd "${DASHBOARD_DIR}"
  VITE_API_BASE_URL="${VITE_API_BASE_URL}" \
    nohup "${DASHBOARD_VITE_BIN}" \
      --host "${DASHBOARD_HOST}" \
      --port "${DASHBOARD_PORT}" \
      --strictPort \
      >"${DASHBOARD_LOG}" 2>&1 &
  echo $! > "${DASHBOARD_PID_FILE}"
)

# Ожидание запуска
API_PID="$(cat "${API_PID_FILE}")"
DASHBOARD_PID="$(cat "${DASHBOARD_PID_FILE}")"

wait_until_alive "${API_PID}" "API" "${API_LOG}"
wait_until_alive "${DASHBOARD_PID}" "Dashboard" "${DASHBOARD_LOG}"

# Получение внешнего IP
EXTERNAL_IP="$(hostname -I | awk '{print $1}')"

# Вывод информации
cat <<EOF

========================================
✅ ALL SERVICES STARTED SUCCESSFULLY!
========================================

📍 API:
   - Local:      http://127.0.0.1:${API_PORT}
   - External:   http://${EXTERNAL_IP}:${API_PORT}
   - PID:        ${API_PID}
   - Log:        ${API_LOG}

📍 Dashboard:
   - Local:      http://127.0.0.1:${DASHBOARD_PORT}
   - External:   http://${EXTERNAL_IP}:${DASHBOARD_PORT}
   - PID:        ${DASHBOARD_PID}
   - Log:        ${DASHBOARD_LOG}

📁 Config:        ${RUNTIME_CONFIG}

🛑 Stop all services:
   kill ${API_PID} ${DASHBOARD_PID}

📊 Test API:
   curl http://${EXTERNAL_IP}:${API_PORT}/api/meta/datasets

🌐 Open Dashboard in browser:
   http://${EXTERNAL_IP}:${DASHBOARD_PORT}

========================================
EOF