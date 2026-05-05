#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.run"
LOG_DIR="${RUN_DIR}/logs"
DASHBOARD_DIR="${ROOT_DIR}/apps/dashboard"

API_PID_FILE="${RUN_DIR}/api.pid"
DASHBOARD_PID_FILE="${RUN_DIR}/dashboard.pid"
RUNTIME_CONFIG="${RUN_DIR}/project.runtime.yaml"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
DASHBOARD_PORT="${DASHBOARD_PORT:-5173}"
API_DISPLAY_URL="${API_DISPLAY_URL:-http://127.0.0.1:${API_PORT}}"
DASHBOARD_DISPLAY_URL="${DASHBOARD_DISPLAY_URL:-http://127.0.0.1:${DASHBOARD_PORT}}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${ROOT_DIR}/configs/project.yaml}"
PREPARE_OUTPUT_PARQUET="${PREPARE_OUTPUT_PARQUET:-${ROOT_DIR}/data/processed/all_prepared.parquet}"
INTERIM_DIR="${INTERIM_DIR:-${ROOT_DIR}/data/interim}"
REPORTS_DIR="${REPORTS_DIR:-${ROOT_DIR}/reports}"
EXPORTS_DIR="${EXPORTS_DIR:-${ROOT_DIR}/exports}"
MODELS_DIR="${MODELS_DIR:-${ROOT_DIR}/models}"
VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://127.0.0.1:${API_PORT}}"
VITE_API_PORT="${VITE_API_PORT:-${API_PORT}}"
VITE_PUBLIC_ORIGIN="${VITE_PUBLIC_ORIGIN:-${DASHBOARD_DISPLAY_URL}}"
VITE_HMR_HOST="${VITE_HMR_HOST:-}"
VITE_HMR_CLIENT_PORT="${VITE_HMR_CLIENT_PORT:-${DASHBOARD_PORT}}"
VITE_HMR_PROTOCOL="${VITE_HMR_PROTOCOL:-ws}"

DEFAULT_PYTHON311_ENV_DIR="${ROOT_DIR}/.venv311"
LEGACY_PYTHON_ENV_DIR="${ROOT_DIR}/.venv"
PYTHON_ENV_DIR=""
PYTHON_ACTIVATE=""
PYTHON_BIN=""
DASHBOARD_VITE_BIN="${DASHBOARD_DIR}/node_modules/.bin/vite"
DASHBOARD_PACKAGE_LOCK="${DASHBOARD_DIR}/package-lock.json"
AUTO_INSTALL_DASHBOARD_DEPS="${AUTO_INSTALL_DASHBOARD_DEPS:-1}"
AUTO_BOOTSTRAP_PYTHON_ENV="${AUTO_BOOTSTRAP_PYTHON_ENV:-1}"
SETUP_CA_ENV_SCRIPT="${ROOT_DIR}/scripts/setup_gigachat_ca_env.sh"

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

wait_until_alive() {
  local pid="$1"
  local name="$2"
  local log_file="$3"

  sleep 2
  if kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  echo "${name} did not start successfully. Check ${log_file}" >&2
  if [[ -f "${log_file}" ]]; then
    tail -n 40 "${log_file}" >&2 || true
  fi
  exit 1
}

is_python_ge_311() {
  local python_bin="$1"
  "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

bootstrap_python_env() {
  local bootstrap_log="${LOG_DIR}/python-bootstrap.log"

  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Python 3.11+ is required. Install python3.11 or create ${DEFAULT_PYTHON311_ENV_DIR} manually." >&2
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
    echo "No Python 3.11+ virtual environment found. README expects Python 3.11+." >&2
    echo "Create ${DEFAULT_PYTHON311_ENV_DIR} or rerun with AUTO_BOOTSTRAP_PYTHON_ENV=1." >&2
    exit 1
  fi

  bootstrap_python_env
}

ensure_dashboard_dependencies() {
  if [[ -x "${DASHBOARD_VITE_BIN}" ]]; then
    return 0
  fi

  if [[ "${AUTO_INSTALL_DASHBOARD_DEPS}" != "1" ]]; then
    echo "Dashboard dependencies are missing. Run: cd ${DASHBOARD_DIR} && npm ci" >&2
    exit 1
  fi

  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is not installed, so dashboard dependencies cannot be installed automatically." >&2
    echo "Install npm and run: cd ${DASHBOARD_DIR} && npm ci" >&2
    exit 1
  fi

  local install_log="${LOG_DIR}/dashboard-npm-install.log"
  echo "Dashboard dependencies are missing. Installing with npm ci..."

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
    echo "Dashboard dependencies installation completed, but vite was not found. Check ${install_log}" >&2
    exit 1
  fi
}

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${INTERIM_DIR}" "${REPORTS_DIR}" "${EXPORTS_DIR}" "${MODELS_DIR}"

ensure_python_environment

if ! ensure_pid_file_is_stale "${API_PID_FILE}"; then
  echo "API is already running with pid $(cat "${API_PID_FILE}")" >&2
  exit 1
fi

if ! ensure_pid_file_is_stale "${DASHBOARD_PID_FILE}"; then
  echo "Dashboard is already running with pid $(cat "${DASHBOARD_PID_FILE}")" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PYTHON_ACTIVATE}"
export PYTHONPATH="${ROOT_DIR}/src"

ensure_dashboard_dependencies

# shellcheck source=/dev/null
source "${SETUP_CA_ENV_SCRIPT}"
setup_gigachat_ca_env "${ROOT_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_runtime_config.py" \
  --input "${CONFIG_TEMPLATE}" \
  --output "${RUNTIME_CONFIG}" \
  --prepare-output "${PREPARE_OUTPUT_PARQUET}" \
  --interim-dir "${INTERIM_DIR}" \
  --reports-dir "${REPORTS_DIR}" \
  --exports-dir "${EXPORTS_DIR}" \
  --models-dir "${MODELS_DIR}"

API_LOG="${LOG_DIR}/api.log"
DASHBOARD_LOG="${LOG_DIR}/dashboard.log"

(
  cd "${ROOT_DIR}"
  RUNTIME_CONFIG="${RUNTIME_CONFIG}" API_PORT="${API_PORT}" API_HOST="${API_HOST}" \
    nohup "${PYTHON_BIN}" "${ROOT_DIR}/scripts/start_api.py" >"${API_LOG}" 2>&1 &
  echo $! > "${API_PID_FILE}"
)

(
  cd "${DASHBOARD_DIR}"
  VITE_API_BASE_URL="${VITE_API_BASE_URL}" \
    VITE_API_PORT="${VITE_API_PORT}" \
    VITE_PUBLIC_ORIGIN="${VITE_PUBLIC_ORIGIN}" \
    VITE_HMR_HOST="${VITE_HMR_HOST}" \
    VITE_HMR_CLIENT_PORT="${VITE_HMR_CLIENT_PORT}" \
    VITE_HMR_PROTOCOL="${VITE_HMR_PROTOCOL}" \
    nohup "${DASHBOARD_VITE_BIN}" --host "${DASHBOARD_HOST}" --port "${DASHBOARD_PORT}" --strictPort >"${DASHBOARD_LOG}" 2>&1 &
  echo $! > "${DASHBOARD_PID_FILE}"
)

API_PID="$(cat "${API_PID_FILE}")"
DASHBOARD_PID="$(cat "${DASHBOARD_PID_FILE}")"

wait_until_alive "${API_PID}" "API" "${API_LOG}"
wait_until_alive "${DASHBOARD_PID}" "Dashboard" "${DASHBOARD_LOG}"

cat <<EOF
API started:       ${API_DISPLAY_URL}
Dashboard started: ${DASHBOARD_DISPLAY_URL}

API bind host:     ${API_HOST}:${API_PORT}
Dashboard bind:    ${DASHBOARD_HOST}:${DASHBOARD_PORT}
Dashboard -> API:  ${VITE_API_BASE_URL}
Vite origin:       ${VITE_PUBLIC_ORIGIN}
Vite HMR:          ${VITE_HMR_PROTOCOL}://${VITE_HMR_HOST:-${DASHBOARD_HOST}}:${VITE_HMR_CLIENT_PORT}

API pid:       ${API_PID}
Dashboard pid: ${DASHBOARD_PID}

API log:       ${API_LOG}
Dashboard log: ${DASHBOARD_LOG}
EOF
