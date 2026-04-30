#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SCRIPT="${ROOT_DIR}/scripts/start_local_stack.sh"
STOP_SCRIPT="${ROOT_DIR}/scripts/stop_local_stack.sh"

API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-5173}"
RESTART_WAIT_SECONDS="${RESTART_WAIT_SECONDS:-2}"
RESTART_FREE_PORTS="${RESTART_FREE_PORTS:-1}"
RESTART_PORT_WAIT_LOOPS="${RESTART_PORT_WAIT_LOOPS:-20}"

ensure_script_exists() {
  local script_path="$1"
  if [[ ! -x "${script_path}" ]]; then
    echo "Required script is missing or not executable: ${script_path}" >&2
    exit 1
  fi
}

soft_stop_port_listener() {
  local name="$1"
  local port="$2"

  if [[ "${RESTART_FREE_PORTS}" != "1" ]]; then
    return 0
  fi

  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  local pids
  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ')"
  if [[ -z "${pids// /}" ]]; then
    return 0
  fi

  echo "${name}: port ${port} is still busy, sending SIGTERM to listeners: ${pids}"
  for pid in ${pids}; do
    kill "${pid}" 2>/dev/null || true
  done

  for _ in $(seq 1 "${RESTART_PORT_WAIT_LOOPS}"); do
    if ! lsof -tiTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "${name}: port ${port} released"
      return 0
    fi
    sleep 0.5
  done

  echo "${name}: port ${port} is still busy after soft stop; start may fail until it is released" >&2
}

ensure_script_exists "${STOP_SCRIPT}"
ensure_script_exists "${START_SCRIPT}"

echo "Stopping local stack..."
"${STOP_SCRIPT}"

soft_stop_port_listener "API" "${API_PORT}"
soft_stop_port_listener "Dashboard" "${DASHBOARD_PORT}"

sleep "${RESTART_WAIT_SECONDS}"

echo "Starting local stack..."
exec "${START_SCRIPT}"
