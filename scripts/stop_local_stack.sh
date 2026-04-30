#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/.run"

API_PID_FILE="${RUN_DIR}/api.pid"
DASHBOARD_PID_FILE="${RUN_DIR}/dashboard.pid"
RUNTIME_CONFIG="${RUN_DIR}/project.runtime.yaml"

stop_from_pid_file() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: pid file not found, skip"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    echo "${name}: empty pid file removed"
    return 0
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${pid_file}"
    echo "${name}: process ${pid} is not running, stale pid file removed"
    return 0
  fi

  kill "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${pid_file}"
      echo "${name}: stopped"
      return 0
    fi
    sleep 0.5
  done

  kill -9 "${pid}" 2>/dev/null || true
  rm -f "${pid_file}"
  echo "${name}: force stopped"
}

stop_from_pid_file "Dashboard" "${DASHBOARD_PID_FILE}"
stop_from_pid_file "API" "${API_PID_FILE}"

rm -f "${RUNTIME_CONFIG}"
