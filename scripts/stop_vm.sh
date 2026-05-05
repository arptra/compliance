#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOP_SCRIPT="${ROOT_DIR}/scripts/stop_local_stack.sh"

if [[ ! -x "${STOP_SCRIPT}" ]]; then
  echo "Required script is missing or not executable: ${STOP_SCRIPT}" >&2
  exit 1
fi

exec "${STOP_SCRIPT}"
