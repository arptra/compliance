#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESTART_SCRIPT="${ROOT_DIR}/scripts/restart_local_stack.sh"
RESOLVE_SCRIPT="${ROOT_DIR}/scripts/resolve_runtime_host.sh"

if [[ ! -x "${RESTART_SCRIPT}" ]]; then
  echo "Required script is missing or not executable: ${RESTART_SCRIPT}" >&2
  exit 1
fi

# VM defaults: keep them separate from local desktop ports.
export API_PORT="${API_PORT:-18000}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-15173}"

# shellcheck source=/dev/null
source "${RESOLVE_SCRIPT}"
setup_vm_runtime_env
source "${ROOT_DIR}/scripts/setup_https_env.sh"
# Validate replacement certificates before stopping a running stack.
setup_https_env "${ROOT_DIR}"

exec "${RESTART_SCRIPT}"
