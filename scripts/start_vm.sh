#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_SCRIPT="${ROOT_DIR}/scripts/start_local_stack.sh"
RESOLVE_SCRIPT="${ROOT_DIR}/scripts/resolve_runtime_host.sh"

if [[ ! -x "${START_SCRIPT}" ]]; then
  echo "Required script is missing or not executable: ${START_SCRIPT}" >&2
  exit 1
fi

# Reuse the two certificate paths in the HTTPS launcher.
source "${ROOT_DIR}/scripts/start_vm_https.sh"
setup_vm_launch_settings

# VM defaults: keep them separate from local desktop ports.
export API_PORT="${API_PORT:-18000}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-15173}"

# shellcheck source=/dev/null
source "${RESOLVE_SCRIPT}"
setup_vm_runtime_env
source "${ROOT_DIR}/scripts/setup_https_env.sh"
setup_https_env "${ROOT_DIR}"

cat <<EOF
Resolved VM runtime settings:
  Public host:       ${RESOLVED_PUBLIC_HOST}
  API bind host:     ${API_HOST}
  Dashboard bind:    ${DASHBOARD_HOST}
  API public URL:    ${API_DISPLAY_URL}
  Dashboard URL:     ${DASHBOARD_DISPLAY_URL}
  Dashboard -> API:  ${VITE_API_BASE_URL}
  HTTPS enabled:     ${HTTPS_ENABLED}
  TLS fullchain:     ${TLS_CERT_FILE:-disabled}
EOF

exec "${START_SCRIPT}"
