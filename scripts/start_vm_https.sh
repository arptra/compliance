#!/usr/bin/env bash
set -euo pipefail

# Настройте эти значения один раз прямо здесь: домен и пути после :-.
# Этот же блок используется scripts/start_vm.sh и scripts/restart_vm.sh.
setup_vm_launch_settings() {
  export PUBLIC_HOST="${PUBLIC_HOST:-your.domain}"
  export TLS_CERT_FILE="${TLS_CERT_FILE:-certs/server/fullchain.pem}"
  export TLS_KEY_FILE="${TLS_KEY_FILE:-certs/server/privkey.pem}"

  # Необязательно: отдельная промежуточная цепочка и доверенный CA для прокси.
  export TLS_CHAIN_FILE="${TLS_CHAIN_FILE:-}"
  export TLS_CA_FILE="${TLS_CA_FILE:-}"
}

# Other VM entrypoints source this file to reuse the settings without launching.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  export HTTPS_ENABLED=1
  export PUBLIC_SCHEME=https
  exec "${ROOT_DIR}/scripts/start_vm.sh" "$@"
fi
