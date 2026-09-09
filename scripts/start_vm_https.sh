#!/usr/bin/env bash
set -euo pipefail

# Укажите только два пути: полная цепочка сертификатов и приватный ключ.
# Этот же блок используется scripts/start_vm.sh и scripts/restart_vm.sh.
setup_vm_launch_settings() {
  export TLS_CERT_FILE="${TLS_CERT_FILE:-certs/server/fullchain.pem}"
  export TLS_KEY_FILE="${TLS_KEY_FILE:-certs/server/privkey.pem}"
}

# Other VM entrypoints source this file to reuse the settings without launching.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  HTTPS_ENABLED=1 PUBLIC_SCHEME=https \
    exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_vm.sh" "$@"
fi
