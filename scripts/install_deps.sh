#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD_DIR="${ROOT_DIR}/apps/dashboard"
PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-${ROOT_DIR}/.venv311}"
BUILD_DASHBOARD="${BUILD_DASHBOARD:-1}"

log() {
  printf '\n==> %s\n' "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

python_is_311_plus() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

resolve_python_bin() {
  local candidate
  for candidate in "${PYTHON_BIN:-}" python3.11 python3; do
    [[ -n "${candidate}" ]] || continue
    if have_cmd "${candidate}" && python_is_311_plus "${candidate}"; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

install_python_deps() {
  local python_bin
  if ! python_bin="$(resolve_python_bin)"; then
    cat >&2 <<'EOF'
Python 3.11+ is required.

Ubuntu/Debian install example:
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip build-essential pkg-config
EOF
    exit 1
  fi

  log "Creating Python virtualenv: ${PYTHON_ENV_DIR}"
  if [[ ! -x "${PYTHON_ENV_DIR}/bin/python" ]]; then
    "${python_bin}" -m venv "${PYTHON_ENV_DIR}"
  fi

  # shellcheck source=/dev/null
  source "${PYTHON_ENV_DIR}/bin/activate"

  log "Installing Python dependencies from requirements.txt"
  python -m pip install --upgrade pip wheel
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
}

install_node_deps() {
  if ! have_cmd npm; then
    cat >&2 <<'EOF'
npm is required.

Ubuntu/Debian Node 20 install example:
  curl -fsSL https://deb.nodesource.com/setup_20.x -o /tmp/nodesource_setup.sh
  sudo bash /tmp/nodesource_setup.sh
  sudo apt-get install -y nodejs
EOF
    exit 1
  fi

  log "Installing dashboard Node dependencies"
  if [[ -f "${DASHBOARD_DIR}/package-lock.json" ]]; then
    (cd "${DASHBOARD_DIR}" && npm ci --no-fund --no-audit)
  else
    (cd "${DASHBOARD_DIR}" && npm install --no-fund --no-audit)
  fi

  if [[ "${BUILD_DASHBOARD}" == "1" ]]; then
    log "Building dashboard"
    (cd "${DASHBOARD_DIR}" && npm run build)
  fi
}

print_next_steps() {
  cat <<EOF

Dependencies are installed.

Activate Python environment in your current shell:
  source .venv311/bin/activate

Start on VM:
  # Put the server fullchain.pem and privkey.pem into certs/server/ first.
  PUBLIC_HOST=<certificate-dns-name> scripts/start_vm.sh

HTTPS is enabled by default. For other certificate paths:
  TLS_CERT_FILE=/path/to/fullchain.pem TLS_KEY_FILE=/path/to/privkey.pem PUBLIC_HOST=<certificate-dns-name> scripts/start_vm_https.sh
EOF
}

main() {
  cd "${ROOT_DIR}"
  install_python_deps
  install_node_deps
  print_next_steps
}

main "$@"
