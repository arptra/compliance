#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD_DIR="${ROOT_DIR}/apps/dashboard"
PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-${ROOT_DIR}/.venv311}"
MIN_NODE_MAJOR="${MIN_NODE_MAJOR:-18}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"
INSTALL_NODE_FROM_NODESOURCE="${INSTALL_NODE_FROM_NODESOURCE:-1}"
BUILD_DASHBOARD="${BUILD_DASHBOARD:-1}"

log() {
  printf '\n==> %s\n' "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

sudo_prefix() {
  if [[ "${EUID}" -eq 0 ]]; then
    return 0
  fi
  if ! have_cmd sudo; then
    echo "sudo is required for system package installation. Re-run as root or install sudo." >&2
    exit 1
  fi
  printf 'sudo'
}

apt_install() {
  local sudo_cmd
  sudo_cmd="$(sudo_prefix)"
  if [[ -n "${sudo_cmd}" ]]; then
    "${sudo_cmd}" apt-get install -y "$@"
  else
    apt-get install -y "$@"
  fi
}

apt_update() {
  local sudo_cmd
  sudo_cmd="$(sudo_prefix)"
  if [[ -n "${sudo_cmd}" ]]; then
    "${sudo_cmd}" apt-get update
  else
    apt-get update
  fi
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

node_major() {
  if ! have_cmd node; then
    printf '0'
    return 0
  fi
  node -p 'Number(process.versions.node.split(".")[0])' 2>/dev/null || printf '0'
}

ensure_system_packages() {
  if [[ "${INSTALL_SYSTEM_PACKAGES}" != "1" ]]; then
    log "Skipping system package installation"
    return 0
  fi

  if ! have_cmd apt-get; then
    echo "This setup script supports apt-based Linux distros. Install Python 3.11+, Node ${MIN_NODE_MAJOR}+, npm, git, curl and build-essential manually, then rerun with INSTALL_SYSTEM_PACKAGES=0." >&2
    exit 1
  fi

  log "Installing base system packages"
  apt_update
  apt_install ca-certificates curl git build-essential pkg-config python3.11 python3.11-venv python3.11-dev python3-pip

  local major
  major="$(node_major)"
  if (( major >= MIN_NODE_MAJOR )); then
    log "Node.js is already usable: $(node --version)"
    return 0
  fi

  if [[ "${INSTALL_NODE_FROM_NODESOURCE}" != "1" ]]; then
    echo "Node.js ${MIN_NODE_MAJOR}+ is required. Current major: ${major}. Install Node manually or rerun with INSTALL_NODE_FROM_NODESOURCE=1." >&2
    exit 1
  fi

  log "Installing Node.js 20 from NodeSource"
  local sudo_cmd
  sudo_cmd="$(sudo_prefix)"
  curl -fsSL https://deb.nodesource.com/setup_20.x -o /tmp/nodesource_setup.sh
  if [[ -n "${sudo_cmd}" ]]; then
    "${sudo_cmd}" bash /tmp/nodesource_setup.sh
  else
    bash /tmp/nodesource_setup.sh
  fi
  apt_install nodejs
  rm -f /tmp/nodesource_setup.sh
}

prepare_directories() {
  log "Preparing runtime directories"
  mkdir -p \
    "${ROOT_DIR}/.run/logs" \
    "${ROOT_DIR}/data/raw" \
    "${ROOT_DIR}/data/interim" \
    "${ROOT_DIR}/data/processed" \
    "${ROOT_DIR}/data/background" \
    "${ROOT_DIR}/data/gigachat_lab/versions" \
    "${ROOT_DIR}/data/lake" \
    "${ROOT_DIR}/reports" \
    "${ROOT_DIR}/exports" \
    "${ROOT_DIR}/models" \
    "${ROOT_DIR}/certs/server"
}

install_python_dependencies() {
  local python_bin
  if ! python_bin="$(resolve_python_bin)"; then
    echo "Python 3.11+ was not found after system package installation." >&2
    exit 1
  fi

  log "Creating Python virtualenv: ${PYTHON_ENV_DIR}"
  if [[ ! -x "${PYTHON_ENV_DIR}/bin/python" ]]; then
    "${python_bin}" -m venv "${PYTHON_ENV_DIR}"
  fi

  log "Installing Python dependencies"
  "${PYTHON_ENV_DIR}/bin/python" -m pip install --upgrade pip wheel
  "${PYTHON_ENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"
}

install_dashboard_dependencies() {
  if ! have_cmd npm; then
    echo "npm is required for dashboard setup." >&2
    exit 1
  fi

  log "Installing dashboard dependencies"
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

VM setup is ready.

Start the app:
  # Once: edit PUBLIC_HOST, TLS_CERT_FILE and TLS_KEY_FILE in scripts/start_vm_https.sh.
  scripts/start_vm_https.sh

Start and restart scripts reuse the settings in scripts/start_vm_https.sh.

HTTPS_ENABLED=1 is the VM default for both API and dashboard.
The public hostname must match the server certificate's SAN.

Useful overrides:
  PUBLIC_HOST=your.vm.host scripts/start_vm.sh
  API_PORT=18000 DASHBOARD_PORT=15173 scripts/start_vm.sh
  TLS_CERT_FILE=/path/to/fullchain.pem TLS_KEY_FILE=/path/to/privkey.pem PUBLIC_HOST=your.domain scripts/start_vm_https.sh
  HTTPS_ENABLED=0 scripts/start_vm.sh  # Explicit HTTP opt-out for testing

Default local dev user is created on first API access:
  email: dev@local
  password: dev-password

If GigaChat mTLS/token files are needed, put them into certs/ or set the
matching paths in configs/project.yaml / environment before start_vm.sh.
EOF
}

main() {
  cd "${ROOT_DIR}"
  ensure_system_packages
  prepare_directories
  install_python_dependencies
  install_dashboard_dependencies
  print_next_steps
}

main "$@"
