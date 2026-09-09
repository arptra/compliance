#!/usr/bin/env bash

setup_https_env() {
  local app_root="$1"
  if [[ "${HTTPS_ENABLED}" != "1" ]]; then
    return 0
  fi

  local cert_file="${TLS_CERT_FILE:-certs/server/fullchain.pem}"
  local key_file="${TLS_KEY_FILE:-certs/server/privkey.pem}"
  local chain_file="${TLS_CHAIN_FILE:-}"
  local ca_file="${TLS_CA_FILE:-}"
  [[ "${cert_file}" == /* ]] || cert_file="${app_root}/${cert_file}"
  [[ "${key_file}" == /* ]] || key_file="${app_root}/${key_file}"
  if [[ -n "${chain_file}" && "${chain_file}" != /* ]]; then
    chain_file="${app_root}/${chain_file}"
  fi
  if [[ -n "${ca_file}" ]]; then
    [[ "${ca_file}" == /* ]] || ca_file="${app_root}/${ca_file}"
    if [[ ! -r "${ca_file}" ]]; then
      echo "HTTPS CA file is not readable: ${ca_file}" >&2
      return 1
    fi
  fi
  if [[ ! -r "${cert_file}" || ! -r "${key_file}" ]]; then
    echo "HTTPS is enabled. Provide a server fullchain and private key:" >&2
    echo "  TLS_CERT_FILE=${cert_file}" >&2
    echo "  TLS_KEY_FILE=${key_file}" >&2
    echo "Save these paths and PUBLIC_HOST in ${app_root}/.env.vm (template: .env.vm.example)." >&2
    echo "For a separate intermediate bundle, also set TLS_CHAIN_FILE. See README.md." >&2
    return 1
  fi

  local python_bin="" candidate
  for candidate in "${app_root}/.venv311/bin/python" "${app_root}/.venv/bin/python" python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 && \
      "${candidate}" -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
      python_bin="${candidate}"
      break
    fi
  done
  if [[ -z "${python_bin}" ]]; then
    echo "Python 3.11+ is required to validate HTTPS certificates. Run scripts/setup_vm.sh first." >&2
    return 1
  fi

  local fullchain_file="${app_root}/.run/tls/fullchain.pem"
  local args=(--cert "${cert_file}" --key "${key_file}" --output "${fullchain_file}")
  if [[ -n "${chain_file}" ]]; then
    args+=(--chain "${chain_file}")
  fi
  "${python_bin}" "${app_root}/scripts/prepare_https.py" "${args[@]}" || return 1
  export TLS_CERT_FILE="${fullchain_file}"
  export TLS_KEY_FILE="${key_file}"
  export TLS_CA_FILE="${ca_file}"
}
