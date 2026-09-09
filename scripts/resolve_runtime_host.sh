#!/usr/bin/env bash
set -euo pipefail

trim_value() {
  local value="${1:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

is_usable_host_name() {
  local host
  host="$(trim_value "${1:-}")"
  [[ -n "${host}" ]] || return 1
  [[ "${host}" != "localhost" ]] || return 1
  [[ "${host}" != "localhost.localdomain" ]] || return 1
  [[ "${host}" != "127.0.0.1" ]] || return 1
  [[ "${host}" != "::1" ]] || return 1
  return 0
}

resolve_first_non_loopback_ip() {
  local candidate=""

  if command -v hostname >/dev/null 2>&1; then
    while read -r candidate; do
      candidate="$(trim_value "${candidate}")"
      [[ -n "${candidate}" ]] || continue
      [[ "${candidate}" == 127.* ]] && continue
      [[ "${candidate}" == "::1" ]] && continue
      printf '%s\n' "${candidate}"
      return 0
    done < <(hostname -I 2>/dev/null | tr ' ' '\n')
  fi

  if command -v ip >/dev/null 2>&1; then
    candidate="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i+1); exit}}')"
    candidate="$(trim_value "${candidate}")"
    if [[ -n "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  printf '127.0.0.1\n'
}

resolve_public_host_name() {
  local candidate=""

  for candidate in \
    "${PUBLIC_HOST:-}" \
    "${VM_HOSTNAME:-}" \
    "$(hostname -f 2>/dev/null || true)" \
    "$(hostname 2>/dev/null || true)"
  do
    candidate="$(trim_value "${candidate}")"
    if is_usable_host_name "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  resolve_first_non_loopback_ip
}

setup_vm_runtime_env() {
  export HTTPS_ENABLED="${HTTPS_ENABLED:-1}"
  local public_scheme="https"
  local hmr_protocol="ws"
  local resolved_public_host=""

  case "${HTTPS_ENABLED}" in
    1) public_scheme="https" ;;
    0) public_scheme="http" ;;
    *) echo "HTTPS_ENABLED must be 1 or 0" >&2; return 1 ;;
  esac
  if [[ -n "${PUBLIC_SCHEME:-}" && "${PUBLIC_SCHEME}" != "${public_scheme}" ]]; then
    echo "PUBLIC_SCHEME conflicts with HTTPS_ENABLED=${HTTPS_ENABLED}. Use HTTPS_ENABLED=0 for HTTP." >&2
    return 1
  fi
  export PUBLIC_SCHEME="${public_scheme}"

  resolved_public_host="$(resolve_public_host_name)"

  if [[ "${public_scheme}" == "https" ]]; then
    hmr_protocol="wss"
  fi

  export API_HOST="${API_HOST:-0.0.0.0}"
  export API_PORT="${API_PORT:-8000}"
  export DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
  export DASHBOARD_PORT="${DASHBOARD_PORT:-5173}"

  export RESOLVED_PUBLIC_HOST="${RESOLVED_PUBLIC_HOST:-${resolved_public_host}}"
  export API_PUBLIC_HOST="${API_PUBLIC_HOST:-${RESOLVED_PUBLIC_HOST}}"
  export DASHBOARD_PUBLIC_HOST="${DASHBOARD_PUBLIC_HOST:-${RESOLVED_PUBLIC_HOST}}"

  export VITE_API_BASE_URL="${VITE_API_BASE_URL:-${public_scheme}://${API_PUBLIC_HOST}:${API_PORT}}"
  export VITE_API_PORT="${VITE_API_PORT:-${API_PORT}}"
  export VITE_PUBLIC_ORIGIN="${VITE_PUBLIC_ORIGIN:-${public_scheme}://${DASHBOARD_PUBLIC_HOST}:${DASHBOARD_PORT}}"
  export VITE_HMR_HOST="${VITE_HMR_HOST:-${DASHBOARD_PUBLIC_HOST}}"
  export VITE_HMR_CLIENT_PORT="${VITE_HMR_CLIENT_PORT:-${DASHBOARD_PORT}}"
  export VITE_HMR_PROTOCOL="${VITE_HMR_PROTOCOL:-${hmr_protocol}}"
  export API_DISPLAY_URL="${API_DISPLAY_URL:-${public_scheme}://${API_PUBLIC_HOST}:${API_PORT}}"
  export DASHBOARD_DISPLAY_URL="${DASHBOARD_DISPLAY_URL:-${public_scheme}://${DASHBOARD_PUBLIC_HOST}:${DASHBOARD_PORT}}"

  if [[ "${HTTPS_ENABLED}" == "1" ]]; then
    local url
    for url in "${VITE_API_BASE_URL}" "${VITE_PUBLIC_ORIGIN}" "${API_DISPLAY_URL}" "${DASHBOARD_DISPLAY_URL}"; do
      if [[ "${url}" != https://* ]]; then
        echo "HTTPS is enabled, but an HTTP/invalid URL override was provided: ${url}" >&2
        return 1
      fi
    done
    if [[ "${VITE_HMR_PROTOCOL}" != "wss" ]]; then
      echo "HTTPS requires VITE_HMR_PROTOCOL=wss" >&2
      return 1
    fi
  fi
}
