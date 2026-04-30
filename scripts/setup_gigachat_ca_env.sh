#!/usr/bin/env bash

setup_gigachat_ca_env() {
  local app_root="${1:-}"
  if [[ -z "${app_root}" ]]; then
    app_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi

  local ca_bundle="${GIGACHAT_CA_BUNDLE_FILE:-${app_root}/certs/ca.pem}"
  if [[ ! -f "${ca_bundle}" ]]; then
    echo "GigaChat CA bundle is missing: ${ca_bundle}" >&2
    return 1
  fi

  export GIGACHAT_CA_BUNDLE_FILE="${ca_bundle}"
  export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-${ca_bundle}}"
  export NODE_EXTRA_CA_CERTS="${NODE_EXTRA_CA_CERTS:-${ca_bundle}}"
  export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH="${GRPC_DEFAULT_SSL_ROOTS_FILE_PATH:-${ca_bundle}}"
}
