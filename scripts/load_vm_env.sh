#!/usr/bin/env bash

load_vm_env() {
  local app_root="$1"
  local env_file="${VM_ENV_FILE:-${app_root}/.env.vm}"
  [[ "${env_file}" == /* ]] || env_file="${app_root}/${env_file}"
  if [[ ! -e "${env_file}" && -z "${VM_ENV_FILE:-}" ]]; then
    return 0
  fi
  if [[ ! -f "${env_file}" || ! -r "${env_file}" ]]; then
    echo "VM settings file is not readable: ${env_file}" >&2
    return 1
  fi

  local line name value line_number=0 loaded_names=" "
  local assignment='^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$'
  local double_quoted='^"([^"]*)"[[:space:]]*(#.*)?$'
  local single_quoted="^'([^']*)'[[:space:]]*(#.*)?$"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line_number=$((line_number + 1))
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    [[ "${line}" != export\ * ]] || line="${line#export }"
    if [[ ! "${line}" =~ ${assignment} ]]; then
      echo "Invalid VM setting at ${env_file}:${line_number}; expected NAME=value" >&2
      return 1
    fi
    name="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    case "${name}" in
      PUBLIC_HOST|PUBLIC_SCHEME|VM_HOSTNAME|HTTPS_ENABLED|TLS_CERT_FILE|TLS_KEY_FILE|TLS_CHAIN_FILE|TLS_CA_FILE|TLS_KEY_PASSWORD|\
      API_HOST|API_PORT|API_PUBLIC_HOST|API_DISPLAY_URL|DASHBOARD_HOST|DASHBOARD_PORT|DASHBOARD_PUBLIC_HOST|DASHBOARD_DISPLAY_URL|\
      VITE_API_BASE_URL|VITE_API_PORT|VITE_PUBLIC_ORIGIN|VITE_HMR_HOST|VITE_HMR_CLIENT_PORT|VITE_HMR_PROTOCOL|GIGACHAT_CA_BUNDLE_FILE) ;;
      *) echo "Unsupported VM setting at ${env_file}:${line_number}: ${name}" >&2; return 1 ;;
    esac
    value="${value#"${value%%[![:space:]]*}"}"
    if [[ "${value}" == \"* || "${value}" == \'* ]]; then
      if [[ "${value}" =~ ${double_quoted} || "${value}" =~ ${single_quoted} ]]; then
        value="${BASH_REMATCH[1]}"
      else
        echo "Invalid quoted VM setting at ${env_file}:${line_number}" >&2
        return 1
      fi
    else
      value="${value%%[[:space:]]#*}"
      value="${value%"${value##*[![:space:]]}"}"
    fi
    # Preserve explicit environment overrides, including intentionally empty values.
    # Never source/eval the file: passwords and paths must remain literal data.
    if [[ -z "${!name+x}" || "${loaded_names}" == *" ${name} "* ]]; then
      export "${name}=${value}"
      loaded_names+="${name} "
    fi
  done < "${env_file}"
}
