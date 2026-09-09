#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HTTPS_ENABLED=1
export PUBLIC_SCHEME=https
exec "${ROOT_DIR}/scripts/start_vm.sh" "$@"
