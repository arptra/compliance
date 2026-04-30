#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="/app"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-/app/configs/project.yaml}"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-/tmp/project.runtime.yaml}"

# shellcheck source=/dev/null
source "${APP_ROOT}/scripts/setup_gigachat_ca_env.sh"
setup_gigachat_ca_env "${APP_ROOT}"

python /app/scripts/generate_runtime_config.py \
  --input "$CONFIG_TEMPLATE" \
  --output "$RUNTIME_CONFIG" \
  --prepare-output "${PREPARE_OUTPUT_PARQUET:-/app/data/processed/all_prepared.parquet}" \
  --interim-dir "${INTERIM_DIR:-/app/data/interim}" \
  --reports-dir "${REPORTS_DIR:-/app/reports}" \
  --exports-dir "${EXPORTS_DIR:-/app/exports}" \
  --models-dir "${MODELS_DIR:-/app/models}"

exec python /app/scripts/start_api.py
