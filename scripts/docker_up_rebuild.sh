#!/usr/bin/env bash
set -euo pipefail

PARQUET_PATH_INPUT="${1:-${HOST_PARQUET_PATH:-./data/processed/all_prepared.parquet}}"
PARQUET_PATH="$(python - <<'PY' "$PARQUET_PATH_INPUT"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
)"

mkdir -p "$(dirname "$PARQUET_PATH")" ./data/interim ./reports ./exports ./models
[ -f "$PARQUET_PATH" ] || touch "$PARQUET_PATH"

export HOST_PARQUET_PATH="$PARQUET_PATH"

echo "Using parquet file: $HOST_PARQUET_PATH"

docker compose down --remove-orphans || true
docker compose up -d --build

echo "API:        http://localhost:8000"
echo "Dashboard:  http://localhost:4173"
