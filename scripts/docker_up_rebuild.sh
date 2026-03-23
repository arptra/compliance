#!/usr/bin/env bash
set -euo pipefail

DATA_DIR_INPUT="${1:-${HOST_DATA_DIR:-./data}}"
PARQUET_REL="${2:-${HOST_PARQUET_REL:-processed/all_prepared.parquet}}"
DATA_DIR="$(python - <<'PY' "$DATA_DIR_INPUT"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
)"
PARQUET_PATH="$DATA_DIR/$PARQUET_REL"

mkdir -p "$DATA_DIR" "$(dirname "$PARQUET_PATH")" ./reports ./exports ./models
[ -f "$PARQUET_PATH" ] || touch "$PARQUET_PATH"

export HOST_DATA_DIR="$DATA_DIR"
export HOST_PARQUET_REL="$PARQUET_REL"

echo "Using data dir: $HOST_DATA_DIR"
echo "Using parquet relative path: $HOST_PARQUET_REL"
echo "Resolved parquet file: $PARQUET_PATH"

docker compose down --remove-orphans || true
docker compose up -d --build

echo "API:        http://localhost:8000"
echo "Dashboard:  http://localhost:4173"
