#!/usr/bin/env bash
set -euo pipefail

PARQUET_PATH_INPUT="${1:-${HOST_PARQUET_PATH:-./data/processed/all_prepared.parquet}}"
PARQUET_PATH="$(python - <<'PY' "$PARQUET_PATH_INPUT"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
)"

export HOST_PARQUET_PATH="$PARQUET_PATH"

docker compose down --volumes --remove-orphans || true

rm -rf ./data/interim/* ./reports/* ./exports/* ./models/*
rm -f "$HOST_PARQUET_PATH"

echo "Stopped and wiped containers + generated files. Removed parquet: $HOST_PARQUET_PATH"
