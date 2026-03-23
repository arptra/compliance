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

export HOST_DATA_DIR="$DATA_DIR"
export HOST_PARQUET_REL="$PARQUET_REL"

docker compose down --volumes --remove-orphans || true

rm -rf "$HOST_DATA_DIR"/interim/* "$HOST_DATA_DIR"/processed/* "$HOST_DATA_DIR"/raw/* "$HOST_DATA_DIR"/uploads/* 2>/dev/null || true
rm -rf ./reports/* ./exports/* ./models/*
rm -f "$PARQUET_PATH"

echo "Stopped and wiped containers + generated files."
echo "Data dir wiped: $HOST_DATA_DIR"
echo "Removed parquet: $PARQUET_PATH"
