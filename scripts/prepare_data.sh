#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${BRISC_DATA_ROOT:-datasets/brisc2025}"
DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
CONFIRMATORY_SPLIT_SEED="${CONFIRMATORY_SPLIT_SEED:-20260926}"
AUTO_DOWNLOAD_DATA="${AUTO_DOWNLOAD_DATA:-0}"

if [[ ! -d "$DATA_ROOT/segmentation_task" && ! -d "$DATA_ROOT/brisc2025/segmentation_task" ]]; then
  if [[ "$AUTO_DOWNLOAD_DATA" == "1" ]]; then
    "$PYTHON_BIN" scripts/download_brisc2025.py --output "$DATA_ROOT"
  else
    cat >&2 <<EOF
BRISC2025 was not found at: $DATA_ROOT
Download it first with:
  python scripts/download_brisc2025.py
or set:
  export BRISC_DATA_ROOT=/path/to/brisc2025
EOF
    exit 2
  fi
fi

mkdir -p data
"$PYTHON_BIN" scripts/prepare_brisc2025.py \
  --data-root "$DATA_ROOT" \
  --output data/brisc2025_manifest.csv \
  --seed "$DATA_SPLIT_SEED" \
  --val-fraction 0.15
cp -f data/brisc2025_manifest.csv data/brisc2025_manifest_paper_standard.csv

"$PYTHON_BIN" scripts/make_confirmatory_manifest.py \
  --input data/brisc2025_manifest_paper_standard.csv \
  --output data/brisc2025_manifest_confirmatory.csv \
  --lock data/brisc2025_confirmatory_lock.json \
  --seed "$CONFIRMATORY_SPLIT_SEED"

"$PYTHON_BIN" scripts/verify_data_protocol.py
