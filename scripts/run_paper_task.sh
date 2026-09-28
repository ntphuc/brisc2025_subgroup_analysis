#!/usr/bin/env bash
set -Eeuo pipefail

stamp(){ date '+%F %T'; }
fail(){ echo "[$(stamp)] FATAL | $*" >&2; exit 1; }
run(){ echo "[$(stamp)] + $(printf '%q ' "$@")"; "$@"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR:-$ROOT}"

if [[ -n "${VENV_DIR:-}" && -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
PYTHON_BIN="${PYTHON_BIN:-python}"
PAPER_STUDY="${PAPER_STUDY:-standard}"
ARCHITECTURE="${ARCHITECTURE:-unet_effb0}"
METHOD="${METHOD:-erm}"
MODEL_SEED="${MODEL_SEED:-42}"
DEVICE="${DEVICE:-auto}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DATA_ROOT="${BRISC_DATA_ROOT:-datasets/brisc2025}"

[[ -f train_paper.py ]] || fail "train_paper.py missing"
[[ -f data/brisc2025_manifest_paper_standard.csv ]] || fail "paper manifest missing; run scripts/prepare_data.sh first"

if [[ "$DEVICE" != "cpu" ]]; then
  "$PYTHON_BIN" - <<'PY'
import os, torch
print("torch", torch.__version__)
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<not-set>"))
print("cuda_available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
fi

MANIFEST="data/brisc2025_manifest_paper_standard.csv"
TRAIN_SPLIT="train"
VAL_SPLIT="val"
EVAL_SPLIT="test"
if [[ "$PAPER_STUDY" == "confirmatory" ]]; then
  [[ -f data/brisc2025_manifest_confirmatory.csv ]] || fail "confirmatory manifest missing; run scripts/prepare_data.sh first"
  MANIFEST="data/brisc2025_manifest_confirmatory.csv"
  EVAL_SPLIT="confirmatory"
elif [[ "$PAPER_STUDY" != "standard" ]]; then
  fail "PAPER_STUDY must be standard or confirmatory; got $PAPER_STUDY"
fi

RUN_DIR="outputs/paper_suite/${PAPER_STUDY}/${ARCHITECTURE}/${METHOD}/seed_${MODEL_SEED}"
RESULT_DIR="results/paper_suite/${PAPER_STUDY}/${ARCHITECTURE}/${METHOD}/seed_${MODEL_SEED}"
mkdir -p "$RUN_DIR" "$RESULT_DIR"

if [[ "$FORCE_TRAIN" == "1" ]]; then rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"; fi
if [[ ! -s "$RUN_DIR/best.pt" || ! -s "$RUN_DIR/training_summary.json" ]]; then
  run "$PYTHON_BIN" -u train_paper.py \
    --config configs/paper_suite.yaml \
    --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
    --train-split "$TRAIN_SPLIT" --val-split "$VAL_SPLIT" \
    --architecture "$ARCHITECTURE" --method "$METHOD" --seed "$MODEL_SEED" \
    --device "$DEVICE" --output-dir "$RUN_DIR"
else
  echo "[$(stamp)] REUSE training: $RUN_DIR/best.pt"
fi

if [[ "$FORCE_EVAL" == "1" ]]; then rm -rf "$RESULT_DIR"; mkdir -p "$RESULT_DIR"; fi

if [[ ! -s "$RESULT_DIR/fixed_0p5/per_case_metrics.csv" ]]; then
  run "$PYTHON_BIN" -u scripts/evaluate_paper_checkpoint.py \
    --checkpoint "$RUN_DIR/best.pt" --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
    --eval-split "$EVAL_SPLIT" --threshold 0.5 --device "$DEVICE" \
    --output-dir "$RESULT_DIR/fixed_0p5"
fi
if [[ ! -s "$RESULT_DIR/calibrated/per_case_metrics.csv" ]]; then
  run "$PYTHON_BIN" -u scripts/evaluate_paper_checkpoint.py \
    --checkpoint "$RUN_DIR/best.pt" --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
    --eval-split "$EVAL_SPLIT" --calibrate --calibrate-split "$VAL_SPLIT" \
    --device "$DEVICE" --output-dir "$RESULT_DIR/calibrated"
fi

run "$PYTHON_BIN" -u scripts/analyze_size_controlled.py \
  --metrics "$RESULT_DIR/calibrated/per_case_metrics.csv" \
  --training-summary "$RUN_DIR/training_summary.json" \
  --output-dir "$RESULT_DIR/size_controlled_analysis"

echo "[$(stamp)] PAPER TASK COMPLETE | study=$PAPER_STUDY arch=$ARCHITECTURE method=$METHOD seed=$MODEL_SEED"
echo "RESULT=$PWD/$RESULT_DIR/calibrated/summary.json"
