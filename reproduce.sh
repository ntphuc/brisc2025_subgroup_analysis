#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-strong}"
case "$PROFILE" in quick|core|strong) ;; *) echo "Usage: bash reproduce.sh [quick|core|strong]" >&2; exit 2;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PROJECT_DIR="$ROOT"

bash scripts/prepare_data.sh
python scripts/save_environment.py
python scripts/lock_paper_protocol.py \
  --profile "$PROFILE" \
  --seeds "${PAPER_SEEDS:-42 43 44}" \
  --architectures "${PAPER_ARCHITECTURES:-unet_effb0 unetpp_effb0 deeplabv3p_resnet50}" \
  --primary-architecture "${PRIMARY_ARCHITECTURE:-unet_effb0}" \
  --data-split-seed "${DATA_SPLIT_SEED:-42}" \
  --confirmatory-split-seed "${CONFIRMATORY_SPLIT_SEED:-20260926}"

run_one(){
  PAPER_STUDY="$1" ARCHITECTURE="$2" METHOD="$3" MODEL_SEED="$4" \
    bash scripts/run_paper_task.sh
}

SEEDS="${PAPER_SEEDS:-42 43 44}"
ARCHS="${PAPER_ARCHITECTURES:-unet_effb0 unetpp_effb0 deeplabv3p_resnet50}"
PRIMARY="${PRIMARY_ARCHITECTURE:-unet_effb0}"

if [[ "$PROFILE" == "quick" ]]; then
  run_one standard "$PRIMARY" erm 42
  run_one standard "$PRIMARY" balanced 42
  run_one standard "$PRIMARY" groupdro 42
else
  for arch in $ARCHS; do
    for seed in $SEEDS; do run_one standard "$arch" erm "$seed"; done
  done
  for method in balanced groupdro; do
    for seed in $SEEDS; do run_one standard "$PRIMARY" "$method" "$seed"; done
  done
fi
if [[ "$PROFILE" == "strong" ]]; then
  for method in erm balanced groupdro; do
    for seed in $SEEDS; do run_one confirmatory "$PRIMARY" "$method" "$seed"; done
  done
fi

python scripts/aggregate_paper_suite.py
if [[ "$PROFILE" == "strong" ]]; then
  python scripts/build_paper_artifacts.py
fi
python scripts/check_paper_readiness.py
python scripts/verify_reproduction.py --allow-incomplete "$([[ "$PROFILE" == strong ]] && echo 0 || echo 1)"

echo "Done. Read results/paper_suite/aggregate/PAPER_SUITE_REPORT.md"
