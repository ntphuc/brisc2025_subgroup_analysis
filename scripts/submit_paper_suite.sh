#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-strong}"
case "$PROFILE" in quick|core|strong) ;; *) echo "Usage: $0 [quick|core|strong]" >&2; exit 2;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$ROOT}"
cd "$PROJECT_DIR"
if [[ -n "${VENV_DIR:-}" && -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PROJECT_DIR
export BRISC_DATA_ROOT="${BRISC_DATA_ROOT:-$PROJECT_DIR/datasets/brisc2025}"
SEEDS="${PAPER_SEEDS:-42 43 44}"
ARCHS="${PAPER_ARCHITECTURES:-unet_effb0 unetpp_effb0 deeplabv3p_resnet50}"
PRIMARY="${PRIMARY_ARCHITECTURE:-unet_effb0}"
DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
CONFIRMATORY_SPLIT_SEED="${CONFIRMATORY_SPLIT_SEED:-20260926}"

# Prepare/verify data before fan-out so worker jobs never race while writing a manifest.
bash scripts/prepare_data.sh
python scripts/save_environment.py
python scripts/lock_paper_protocol.py \
  --profile "$PROFILE" --seeds "$SEEDS" --architectures "$ARCHS" \
  --primary-architecture "$PRIMARY" --data-split-seed "$DATA_SPLIT_SEED" \
  --confirmatory-split-seed "$CONFIRMATORY_SPLIT_SEED"

mkdir -p logs results/paper_suite/submissions
jobs=()
SUBMISSION_STAMP="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
SUBMISSION_LOG="results/paper_suite/submissions/submission_${SUBMISSION_STAMP}.txt"
{
  echo "profile=$PROFILE"
  echo "launcher_job_id=${SLURM_JOB_ID:-manual}"
  echo "seeds=$SEEDS"
  echo "architectures=$ARCHS"
  echo "primary_architecture=$PRIMARY"
  echo "data_root=$BRISC_DATA_ROOT"
} > "$SUBMISSION_LOG"

submit_one(){
  local study="$1" arch="$2" method="$3" seed="$4"
  local name="p-${study:0:3}-${arch:0:5}-${method:0:3}-s${seed}"
  local jid
  jid=$(sbatch --parsable --job-name="$name" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,BRISC_DATA_ROOT=$BRISC_DATA_ROOT,PAPER_STUDY=$study,ARCHITECTURE=$arch,METHOD=$method,MODEL_SEED=$seed" \
    run_paper_worker.slurm)
  jid="${jid%%;*}"
  jobs+=("$jid")
  echo "submitted $jid | $study $arch $method seed=$seed"
  echo "experiment_job=$jid study=$study architecture=$arch method=$method seed=$seed" >> "$SUBMISSION_LOG"
}

if [[ "$PROFILE" == "quick" ]]; then
  submit_one standard "$PRIMARY" erm 42
  submit_one standard "$PRIMARY" balanced 42
  submit_one standard "$PRIMARY" groupdro 42
else
  for arch in $ARCHS; do
    for seed in $SEEDS; do submit_one standard "$arch" erm "$seed"; done
  done
  for method in balanced groupdro; do
    for seed in $SEEDS; do submit_one standard "$PRIMARY" "$method" "$seed"; done
  done
fi

if [[ "$PROFILE" == "strong" ]]; then
  for method in erm balanced groupdro; do
    for seed in $SEEDS; do submit_one confirmatory "$PRIMARY" "$method" "$seed"; done
  done
fi

if [[ ${#jobs[@]} -eq 0 ]]; then
  echo "No worker jobs were submitted" >&2; exit 3
fi
dep=$(IFS=:; echo "${jobs[*]}")
agg=$(sbatch --parsable --dependency="afterok:$dep" \
  --export="ALL,PROJECT_DIR=$PROJECT_DIR,BRISC_DATA_ROOT=$BRISC_DATA_ROOT" \
  run_paper_aggregate.slurm)
agg="${agg%%;*}"
echo "aggregate_job=$agg dependency=afterok:$dep" >> "$SUBMISSION_LOG"
printf 'TOTAL experiment jobs: %d\n' "${#jobs[@]}"
echo "Aggregate job: $agg"
echo "Monitor: squeue -u $USER"
echo "Final report: results/paper_suite/aggregate/PAPER_SUITE_REPORT.md"
