#!/usr/bin/env bash
#SBATCH --job-name=brisc-paper-launch
#SBATCH --output=slurm_launcher_%j.out
#SBATCH --error=slurm_launcher_%j.err
#SBATCH --account=uit
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -Eeuo pipefail
export PAPER_PROFILE="${PAPER_PROFILE:-strong}"
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export BRISC_DATA_ROOT="${BRISC_DATA_ROOT:-$PROJECT_DIR/datasets/brisc2025}"
export DATA_SPLIT_SEED="${DATA_SPLIT_SEED:-42}"
export CONFIRMATORY_SPLIT_SEED="${CONFIRMATORY_SPLIT_SEED:-20260926}"
export PAPER_SEEDS="${PAPER_SEEDS:-42 43 44}"
export PAPER_ARCHITECTURES="${PAPER_ARCHITECTURES:-unet_effb0 unetpp_effb0 deeplabv3p_resnet50}"
export PRIMARY_ARCHITECTURE="${PRIMARY_ARCHITECTURE:-unet_effb0}"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"
exec bash scripts/submit_paper_suite.sh "$PAPER_PROFILE"
