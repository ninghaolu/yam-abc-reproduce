#!/usr/bin/env bash

#SBATCH --job-name=pi05_abc_valerr
#SBATCH --chdir=/home/alex/policy-finetuning/yam-abc-reproduce/third_party/policy/openpi
#SBATCH --partition=defq
#SBATCH --exclude=cld2-bom-comp004
#SBATCH --output=/home/alex/policy-finetuning/policy-finetuning-logs/eval-action-error/slurm-%j.out
#SBATCH --error=/home/alex/policy-finetuning/policy-finetuning-logs/eval-action-error/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=08:00:00
#SBATCH --export=ALL

# Validation action error and training flow loss for a trained pi0.5 checkpoint, scored on the held-out
# abc_130k_v3_val split.  This is an offline evaluation: it does not touch the
# training loop and does not write into the checkpoint directory.
#
# One-time setup before sbatch:
#   mkdir -p /home/alex/policy-finetuning/policy-finetuning-logs/eval-action-error
#
# Check the episode selection without a GPU (safe on a login node):
#   DRY_RUN=1 bash slurm/eval_abc_pi05_action_error.bash
#
# Score the default checkpoint:
#   sbatch slurm/eval_abc_pi05_action_error.bash
#
# Score a different step or task:
#   CHECKPOINT_DIR=.../pi05_abc130k_task/<run>/4000 sbatch slurm/eval_abc_pi05_action_error.bash
#   TASK_NAME='put the plastic bottles in the bin' sbatch slurm/eval_abc_pi05_action_error.bash
#
# Quick smoke test on a few batches:
#   MAX_BATCHES=4 sbatch slurm/eval_abc_pi05_action_error.bash
#
# Statistics default to the checkpoint's saved assets (global or task-specific).
# Override explicitly with NORM_STATS_PATH=/.../<task>/norm_stats.json.
# COMPUTE_VAL_LOSS=0 restores action-error-only evaluation.

set -euo pipefail

REPO_ROOT="/home/alex/policy-finetuning/yam-abc-reproduce"
OPENPI_DIR="${REPO_ROOT}/third_party/policy/openpi"
PYTHON="${REPO_ROOT}/.venv/bin/python"
# Activation supplies the project-local FFmpeg shared libraries for TorchCodec.
source "${REPO_ROOT}/.venv/bin/activate"

CACHE_ROOT="${CACHE_ROOT:-/home/alex/policy-finetuning/openpi-cache}"
export HF_HOME="${EVAL_HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/projects/data/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${CACHE_ROOT}/openpi-assets}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
if [[ -n "${FFMPEG_LIB_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
# See the note in finetune_abc_pi05_yam.bash: LeRobot's default 32-decoder cache
# buys nothing here and only adds memory pressure.
export LEROBOT_DECODER_CACHE_SIZE="${LEROBOT_DECODER_CACHE_SIZE:-4}"

BASE_CONFIG="${BASE_CONFIG:-pi05_abc130k}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/projects/data/datasets/checkpoints/pi05_abc130k/pi05_abc130k_all_data_lr1e-5_91/8000}"
TASK_NAME="${TASK_NAME:-put the plastic bottles in the bin}"
VAL_REPO_ID="${VAL_REPO_ID:-abc_130k_v3_val}"
VAL_ROOT="${VAL_ROOT:-/projects/data/datasets/abc_130k_v3_val}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-10}"
# Fixes the flow-matching noise, so two checkpoints scored with the same seed are
# directly comparable.
SEED="${SEED:-0}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
DRY_RUN="${DRY_RUN:-0}"
COMPUTE_VAL_LOSS="${COMPUTE_VAL_LOSS:-1}"

EVAL_ARGS=(
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --task-name "${TASK_NAME}"
    --base-config "${BASE_CONFIG}"
    --val-repo-id "${VAL_REPO_ID}"
    --val-root "${VAL_ROOT}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --num-denoising-steps "${NUM_DENOISING_STEPS}"
    --seed "${SEED}"
    --fsdp-devices "${FSDP_DEVICES}"
)
if [[ -n "${MAX_BATCHES:-}" ]]; then
    EVAL_ARGS+=(--max-batches "${MAX_BATCHES}")
fi
if [[ -n "${OUTPUT:-}" ]]; then
    EVAL_ARGS+=(--output "${OUTPUT}")
fi
if [[ -n "${NORM_STATS_PATH:-}" ]]; then
    EVAL_ARGS+=(--norm-stats-path "${NORM_STATS_PATH}")
fi
if [[ "${COMPUTE_VAL_LOSS}" == "0" ]]; then
    EVAL_ARGS+=(--no-compute-val-loss)
fi

if [[ ! -d "${CHECKPOINT_DIR}/params" ]]; then
    echo "ERROR: no params/ under ${CHECKPOINT_DIR}; point CHECKPOINT_DIR at a step directory." >&2
    exit 2
fi

mkdir -p "${HF_HOME}" "${OPENPI_DATA_HOME}"

cd "${OPENPI_DIR}"

echo "============================================================"
echo "pi0.5 ABC-130K validation action error"
echo "job_id=${SLURM_JOB_ID:-not-in-slurm}"
echo "host=$(hostname)"
echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "base_config=${BASE_CONFIG}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "task_name=${TASK_NAME}"
echo "val_root=${VAL_ROOT}"
echo "batch_size=${BATCH_SIZE}"
echo "num_denoising_steps=${NUM_DENOISING_STEPS}"
echo "max_batches=${MAX_BATCHES:-all}"
echo "seed=${SEED}"
echo "dry_run=${DRY_RUN}"
echo "compute_val_loss=${COMPUTE_VAL_LOSS}"
echo "norm_stats_path=${NORM_STATS_PATH:-auto-from-checkpoint}"
echo "============================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    JAX_PLATFORMS=cpu "${PYTHON}" "${REPO_ROOT}/slurm/exec_without_thp.py" "${PYTHON}" "${REPO_ROOT}/scripts/eval_action_error.py" "${EVAL_ARGS[@]}" --dry-run
    echo "Dry run completed; the model was not loaded."
    exit 0
fi

"${PYTHON}" "${REPO_ROOT}/slurm/exec_without_thp.py" "${PYTHON}" "${REPO_ROOT}/scripts/eval_action_error.py" "${EVAL_ARGS[@]}"

echo "Evaluation completed successfully."
