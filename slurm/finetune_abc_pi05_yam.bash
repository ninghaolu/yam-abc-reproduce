#!/usr/bin/env bash

#SBATCH --job-name=pi05_abc130k
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce/third_party/policy/openpi
#SBATCH --account=torch_pr_147_courant
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc130k/slurm-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc130k/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500GB
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# Full pi0.5 fine-tuning on the local ABC-130K LeRobot v3 dataset.
#
# One-time setup before sbatch:
#   mkdir -p /projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc130k
#   /projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce/.venv/bin/wandb login --relogin
#
# Submit a fresh run:
#   sbatch /projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce/slurm/finetune_abc_pi05_yam.bash
#
# Common overrides:
#   EXP_NAME=pi05_abc130k_trial1 sbatch slurm/finetune_abc_pi05_yam.bash
#   RUN_MODE=resume EXP_NAME=pi05_abc130k_trial1 sbatch slurm/finetune_abc_pi05_yam.bash
#
# To use the current H200 reservation when applicable, submit with:
#   sbatch --reservation=my3149-h200 slurm/finetune_abc_pi05_yam.bash

set -euo pipefail

REPO_ROOT="/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce"
OPENPI_DIR="${REPO_ROOT}/third_party/policy/openpi"
PYTHON="${REPO_ROOT}/.venv/bin/python"
# TorchCodec loads FFmpeg through its shared libraries, not through the ffmpeg
# executable on PATH.  The cluster's FFmpeg 6 installation lives in the base
# Conda prefix while this job runs Python from the project venv.
FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-/scratch/nl2752/miniconda3/lib}"

CACHE_ROOT="${CACHE_ROOT:-/projects/work/yang-lab/projects/policy-finetuning/openpi-cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_LEROBOT_HOME="/projects/work/yang-lab/projects/pretrain_world_model"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${CACHE_ROOT}/openpi-assets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb-cache}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb-runs}"
export WANDB_ENTITY="${WANDB_ENTITY:-oneworld-ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-policy-finetuning}"
export WANDB_MODE="${WANDB_MODE:-online}"
# Keep this launcher's W&B identity separate from the default account in
# ~/.netrc. An explicitly exported WANDB_API_KEY still takes precedence.
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${REPO_ROOT}/.secrets/wandb_alex_api_key}"
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE}" == "online" ]]; then
    if [[ ! -r "${WANDB_API_KEY_FILE}" ]]; then
        echo "ERROR: W&B API key file is missing or unreadable: ${WANDB_API_KEY_FILE}" >&2
        exit 2
    fi
    IFS= read -r WANDB_API_KEY < "${WANDB_API_KEY_FILE}"
    if [[ -z "${WANDB_API_KEY}" || "${WANDB_API_KEY}" =~ [[:space:]] ]]; then
        echo "ERROR: W&B API key file is empty or malformed: ${WANDB_API_KEY_FILE}" >&2
        exit 2
    fi
    export WANDB_API_KEY
fi
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# LeRobot caches 32 open video decoders per data-loader worker. ABC-130K spreads
# 38k video files averaging 173 MB, so a shuffled batch almost never reuses a
# cached decoder and the default only buys memory pressure. Holding the job below
# its cgroup limit is what keeps Orbax's host-side checkpoint staging fast.
export LEROBOT_DECODER_CACHE_SIZE="${LEROBOT_DECODER_CACHE_SIZE:-4}"

CONFIG_NAME="pi05_abc130k"
EXP_NAME="${EXP_NAME:-pi05_abc130k_${SLURM_JOB_ID:-manual}}"
RUN_MODE="${RUN_MODE:-fresh}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-80000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
# Keep every regular checkpoint. If SAVE_INTERVAL is overridden, set
# KEEP_PERIOD to the same value to retain every saved step.
KEEP_PERIOD="${KEEP_PERIOD:-2000}"
MAX_TO_KEEP="${MAX_TO_KEEP:-1000}"
# Shard model parameters across both H200s; the batch size remains global.
FSDP_DEVICES="${FSDP_DEVICES:-2}"

CHECKPOINT_DIR="${OPENPI_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"

case "${RUN_MODE}" in
    fresh)
        TRAIN_STATE_ARGS=()
        ;;
    resume)
        TRAIN_STATE_ARGS=(--resume)
        ;;
    overwrite)
        TRAIN_STATE_ARGS=(--overwrite)
        ;;
    *)
        echo "ERROR: RUN_MODE must be fresh, resume, or overwrite (got ${RUN_MODE})." >&2
        exit 2
        ;;
esac

mkdir -p \
    "${HF_HOME}" \
    "${OPENPI_DATA_HOME}" \
    "${UV_CACHE_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_DIR}"

cd "${OPENPI_DIR}"

echo "============================================================"
echo "pi0.5 ABC-130K YAM full fine-tuning"
echo "job_id=${SLURM_JOB_ID:-not-in-slurm}"
echo "host=$(hostname)"
echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
echo "config=${CONFIG_NAME}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "num_workers=${NUM_WORKERS}"
echo "lerobot_decoder_cache_size=${LEROBOT_DECODER_CACHE_SIZE}"
echo "num_train_steps=${NUM_TRAIN_STEPS}"
echo "save_interval=${SAVE_INTERVAL}"
echo "keep_period=${KEEP_PERIOD}"
echo "max_to_keep=${MAX_TO_KEEP}"
echo "fsdp_devices=${FSDP_DEVICES}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "wandb=${WANDB_ENTITY}/${WANDB_PROJECT}/${EXP_NAME}"
echo "run_mode=${RUN_MODE}"
echo "============================================================"

echo "Starting training."
# Disable THP before importing training dependencies; workers inherit it.
/usr/bin/python3 "${REPO_ROOT}/slurm/exec_without_thp.py" \
    "${PYTHON}" scripts/train.py "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --project-name "${WANDB_PROJECT}" \
    --batch-size "${GLOBAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --save-interval "${SAVE_INTERVAL}" \
    --keep-period "${KEEP_PERIOD}" \
    --max-to-keep "${MAX_TO_KEEP}" \
    --fsdp-devices "${FSDP_DEVICES}" \
    "${TRAIN_STATE_ARGS[@]}"

echo "Training completed successfully."
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "wandb_project=https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
