#!/usr/bin/env bash

#SBATCH --job-name=pi05_abc130k_all
#SBATCH --chdir=/home/alex/policy-finetuning/yam-abc-reproduce/third_party/policy/openpi
#SBATCH --partition=defq
#SBATCH --exclude=cld2-bom-comp004
#SBATCH --output=/home/alex/policy-finetuning/policy-finetuning-logs/pi05-abc130k-all-data/slurm-%j.out
#SBATCH --error=/home/alex/policy-finetuning/policy-finetuning-logs/pi05-abc130k-all-data/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500GB
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# Full pi0.5 fine-tuning on every episode in ABC-130K, with global delta-action
# normalization statistics. The all_data_lr launchers select the learning rate.
# TASK_NAME and TASK_NORM_STATS_PATH are not used by this launcher.
#
# Inspect configuration and statistics without loading training rows or GPUs:
#   DRY_RUN=1 bash slurm/finetune_abc_pi05_all_data.bash
# Override FULL_NORM_STATS_PATH to use another full-dataset norm_stats.json
# with a matching adjacent global_stats_manifest.json.
# Checkpoints are saved under CHECKPOINT_BASE_DIR/<config>/<experiment>.
# Override CHECKPOINT_BASE_DIR to use another checkpoint filesystem.

set -euo pipefail

REPO_ROOT="/home/alex/policy-finetuning/yam-abc-reproduce"
OPENPI_DIR="${REPO_ROOT}/third_party/policy/openpi"
PYTHON="${REPO_ROOT}/.venv/bin/python"
source "${REPO_ROOT}/.venv/bin/activate"

CACHE_ROOT="${CACHE_ROOT:-/home/alex/policy-finetuning/openpi-cache}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/projects/data/datasets/checkpoints}"
DATASET_ROOT="${DATASET_ROOT:-/projects/data/datasets/abc_130k_v3_train}"
FULL_NORM_STATS_PATH="${FULL_NORM_STATS_PATH:-/home/alex/policy-finetuning/norm_stats/pi05_abc130k/abc130k_yam/norm_stats.json}"
NORM_STATS_DIR="$(dirname "${FULL_NORM_STATS_PATH}")"
# The cluster login environment can set a read-only HF_HOME.
export HF_HOME="${ALL_DATA_HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_LEROBOT_HOME="$(dirname "${DATASET_ROOT}")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${CACHE_ROOT}/openpi-assets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb-cache}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb-runs}"
export WANDB_ENTITY="${WANDB_ENTITY:-oneworld-ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-policy-finetuning}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export LEROBOT_DECODER_CACHE_SIZE="${LEROBOT_DECODER_CACHE_SIZE:-4}"

CONFIG_NAME="pi05_abc130k"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
EXP_NAME="${EXP_NAME:-pi05_abc130k_all_data_lr${LEARNING_RATE}_${SLURM_JOB_ID:-manual}}"
RUN_MODE="${RUN_MODE:-fresh}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
FSDP_DEVICES="${FSDP_DEVICES:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
KEEP_PERIOD="${KEEP_PERIOD:-${SAVE_INTERVAL}}"
MAX_TO_KEEP="${MAX_TO_KEEP:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}/${EXP_NAME}"

if [[ ! "${NUM_TRAIN_STEPS}" =~ ^[1-9][0-9]{0,5}$ ]] || (( NUM_TRAIN_STEPS > 200000 )); then
    echo "ERROR: NUM_TRAIN_STEPS must be between 1 and 200000." >&2
    exit 2
fi
if [[ "$(basename "${FULL_NORM_STATS_PATH}")" != "norm_stats.json" ]]; then
    echo "ERROR: FULL_NORM_STATS_PATH must point to a norm_stats.json file." >&2
    exit 2
fi

TRAIN_ARGS=(
    "${CONFIG_NAME}"
    --exp-name "${EXP_NAME}"
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
    --project-name "${WANDB_PROJECT}"
    --data.lerobot-root "${DATASET_ROOT}"
    --data.assets.assets-dir "$(dirname "${NORM_STATS_DIR}")"
    --data.assets.asset-id "$(basename "${NORM_STATS_DIR}")"
    --batch-size "${GLOBAL_BATCH_SIZE}"
    --fsdp-devices "${FSDP_DEVICES}"
    --num-workers "${NUM_WORKERS}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
    --save-interval "${SAVE_INTERVAL}"
    --keep-period "${KEEP_PERIOD}"
    --max-to-keep "${MAX_TO_KEEP}"
    --log-interval "${LOG_INTERVAL}"
    --seed "${SEED}"
    --lr-schedule.warmup-steps "${WARMUP_STEPS}"
    --lr-schedule.peak-lr "${LEARNING_RATE}"
    --lr-schedule.decay-steps 30000
    --lr-schedule.decay-lr "${LEARNING_RATE}"
)
case "${RUN_MODE}" in
    fresh) ;;
    resume) TRAIN_ARGS+=(--resume) ;;
    overwrite) TRAIN_ARGS+=(--overwrite) ;;
    *)
        echo "ERROR: RUN_MODE must be fresh, resume, or overwrite (got ${RUN_MODE})." >&2
        exit 2
        ;;
esac

mkdir -p "${HF_HOME}" "${OPENPI_DATA_HOME}" "${UV_CACHE_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DIR}"
cd "${OPENPI_DIR}"

echo "============================================================"
echo "pi0.5 ABC-130K all-data full fine-tuning"
echo "job_id=${SLURM_JOB_ID:-not-in-slurm}"
echo "dataset_root=${DATASET_ROOT}"
echo "episode_filter=none"
echo "norm_stats_path=${FULL_NORM_STATS_PATH}"
echo "learning_rate=${LEARNING_RATE}"
echo "warmup_steps=${WARMUP_STEPS}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "fsdp_devices=${FSDP_DEVICES}"
echo "num_train_steps=${NUM_TRAIN_STEPS}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "run_mode=${RUN_MODE}"
echo "dry_run=${DRY_RUN}"
echo "============================================================"

# Parse exactly the same CLI arguments that training will receive. This checks
# the supplied statistics against dataset metadata without scanning frame data.
"${PYTHON}" "${REPO_ROOT}/slurm/exec_without_thp.py" "${PYTHON}" - "${TRAIN_ARGS[@]}" <<'PY'
import json
from pathlib import Path

import numpy as np

from openpi.training import config as openpi_config
from yam_abc_reproduce.data.abc_delta_norm_stats import (
    dataset_fingerprint,
    discover_data_files,
    make_delta_mask,
)

config = openpi_config.cli()
root = Path(config.data.lerobot_root)
info = json.loads((root / "meta/info.json").read_text())
stats_dir = Path(config.data.assets.assets_dir) / config.data.assets.asset_id
manifest = json.loads((stats_dir / "global_stats_manifest.json").read_text())
files = discover_data_files(root)
expected = {
    "all_dataset_rows_used": True,
    "declared_total_frames": info["total_frames"],
    "state_count": info["total_frames"],
    "action_count": info["total_frames"] * config.model.action_horizon,
    "action_horizon": config.model.action_horizon,
    "delta_mask": list(make_delta_mask(config.data.num_arms)),
    "total_files": len(files),
    "dataset_fingerprint": dataset_fingerprint(root, files),
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise ValueError(f"Full-dataset normalization manifest mismatch: {key}")

data_config = config.data.create_base_config(config.assets_dirs, config.model)
if data_config.norm_stats is None:
    raise ValueError(f"Could not load normalization statistics from {stats_dir}")
for name in ("state", "actions"):
    for field in ("mean", "std", "q01", "q99"):
        values = np.asarray(getattr(data_config.norm_stats[name], field))
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid normalization statistics: {name}.{field}")

print(f"norm_stats=ok path={stats_dir / 'norm_stats.json'}", flush=True)
print("normalization_full_dataset_manifest=matched", flush=True)
print(
    f"all_data=ok episodes={info['total_episodes']} frames={info['total_frames']} "
    f"files={len(files)} episode_filter=none",
    flush=True,
)
print(
    f"training_config=ok batch_size={config.batch_size} fsdp_devices={config.fsdp_devices} "
    f"peak_lr={config.lr_schedule.peak_lr} decay_lr={config.lr_schedule.decay_lr}",
    flush=True,
)
print(f"training_checkpoint_dir={config.checkpoint_dir}", flush=True)
PY

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Full-dataset configuration and statistics inspection completed; training was not started."
    exit 0
fi

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

bash "${REPO_ROOT}/slurm/prepare_abc_pi05.bash" \
    "${REPO_ROOT}" "${DATASET_ROOT}" "${FSDP_DEVICES}" "${FULL_NORM_STATS_PATH}"

if [[ "${RUN_MODE}" == "resume" ]]; then
    echo "Resuming all-data training from ${CHECKPOINT_DIR}."
else
    echo "Starting all-data training from the pi0.5 base checkpoint."
fi
# Invoke OpenPI directly, without the task launcher's episode filtering patch.
# THP is disabled before importing training dependencies; workers inherit it.
exec "${PYTHON}" "${REPO_ROOT}/slurm/exec_without_thp.py" \
    "${PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}"
