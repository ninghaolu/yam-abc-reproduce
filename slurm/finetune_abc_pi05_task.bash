#!/usr/bin/env bash

#SBATCH --job-name=pi05_abc_task
#SBATCH --chdir=/home/alex/policy-finetuning/yam-abc-reproduce/third_party/policy/openpi
#SBATCH --partition=defq
#SBATCH --output=/home/alex/policy-finetuning/policy-finetuning-logs/pi05-abc-task/slurm-%j.out
#SBATCH --error=/home/alex/policy-finetuning/policy-finetuning-logs/pi05-abc-task/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500GB
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# Task-specific pi0.5 full fine-tuning on the local ABC-130K LeRobot v3 dataset.
# The two learning-rate launchers delegate to this script.
#
# The pinned LeRobot reader already supports an `episodes=` filter, but this
# OpenPI checkout does not expose that option in its config. To keep this
# task filter self-contained, the Python block below injects the
# selected episode IDs into LeRobotDataset at runtime. It does not edit OpenPI.
#
# Inspect the default task without training (safe on a login node):
#   DRY_RUN=1 bash slurm/finetune_abc_pi05_task.bash
#
# Default task: put the plastic bottles in the bin (3,793 episodes / 63.3 h):
#   module load slurm/slurm/23.02.8
#   mkdir -p /home/alex/policy-finetuning/policy-finetuning-logs/pi05-abc-task
#   sbatch slurm/finetune_abc_pi05_task.bash
#
# Use the exact natural-language task from meta/tasks.parquet. TASK_SLUG is
# derived automatically and isolates checkpoints. Normalization statistics are
# loaded from ../norm_stats/norm_stats_abc/${TASK_SLUG}/norm_stats.json. Override
# TASK_NORM_STATS_PATH to use a different task statistics file.
#
# specify norm stats, task name

set -euo pipefail

REPO_ROOT="/home/alex/policy-finetuning/yam-abc-reproduce"
OPENPI_DIR="${REPO_ROOT}/third_party/policy/openpi"
PYTHON="${REPO_ROOT}/.venv/bin/python"
source "${REPO_ROOT}/.venv/bin/activate"

CACHE_ROOT="${TASK_CACHE_ROOT:-${CACHE_ROOT:-/home/alex/policy-finetuning/openpi-cache}}"
export DATASET_ROOT="${DATASET_ROOT:-/projects/data/datasets/abc_130k_v3_train}"
# Do not inherit the cluster login environment's read-only HF_HOME. Override
# TASK_HF_HOME explicitly if a different writable cache is desired.
export HF_HOME="${TASK_HF_HOME:-${CACHE_ROOT}/huggingface}"
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
# LeRobot caches 32 open video decoders per data-loader worker. ABC-130K spreads
# 38k video files averaging 173 MB, so a shuffled batch almost never reuses a
# cached decoder and the default only buys memory pressure. Holding the job below
export LEROBOT_DECODER_CACHE_SIZE="${LEROBOT_DECODER_CACHE_SIZE:-4}"

export BASE_CONFIG_NAME="pi05_abc130k"
export TASK_CONFIG_NAME="pi05_abc130k_task"
# specify tasks
export TASK_NAME="${TASK_NAME:-put the plastic bottles in the bin}"
if [[ -z "${TASK_SLUG:-}" ]]; then
    TASK_SLUG="$(
        printf '%s' "${TASK_NAME}" \
            | tr '[:upper:] ' '[:lower:]_' \
            | tr -cd '[:alnum:]_-' \
            | sed -E 's/_+/_/g; s/^_//; s/_$//'
    )"
fi
if [[ -z "${TASK_SLUG}" ]]; then
    echo "ERROR: TASK_NAME did not produce a usable TASK_SLUG." >&2
    exit 2
fi
export TASK_SLUG
# specify norm stats
export TASK_NORM_STATS_PATH="${TASK_NORM_STATS_PATH:-/home/alex/policy-finetuning/norm_stats/norm_stats_abc/${TASK_SLUG}/norm_stats.json}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-1000}"
export EXP_NAME="${EXP_NAME:-pi05_abc_${TASK_SLUG}_lr${LEARNING_RATE}_${SLURM_JOB_ID:-manual}}"
export RUN_MODE="${RUN_MODE:-fresh}"
# Batch 256 requires four H100 80 GB GPUs; two exhausted memory at the first update.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200000}"
if [[ ! "${NUM_TRAIN_STEPS}" =~ ^[1-9][0-9]{0,5}$ ]] || (( NUM_TRAIN_STEPS > 200000 )); then
    echo "ERROR: NUM_TRAIN_STEPS must be between 1 and 200000." >&2
    exit 2
fi
export SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
# Keep every regular checkpoint by default, including when SAVE_INTERVAL changes.
export KEEP_PERIOD="${KEEP_PERIOD:-${SAVE_INTERVAL}}"
export MAX_TO_KEEP="${MAX_TO_KEEP:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export SEED="${SEED:-42}"
# Shard model parameters across all four allocated GPUs; the batch size remains global.
export FSDP_DEVICES="${FSDP_DEVICES:-4}"
export DRY_RUN="${DRY_RUN:-0}"

CHECKPOINT_DIR="${OPENPI_DIR}/checkpoints/${TASK_CONFIG_NAME}/${EXP_NAME}"

case "${RUN_MODE}" in
    fresh | resume | overwrite) ;;
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

# Run one isolated Python process for inspection or training.
# Patching __init__ (instead of replacing the class) keeps dataset objects
# pickleable by spawned PyTorch data-loader workers.
run_openpi_task() {
    "${PYTHON}" "${REPO_ROOT}/slurm/exec_without_thp.py" \
        "${PYTHON}" -c 'import sys; exec(sys.stdin.read())' "$1" <<'PY'
import dataclasses
import difflib
import importlib
import json
import os
from pathlib import Path
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from openpi.training import config as openpi_config
from openpi.training import data_loader


mode = sys.argv[1]
task_name = os.environ["TASK_NAME"].strip()
if not task_name:
    raise ValueError("TASK_NAME must be non-empty")
scripts_dir = str(Path.cwd() / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)


def selected_episodes(meta):
    episode_indices = meta.episodes["episode_index"]
    episode_tasks = meta.episodes["tasks"]
    episode_lengths = meta.episodes["length"]
    available = sorted({str(task) for tasks in episode_tasks for task in tasks})
    if task_name not in available:
        suggestions = difflib.get_close_matches(task_name, available, n=5, cutoff=0.2)
        raise ValueError(f"Unknown exact ABC task {task_name!r}. Closest tasks: {suggestions}")

    selected = []
    frames = 0
    mixed = []
    for episode_index, tasks, length in zip(
        episode_indices, episode_tasks, episode_lengths, strict=True
    ):
        names = {str(task) for task in tasks}
        if task_name not in names:
            continue
        if names != {task_name}:
            mixed.append((int(episode_index), sorted(names)))
            continue
        selected.append(int(episode_index))
        frames += int(length)
    if mixed:
        raise ValueError(
            "The requested task occurs in mixed-task episodes, which cannot be safely "
            f"selected as whole episodes. First examples: {mixed[:5]}"
        )
    if not selected:
        raise ValueError(f"No episodes found for {task_name!r}")
    return selected, frames


base = openpi_config.get_config(os.environ["BASE_CONFIG_NAME"])
# OpenPI loads assets_dir / asset_id / norm_stats.json and saves those stats
# with checkpoints for inference.
norm_stats_path = Path(os.environ["TASK_NORM_STATS_PATH"]).resolve(strict=True)
if norm_stats_path.name != "norm_stats.json":
    raise ValueError("TASK_NORM_STATS_PATH must point to a norm_stats.json file")
data_factory = dataclasses.replace(
    base.data,
    lerobot_root=os.environ["DATASET_ROOT"],
    assets=dataclasses.replace(
        base.data.assets,
        assets_dir=str(norm_stats_path.parent.parent),
        asset_id=norm_stats_path.parent.name,
    ),
)
config = dataclasses.replace(
    base,
    name=os.environ["TASK_CONFIG_NAME"],
    data=data_factory,
    project_name=os.environ["WANDB_PROJECT"],
    exp_name=os.environ["EXP_NAME"],
    batch_size=int(os.environ["GLOBAL_BATCH_SIZE"]),
    num_workers=int(os.environ["NUM_WORKERS"]),
    num_train_steps=int(os.environ["NUM_TRAIN_STEPS"]),
    save_interval=int(os.environ["SAVE_INTERVAL"]),
    max_to_keep=int(os.environ["MAX_TO_KEEP"]),
    keep_period=int(os.environ["KEEP_PERIOD"]),
    log_interval=int(os.environ["LOG_INTERVAL"]),
    seed=int(os.environ["SEED"]),
    fsdp_devices=int(os.environ["FSDP_DEVICES"]),
    lr_schedule=dataclasses.replace(
        base.lr_schedule,
        warmup_steps=int(os.environ["WARMUP_STEPS"]),
        peak_lr=float(os.environ["LEARNING_RATE"]),
        decay_lr=float(os.environ["LEARNING_RATE"]),
        decay_steps=30000,
    ),
    resume=os.environ["RUN_MODE"] == "resume",
    overwrite=os.environ["RUN_MODE"] == "overwrite",
)

# Validate the actual asset-loading path even during a dry run.
norm_config = data_factory.create_base_config(config.assets_dirs, config.model)
if norm_config.norm_stats is None or not {"state", "actions"} <= norm_config.norm_stats.keys():
    raise ValueError(f"Missing state/actions normalization statistics: {norm_stats_path}")
print(f"norm_stats=ok path={norm_stats_path}", flush=True)

meta = LeRobotDatasetMetadata(data_factory.repo_id, root=data_factory.lerobot_root)
selected_episode_ids, frames = selected_episodes(meta)
selection_path = norm_stats_path.with_name("task_selection.json")
if selection_path.is_file():
    selection = json.loads(selection_path.read_text())
    if (
        selection["task_name"] != task_name
        or sorted(selection["episode_ids"]) != sorted(selected_episode_ids)
        or selection["total_frames"] != frames
    ):
        raise ValueError(f"Normalization statistics do not match the selected task episodes: {selection_path}")
    print("normalization_task_selection=matched", flush=True)
print(
    "task_filter=ok",
    f"task={task_name!r}",
    f"episodes={len(selected_episode_ids)}",
    f"frames={frames}",
    f"hours={frames / meta.fps / 3600:.2f}",
    flush=True,
)
if mode == "inspect":
    raise SystemExit(0)

original_init = data_loader.lerobot_dataset.LeRobotDataset.__init__


def task_filtered_init(self, repo_id, root=None, episodes=None, *args, **kwargs):
    if episodes is not None:
        raise ValueError("The feasibility launcher expected OpenPI not to set episodes")
    return original_init(self, repo_id, root, selected_episode_ids, *args, **kwargs)


data_loader.lerobot_dataset.LeRobotDataset.__init__ = task_filtered_init

if mode == "train":
    if os.environ.get("CHECKPOINT_DIAGNOSTICS") == "1":
        import runpy

        runpy.run_path(str(Path.cwd().parents[2] / "slurm" / "diagnose_abc_checkpoint.py"))
    module = importlib.import_module("train")
    module.main(config)
else:
    raise ValueError(f"Unknown mode: {mode}")
PY
}

echo "============================================================"
echo "pi0.5 ABC-130K task-specific full fine-tuning"
echo "job_id=${SLURM_JOB_ID:-not-in-slurm}"
echo "task_name=${TASK_NAME}"
echo "dataset_root=${DATASET_ROOT}"
echo "norm_stats_path=${TASK_NORM_STATS_PATH}"
echo "learning_rate=${LEARNING_RATE}"
echo "warmup_steps=${WARMUP_STEPS}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "num_train_steps=${NUM_TRAIN_STEPS}"
echo "fsdp_devices=${FSDP_DEVICES}"
echo "lerobot_decoder_cache_size=${LEROBOT_DECODER_CACHE_SIZE}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "save_interval=${SAVE_INTERVAL}"
echo "keep_period=${KEEP_PERIOD}"
echo "max_to_keep=${MAX_TO_KEEP}"
echo "dry_run=${DRY_RUN}"
echo "============================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    run_openpi_task inspect
    echo "Task and normalization statistics inspection completed; training was not started."
    exit 0
fi

bash "${REPO_ROOT}/slurm/prepare_abc_pi05.bash" \
    "${REPO_ROOT}" "${DATASET_ROOT}" "${FSDP_DEVICES}" "${TASK_NORM_STATS_PATH}"

echo "Starting task-specific training from the pi0.5 base checkpoint."
run_openpi_task train

echo "Training completed successfully."
echo "checkpoint_dir=${CHECKPOINT_DIR}"
