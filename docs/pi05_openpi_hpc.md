# Launch π0.5 OpenPI training on the cluster

This note is for direct π0.5 training from this checkout on a Slurm GPU node. It uses the
training-only environment at the repository root and the OpenPI source vendored under
`third_party/policy/openpi`.

## Paths used by the experiment

```bash
REPO_DIR=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
OPENPI_DIR="$REPO_DIR/third_party/policy/openpi"
PYTHON="$REPO_DIR/.venv/bin/python"
CACHE_ROOT=/projects/work/yang-lab/projects/policy-finetuning/openpi-cache
```

- Python environment: `$REPO_DIR/.venv`
- LeRobot datasets: `$REPO_DIR/data/lerobot/<repo-id>`
- Normalization statistics: `$OPENPI_DIR/assets/<config>/yam/norm_stats.json`
- Checkpoints: `$OPENPI_DIR/checkpoints/<config>/<experiment-name>/<step>`
- Shared downloads and logs: `$CACHE_ROOT`

Run OpenPI commands from `$OPENPI_DIR`. Its `assets` and `checkpoints` defaults are relative
to the current working directory. Use `$PYTHON` directly; running `uv run` from the repository
root would resolve the root project rather than this training-only environment.

## 1. Check the dataset

Choose the converted LeRobot dataset and arm layout. The default bimanual configuration has
two arms and 14 action dimensions; a single-arm dataset uses one arm and 7 action dimensions.

```bash
cd "$REPO_DIR"

DATASET_ID=put_the_bottle_into_the_bin
NUM_ARMS=2

test -d "data/lerobot/$DATASET_ID"
find "data/lerobot/$DATASET_ID" -maxdepth 2 -type f | head
```

If the first command fails, convert or copy the dataset into `data/lerobot/$DATASET_ID`
before submitting training. The same `DATASET_ID` and `NUM_ARMS` must be passed to both
normalization and training.

## 2. Set the shared caches

Use these exports in an interactive allocation or in every Slurm job:

```bash
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_LEROBOT_HOME="$REPO_DIR/data/lerobot"
export OPENPI_DATA_HOME="$CACHE_ROOT/openpi-assets"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb-cache"
export WANDB_DIR="$CACHE_ROOT/wandb-runs"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1

# Let GCS downloads honor any proxy variables inherited by the job.
export FSSPEC_GS='{"session_kwargs": {"trust_env": true}}'
```

The first training run downloads the π0.5 base parameters from
`gs://openpi-assets/checkpoints/pi05_base/params` into `$OPENPI_DATA_HOME`. Later jobs reuse
the cached copy.

## 3. Compute normalization statistics

Choose full fine-tuning (`pi05_yam`) or LoRA (`pi05_yam_lora`):

```bash
CONFIG_NAME=pi05_yam
# CONFIG_NAME=pi05_yam_lora

cd "$OPENPI_DIR"
"$PYTHON" scripts/compute_norm_stats.py \
    --config-name "$CONFIG_NAME" \
    --repo-id "$DATASET_ID" \
    --num-arms "$NUM_ARMS"
```

Expected output:

```text
assets/pi05_yam/yam/norm_stats.json
```

The YAM configs deliberately use the fixed asset ID `yam`. If the dataset changes, recompute
the statistics even when a file already exists; otherwise the old dataset's statistics will
be reused.

## 4. Launch manually in an interactive GPU allocation

This example performs full fine-tuning on one visible GPU, disables W&B, and overwrites an
existing checkpoint directory with the same experiment name:

```bash
cd "$OPENPI_DIR"

EXP_NAME=pi05_yam_smoke

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
"$PYTHON" scripts/train.py "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    --data.repo-id "$DATASET_ID" \
    --data.num-arms "$NUM_ARMS" \
    --batch-size 8 \
    --fsdp-devices 1 \
    --num-train-steps 100 \
    --save-interval 100 \
    --log-interval 10 \
    --no-wandb-enabled \
    --overwrite
```

For a real run, increase `--num-train-steps` (the config default is 30,000), choose a unique
experiment name, and tune the batch size for the allocated GPU. The global batch size must be
divisible by the number of visible JAX devices.

## 5. Submit a Slurm experiment

Create the log directory before `sbatch`; Slurm opens the output files before the job script
runs:

```bash
mkdir -p /projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-yam
```

Save the following as a job script, for example `pi05_yam_finetune.sbatch`. Check the account,
reservation, GPU type/count, memory, and time limit against the allocation being used.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=pi05_yam
#SBATCH --account=torch_pr_147_courant
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-yam/slurm-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-yam/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400GB
#SBATCH --time=48:00:00
#SBATCH --export=ALL
# Add the current reservation only when it is required and still active:
##SBATCH --reservation=my3149-h200

set -euo pipefail

REPO_DIR=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
OPENPI_DIR="$REPO_DIR/third_party/policy/openpi"
PYTHON="$REPO_DIR/.venv/bin/python"
CACHE_ROOT=/projects/work/yang-lab/projects/policy-finetuning/openpi-cache

CONFIG_NAME="${CONFIG_NAME:-pi05_yam}"
DATASET_ID="${DATASET_ID:-put_the_bottle_into_the_bin}"
NUM_ARMS="${NUM_ARMS:-2}"
EXP_NAME="${EXP_NAME:-pi05_yam_${SLURM_JOB_ID:-manual}}"
RUN_MODE="${RUN_MODE:-fresh}"
RECOMPUTE_NORM_STATS="${RECOMPUTE_NORM_STATS:-0}"

FSDP_DEVICES="${FSDP_DEVICES:-2}"
BATCH_PER_GPU="${BATCH_PER_GPU:-16}"
GLOBAL_BATCH=$((BATCH_PER_GPU * FSDP_DEVICES))
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SEED="${SEED:-0}"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_LEROBOT_HOME="$REPO_DIR/data/lerobot"
export HF_HUB_OFFLINE=1
export OPENPI_DATA_HOME="$CACHE_ROOT/openpi-assets"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb-cache"
export WANDB_DIR="$CACHE_ROOT/wandb-runs"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONUNBUFFERED=1
export FSSPEC_GS='{"session_kwargs": {"trust_env": true}}'
export PATH="$REPO_DIR/.venv/bin:$PATH"

mkdir -p \
    "$HF_LEROBOT_HOME" \
    "$OPENPI_DATA_HOME" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_DIR"

case "$RUN_MODE" in
    fresh)     TRAIN_STATE_ARGS=() ;;
    resume)    TRAIN_STATE_ARGS=(--resume) ;;
    overwrite) TRAIN_STATE_ARGS=(--overwrite) ;;
    *) echo "RUN_MODE must be fresh, resume, or overwrite" >&2; exit 2 ;;
esac

if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
    WANDB_ARGS=(--project-name "${WANDB_PROJECT:-policy-finetuning}")
    export WANDB_ENTITY="${WANDB_ENTITY:-oneworld-ai}"
else
    WANDB_ARGS=(--no-wandb-enabled)
fi

DATASET_PATH="$HF_LEROBOT_HOME/$DATASET_ID"
NORM_STATS_PATH="$OPENPI_DIR/assets/$CONFIG_NAME/yam/norm_stats.json"
CHECKPOINT_DIR="$OPENPI_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME"

[[ -x "$PYTHON" ]] || { echo "Missing environment: $PYTHON" >&2; exit 2; }
[[ -d "$DATASET_PATH" ]] || { echo "Missing dataset: $DATASET_PATH" >&2; exit 2; }

cd "$OPENPI_DIR"

echo "host=$(hostname)"
echo "config=$CONFIG_NAME dataset=$DATASET_ID arms=$NUM_ARMS"
echo "experiment=$EXP_NAME mode=$RUN_MODE checkpoint=$CHECKPOINT_DIR"
echo "fsdp_devices=$FSDP_DEVICES global_batch=$GLOBAL_BATCH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PYTHON" -c 'import jax; print("jax_devices=", jax.devices())'

if [[ "$RECOMPUTE_NORM_STATS" == "1" || ! -f "$NORM_STATS_PATH" ]]; then
    "$PYTHON" scripts/compute_norm_stats.py \
        --config-name "$CONFIG_NAME" \
        --repo-id "$DATASET_ID" \
        --num-arms "$NUM_ARMS"
else
    echo "Reusing normalization statistics: $NORM_STATS_PATH"
fi

[[ -f "$NORM_STATS_PATH" ]] || { echo "Missing norm stats: $NORM_STATS_PATH" >&2; exit 3; }

"$PYTHON" scripts/train.py "$CONFIG_NAME" \
    --exp-name "$EXP_NAME" \
    --data.repo-id "$DATASET_ID" \
    --data.num-arms "$NUM_ARMS" \
    --batch-size "$GLOBAL_BATCH" \
    --fsdp-devices "$FSDP_DEVICES" \
    --num-train-steps "$NUM_TRAIN_STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --seed "$SEED" \
    "${WANDB_ARGS[@]}" \
    "${TRAIN_STATE_ARGS[@]}"
```

Submit a new run:

```bash
sbatch pi05_yam_finetune.sbatch
```

Useful overrides:

```bash
# LoRA instead of full fine-tuning.
CONFIG_NAME=pi05_yam_lora EXP_NAME=pi05_yam_lora_01 \
    sbatch pi05_yam_finetune.sbatch

# Recompute statistics after changing the dataset.
RECOMPUTE_NORM_STATS=1 DATASET_ID=my_new_dataset \
    sbatch pi05_yam_finetune.sbatch

# Resume the most recent checkpoint for an existing experiment.
EXP_NAME=pi05_yam_12345678 RUN_MODE=resume \
    sbatch pi05_yam_finetune.sbatch

# Enable W&B after logging in once with `.venv/bin/wandb login --relogin`.
WANDB_ENABLED=1 WANDB_ENTITY=oneworld-ai WANDB_PROJECT=policy-finetuning \
    sbatch pi05_yam_finetune.sbatch
```
