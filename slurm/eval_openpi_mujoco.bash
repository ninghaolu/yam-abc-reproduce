#!/usr/bin/env bash

#SBATCH --job-name=pi05_mujoco
#SBATCH --chdir=/home/alex/policy-finetuning/yam-abc-reproduce
#SBATCH --partition=defq
#SBATCH --exclude=cld2-bom-comp004
#SBATCH --output=/home/alex/policy-finetuning/policy-finetuning-logs/pi05-mujoco/%x-%j.out
#SBATCH --error=/home/alex/policy-finetuning/policy-finetuning-logs/pi05-mujoco/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00

set -euo pipefail
umask 027

# Two processes share one GPU and default to the project .venv:
#   1. the OpenPI/JAX checkpoint server;
#   2. the ABC MuJoCo-Warp simulator (SIM_PYTHON can override its interpreter).
# Required simulation additions: mujoco-warp==3.10.0.3 warp-lang==1.15.0.
# Before submitting, create the Slurm log directory:
#   mkdir -p /home/alex/policy-finetuning/policy-finetuning-logs/pi05-mujoco
# JAX preallocation is disabled so the simulator can share the allocated GPU.
#
# Two-chunk integration smoke (includes a short three-camera video):
#   NUM_WORLDS=1 NUM_CHUNKS=2 sbatch slurm/eval_openpi_mujoco.bash
#
# One complete randomized rollout with video:
#   NUM_WORLDS=1 sbatch slurm/eval_openpi_mujoco.bash
#
# Default: 20 randomized worlds with a video for every world:
#   NUM_WORLDS=20 sbatch slurm/eval_openpi_mujoco.bash
# Reuse a validation sweep's frozen checkpoint manifest (one checkpoint per task):
#   CHECKPOINT_MANIFEST=/absolute/path/to/manifest.json \
#     sbatch --array=0-10%11 slurm/eval_openpi_mujoco.bash

WORKSPACE_ROOT=/home/alex/policy-finetuning
REPO_ROOT="$WORKSPACE_ROOT/yam-abc-reproduce"
OPENPI_ROOT="$REPO_ROOT/third_party/policy/openpi"
ABC_ROOT="${ABC_ROOT:-$WORKSPACE_ROOT/abc}"
OPENPI_PYTHON="$REPO_ROOT/.venv/bin/python"
SIM_PYTHON="${SIM_PYTHON:-$OPENPI_PYTHON}"
source "$REPO_ROOT/.venv/bin/activate"
export ABC_ROOT
SERVER_SCRIPT="$REPO_ROOT/yam_abc_reproduce/deploy/servers/openpi_server.py"
EVAL_SCRIPT="$REPO_ROOT/scripts/eval_openpi_mujoco.py"
OPENPI_CLIENT_SRC="$OPENPI_ROOT/packages/openpi-client/src"

CHECKPOINT="${CHECKPOINT:-/projects/data/datasets/checkpoints/pi05_abc130k/pi05_abc130k_all_data_lr1e-5_91/8000}"
if [[ -n "${CHECKPOINT_MANIFEST:-}" ]]; then
    CHECKPOINT="$("$OPENPI_PYTHON" - "$CHECKPOINT_MANIFEST" "${SLURM_ARRAY_TASK_ID:?Use --array with CHECKPOINT_MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    entries = json.load(stream)["checkpoints"]
index = int(sys.argv[2])
if not 0 <= index < len(entries):
    raise ValueError(f"Checkpoint index {index} is out of range")
entry = entries[index]
if entry["group"] == "abc_dit":
    raise ValueError("This launcher requires an OpenPI checkpoint")
print(entry["checkpoint"])
PY
)"
fi
CONFIG_NAME="${CONFIG_NAME:-pi05_abc130k}"
# Prefer an explicit statistics path/asset; otherwise discover the saved stats.
# All-data checkpoints normally contain assets/abc130k_yam/norm_stats.json.
if [[ -z "${NORM_STATS_PATH:-}" ]]; then
    if [[ -n "${NORM_STATS_ASSET_ID:-}" ]]; then
        NORM_STATS_PATH="$CHECKPOINT/assets/$NORM_STATS_ASSET_ID/norm_stats.json"
    else
        shopt -s nullglob
        STATS_PATHS=("$CHECKPOINT"/assets/*/norm_stats.json)
        shopt -u nullglob
        if (( ${#STATS_PATHS[@]} != 1 )); then
            echo "ERROR: expected one checkpoint norm_stats.json; set NORM_STATS_PATH explicitly." >&2
            exit 2
        fi
        NORM_STATS_PATH="${STATS_PATHS[0]}"
    fi
fi
# Evaluate the bottles scene with its matching task prompt.
PROMPT="${PROMPT:-put the plastic bottles in the bin}"
NUM_WORLDS="${NUM_WORLDS:-20}"
NUM_CHUNKS="${NUM_CHUNKS:-120}"
EXECUTE_CHUNK_DIM="${EXECUTE_CHUNK_DIM:-15}"
SCENE_SEED="${SCENE_SEED:-20260511}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
# Print per-chunk gripper commands/positions and save per-action CSV traces.
GRIPPER_DIAGNOSTICS="${GRIPPER_DIAGNOSTICS:-1}"
VANILLA_PHYSICS="${VANILLA_PHYSICS:-0}"
VIDEO_EVERY_N_ACTIONS="${VIDEO_EVERY_N_ACTIONS:-1}"

CHECKPOINT_STEP="${CHECKPOINT##*/}"
CHECKPOINT_PARENT="${CHECKPOINT%/*}"
CHECKPOINT_RUN="${CHECKPOINT_PARENT##*/}"
MUJOCO_OUTPUT_ROOT="${MUJOCO_OUTPUT_ROOT:-/projects/data/cloud_user/mujoco_eval_outputs/pi05-all-data}"
OUTPUT_DIR="${OUTPUT_DIR:-$MUJOCO_OUTPUT_ROOT/${CHECKPOINT_RUN}/step-${CHECKPOINT_STEP}/job-${SLURM_JOB_ID:-manual}}"

JOB_NUMBER="${SLURM_JOB_ID:-1}"
if [[ ! "$JOB_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SLURM_JOB_ID must be numeric (got $JOB_NUMBER)" >&2
    exit 2
fi
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-$((18000 + JOB_NUMBER % 10000))}"

CACHE_ROOT="${CACHE_ROOT:-$WORKSPACE_ROOT/openpi-cache}"
export HF_HOME="${EVAL_HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_ROOT/openpi-assets}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$CACHE_ROOT/jax-compilation-cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$REPO_ROOT/outputs/cache/warp}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$SERVER_HOST"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,$SERVER_HOST"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

[[ -x "$OPENPI_PYTHON" ]] || { echo "ERROR: missing OpenPI Python: $OPENPI_PYTHON" >&2; exit 2; }
[[ -x "$SIM_PYTHON" ]] || { echo "ERROR: missing simulator Python: $SIM_PYTHON" >&2; exit 2; }
[[ -f "$CHECKPOINT/params/_METADATA" ]] || {
    echo "ERROR: incomplete/missing checkpoint: $CHECKPOINT" >&2
    exit 2
}
[[ -s "$CHECKPOINT/params/manifest.ocdbt" ]] || {
    echo "ERROR: checkpoint params manifest is missing: $CHECKPOINT/params/manifest.ocdbt" >&2
    exit 2
}
[[ -s "$NORM_STATS_PATH" ]] || {
    echo "ERROR: checkpoint normalization stats are missing: $NORM_STATS_PATH" >&2
    exit 2
}

mkdir -p "$OUTPUT_DIR" "$JAX_COMPILATION_CACHE_DIR" "$WARP_CACHE_PATH"
chmod 750 "$OUTPUT_DIR"
SERVER_STDOUT="$OUTPUT_DIR/server.out"
SERVER_STDERR="$OUTPUT_DIR/server.err"

echo "============================================================"
echo "pi0.5 checkpoint in ABC MuJoCo"
echo "job_id=${SLURM_JOB_ID:-not-in-slurm} host=$(hostname)"
echo "checkpoint=$CHECKPOINT"
echo "config_name=$CONFIG_NAME"
echo "norm_stats_path=$NORM_STATS_PATH"
echo "prompt=$PROMPT"
echo "server=$SERVER_HOST:$SERVER_PORT"
echo "output_dir=$OUTPUT_DIR"
echo "num_worlds=$NUM_WORLDS num_chunks=$NUM_CHUNKS execute_chunk_dim=$EXECUTE_CHUNK_DIM"
echo "save_video=$SAVE_VIDEO video_every_n_actions=$VIDEO_EVERY_N_ACTIONS"
echo "gripper_diagnostics=$GRIPPER_DIAGNOSTICS"
echo "vanilla_physics=$VANILLA_PHYSICS scene_seed=$SCENE_SEED"
echo "xla_preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE"
echo "============================================================"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
OPENPI_EVAL_CONFIG_NAME="$CONFIG_NAME" "$OPENPI_PYTHON" -c 'import os; import jax; from openpi.training import config; cfg = config.get_config(os.environ["OPENPI_EVAL_CONFIG_NAME"]); print(f"jax={jax.__version__} model={cfg.model.model_type} action_horizon={cfg.model.action_horizon} action_dim={cfg.model.action_dim}")'
PYTHONPATH="$OPENPI_CLIENT_SRC${PYTHONPATH:+:$PYTHONPATH}" \
    "$SIM_PYTHON" -c 'import msgpack, mujoco, mujoco_warp, openpi_client, warp; print(f"mujoco={mujoco.__version__} warp={warp.__version__} msgpack={msgpack.version}")'

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

cd "$OPENPI_ROOT"
"$OPENPI_PYTHON" "$REPO_ROOT/slurm/exec_without_thp.py" "$OPENPI_PYTHON" "$SERVER_SCRIPT" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --config "$CONFIG_NAME" \
    --checkpoint "$CHECKPOINT" \
    --norm-stats-path "$NORM_STATS_PATH" \
    --prompt "$PROMPT" \
    --image-key-map "top=image,left=left_wrist,right=right_wrist" \
    --flatten-prefix "observation/" \
    --state-key "observation/state" \
    >"$SERVER_STDOUT" 2>"$SERVER_STDERR" &
SERVER_PID=$!

echo "waiting for OpenPI server pid=$SERVER_PID (logs: $SERVER_STDOUT, $SERVER_STDERR)"
SERVER_DEADLINE=$((SECONDS + 1200))
until curl --noproxy '*' -fsS "http://$SERVER_HOST:$SERVER_PORT/healthz" >/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: OpenPI server exited during startup" >&2
        tail -n 120 "$SERVER_STDERR" >&2 || true
        exit 3
    fi
    if (( SECONDS >= SERVER_DEADLINE )); then
        echo "ERROR: timed out waiting 20 minutes for OpenPI server" >&2
        tail -n 120 "$SERVER_STDERR" >&2 || true
        exit 3
    fi
    sleep 5
done
echo "OpenPI server is ready"

ARGS=(
    --server-host "$SERVER_HOST"
    --server-port "$SERVER_PORT"
    --checkpoint "$CHECKPOINT"
    --output-dir "$OUTPUT_DIR"
    --num-worlds "$NUM_WORLDS"
    --num-chunks "$NUM_CHUNKS"
    --execute-chunk-dim "$EXECUTE_CHUNK_DIM"
    --seed "$SCENE_SEED"
    --gpu-id 0
    --prompt "$PROMPT"
    --video-every-n-actions "$VIDEO_EVERY_N_ACTIONS"
)

case "$SAVE_VIDEO" in
    1) ARGS+=(--save-video) ;;
    0) ARGS+=(--no-save-video) ;;
    *) echo "ERROR: SAVE_VIDEO must be 0 or 1 (got $SAVE_VIDEO)" >&2; exit 2 ;;
esac
case "$VANILLA_PHYSICS" in
    1) ARGS+=(--vanilla-physics) ;;
    0) ARGS+=(--no-vanilla-physics) ;;
    *) echo "ERROR: VANILLA_PHYSICS must be 0 or 1 (got $VANILLA_PHYSICS)" >&2; exit 2 ;;
esac

case "$GRIPPER_DIAGNOSTICS" in
    1) ARGS+=(--log-grippers) ;;
    0) ARGS+=(--no-log-grippers) ;;
    *) echo "ERROR: GRIPPER_DIAGNOSTICS must be 0 or 1 (got $GRIPPER_DIAGNOSTICS)" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
PYTHONPATH="$ABC_ROOT:$OPENPI_CLIENT_SRC${PYTHONPATH:+:$PYTHONPATH}" \
    "$SIM_PYTHON" "$EVAL_SCRIPT" "${ARGS[@]}"

echo "pi0.5 MuJoCo evaluation completed successfully"
echo "summary=$OUTPUT_DIR/summary.json"
