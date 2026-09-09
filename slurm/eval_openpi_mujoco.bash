#!/usr/bin/env bash

#SBATCH --job-name=pi05_mujoco
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
#SBATCH --account=torch_pr_147_courant
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-mujoco/%x-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-mujoco/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:l40s:1
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00

set -euo pipefail
umask 027

# This launcher deliberately uses two isolated Python environments on one GPU:
#   1. yam-abc-reproduce/.venv serves the OpenPI/JAX checkpoint;
#   2. ../abc/.venv runs the validated ABC MuJoCo-Warp environment.
# JAX preallocation is disabled so the simulator can share the allocated GPU.
#
# Two-chunk integration smoke (includes a short three-camera video):
#   NUM_CHUNKS=2 sbatch slurm/eval_openpi_mujoco.bash
#
# One complete randomized rollout with video (the defaults):
#   sbatch slurm/eval_openpi_mujoco.bash
#
# Upstream-scale comparison, 20 randomized worlds with a video for every world:
#   NUM_WORLDS=20 sbatch slurm/eval_openpi_mujoco.bash

WORKSPACE_ROOT=/projects/work/yang-lab/projects/policy-finetuning
REPO_ROOT="$WORKSPACE_ROOT/yam-abc-reproduce"
OPENPI_ROOT="$REPO_ROOT/third_party/policy/openpi"
ABC_ROOT="$WORKSPACE_ROOT/abc"
OPENPI_PYTHON="$REPO_ROOT/.venv/bin/python"
SIM_PYTHON="$ABC_ROOT/.venv/bin/python"
SERVER_SCRIPT="$REPO_ROOT/yam_abc_reproduce/deploy/servers/openpi_server.py"
EVAL_SCRIPT="$REPO_ROOT/scripts/eval_openpi_mujoco.py"
OPENPI_CLIENT_SRC="$OPENPI_ROOT/packages/openpi-client/src"

CHECKPOINT="${CHECKPOINT:-$OPENPI_ROOT/checkpoints/pi05_abc130k_task/pi05_abc_throw_the_plastic_bottles_in_the_bin_16923749/8000}"
# The task launcher changed only the transient training run name. Architecture,
# transforms, camera mapping, and asset ID remain those of this registered base config.
CONFIG_NAME="${CONFIG_NAME:-pi05_abc130k}"
# Match this task-specific checkpoint's training prompt.
PROMPT="${PROMPT:-throw the plastic bottles in the bin}"
NUM_WORLDS="${NUM_WORLDS:-1}"
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
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/pi05_mujoco/${CHECKPOINT_RUN}/step-${CHECKPOINT_STEP}/job-${SLURM_JOB_ID:-manual}}"

JOB_NUMBER="${SLURM_JOB_ID:-1}"
if [[ ! "$JOB_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SLURM_JOB_ID must be numeric (got $JOB_NUMBER)" >&2
    exit 2
fi
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-$((18000 + JOB_NUMBER % 10000))}"

CACHE_ROOT="${CACHE_ROOT:-$WORKSPACE_ROOT/openpi-cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_ROOT/openpi-assets}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$CACHE_ROOT/jax-compilation-cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$ABC_ROOT/cache/warp}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$SERVER_HOST"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,$SERVER_HOST"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

[[ -x "$OPENPI_PYTHON" ]] || { echo "ERROR: missing OpenPI Python: $OPENPI_PYTHON" >&2; exit 2; }
[[ -x "$SIM_PYTHON" ]] || { echo "ERROR: missing simulator Python: $SIM_PYTHON" >&2; exit 2; }
[[ -f "$CHECKPOINT/_CHECKPOINT_METADATA" ]] || {
    echo "ERROR: incomplete/missing checkpoint: $CHECKPOINT" >&2
    exit 2
}
[[ -s "$CHECKPOINT/params/manifest.ocdbt" ]] || {
    echo "ERROR: checkpoint params manifest is missing: $CHECKPOINT/params/manifest.ocdbt" >&2
    exit 2
}
[[ -s "$CHECKPOINT/assets/abc130k_yam/norm_stats.json" ]] || {
    echo "ERROR: checkpoint normalization stats are missing" >&2
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
"$OPENPI_PYTHON" "$SERVER_SCRIPT" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --config "$CONFIG_NAME" \
    --checkpoint "$CHECKPOINT" \
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
