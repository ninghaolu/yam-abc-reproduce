#!/usr/bin/env bash
# Submit the two map/reduce passes for full ABC delta normalization stats.

set -euo pipefail

REPO_ROOT="/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/scripts/compute_abc_delta_norm_stats.py"
WORKER="${REPO_ROOT}/slurm/compute_abc_delta_norm_stats.bash"
DATASET_ROOT="${ABC_STATS_DATASET_ROOT:-/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_train}"
LOG_DIR="/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc130k"
MAX_CONCURRENT="${ABC_STATS_MAX_CONCURRENT:-32}"

mkdir -p "${LOG_DIR}"

if [[ ! "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ABC_STATS_MAX_CONCURRENT must be positive" >&2
    exit 2
fi
NUM_FILES="$("${PYTHON}" "${SCRIPT}" --stage count-files --dataset-root "${DATASET_ROOT}")"
DEFAULT_SHARDS="${MAX_CONCURRENT}"
if [[ "${DEFAULT_SHARDS}" -gt "${NUM_FILES}" ]]; then
    DEFAULT_SHARDS="${NUM_FILES}"
fi
NUM_SHARDS="${ABC_STATS_NUM_SHARDS:-${DEFAULT_SHARDS}}"
if [[ ! "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ || "${NUM_SHARDS}" -gt "${NUM_FILES}" ]]; then
    echo "ERROR: ABC_STATS_NUM_SHARDS must be in [1, ${NUM_FILES}]" >&2
    exit 2
fi

ARRAY_SPEC="0-$((NUM_SHARDS - 1))%${MAX_CONCURRENT}"
COMMON_EXPORT="ALL,ABC_STATS_DATASET_ROOT=${DATASET_ROOT},ABC_STATS_NUM_SHARDS=${NUM_SHARDS},ABC_STATS_FORCE=${ABC_STATS_FORCE:-0}"
if [[ -n "${ABC_STATS_WORK_DIR:-}" ]]; then
    COMMON_EXPORT+=",ABC_STATS_WORK_DIR=${ABC_STATS_WORK_DIR}"
fi
if [[ -n "${ABC_STATS_OUTPUT_DIR:-}" ]]; then
    COMMON_EXPORT+=",ABC_STATS_OUTPUT_DIR=${ABC_STATS_OUTPUT_DIR}"
fi

PASS1_JOB="$(sbatch --parsable --array="${ARRAY_SPEC}" \
    --export="${COMMON_EXPORT},ABC_STATS_STAGE=pass1" "${WORKER}")"
PASS1_ID="${PASS1_JOB%%;*}"

REDUCE1_JOB="$(sbatch --parsable --dependency="afterok:${PASS1_ID}" \
    --export="${COMMON_EXPORT},ABC_STATS_STAGE=reduce-pass1" "${WORKER}")"
REDUCE1_ID="${REDUCE1_JOB%%;*}"

PASS2_JOB="$(sbatch --parsable --dependency="afterok:${REDUCE1_ID}" \
    --array="${ARRAY_SPEC}" --export="${COMMON_EXPORT},ABC_STATS_STAGE=pass2" "${WORKER}")"
PASS2_ID="${PASS2_JOB%%;*}"

FINAL_JOB="$(sbatch --parsable --dependency="afterok:${PASS2_ID}" \
    --export="${COMMON_EXPORT},ABC_STATS_STAGE=finalize" "${WORKER}")"
FINAL_ID="${FINAL_JOB%%;*}"

echo "Submitted full ABC-130K global delta statistics pipeline."
echo "files=${NUM_FILES} shards=${NUM_SHARDS} max_concurrent=${MAX_CONCURRENT}"
echo "pass1_array=${PASS1_ID}"
echo "reduce_pass1=${REDUCE1_ID}"
echo "pass2_array=${PASS2_ID}"
echo "finalize=${FINAL_ID}"
echo "Monitor: squeue -j ${PASS1_ID},${REDUCE1_ID},${PASS2_ID},${FINAL_ID}"
