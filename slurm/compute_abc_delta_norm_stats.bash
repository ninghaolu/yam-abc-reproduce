#!/usr/bin/env bash

#SBATCH --job-name=abc-delta-stats
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
#SBATCH --account=torch_pr_147_courant
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc130k/stats-%A_%a.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc130k/stats-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --export=ALL

set -euo pipefail

REPO_ROOT="/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/scripts/compute_abc_delta_norm_stats.py"
DATASET_ROOT="${ABC_STATS_DATASET_ROOT:-/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_train}"
OUTPUT_DIR="${ABC_STATS_OUTPUT_DIR:-${REPO_ROOT}/third_party/policy/openpi/assets/pi05_abc130k/abc130k_yam}"
WORK_DIR="${ABC_STATS_WORK_DIR:-/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc130k/global-delta-norm-stats}"
STAGE="${ABC_STATS_STAGE:?ABC_STATS_STAGE is required}"
NUM_SHARDS="${ABC_STATS_NUM_SHARDS:?ABC_STATS_NUM_SHARDS is required}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ARGS=(
    --stage "${STAGE}"
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --work-dir "${WORK_DIR}"
    --num-shards "${NUM_SHARDS}"
    --workers "${SLURM_CPUS_PER_TASK:-1}"
)
if [[ "${ABC_STATS_FORCE:-0}" == "1" ]]; then
    ARGS+=(--force)
fi

echo "stage=${STAGE} job=${SLURM_JOB_ID:-manual} array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "dataset_root=${DATASET_ROOT}"
echo "work_dir=${WORK_DIR}"
echo "output_dir=${OUTPUT_DIR}"

exec "${PYTHON}" "${SCRIPT}" "${ARGS[@]}"
