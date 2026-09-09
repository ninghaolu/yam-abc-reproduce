#!/usr/bin/env bash
#SBATCH --job-name=abc-task-stats
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
#SBATCH --account=torch_pr_147_courant
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc-task/task-stats-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc-task/task-stats-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=ALL

# Same default exact task as finetune_abc_pi05_task.bash.
# TASK_NAME='another exact task' sbatch slurm/compute_abc_task_norm_stats.bash
# Outputs: ../norm_stats_abc/<task_slug>/{norm_stats.json,task_selection.json,...}
set -euo pipefail
REPO_ROOT="/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS=(--task-name "${TASK_NAME:-throw the plastic bottles in the bin}" --workers "${SLURM_CPUS_PER_TASK:-1}")
if [[ -n "${ABC_STATS_DATASET_ROOT:-}" ]]; then ARGS+=(--dataset-root "${ABC_STATS_DATASET_ROOT}"); fi
if [[ -n "${ABC_STATS_OUTPUT_DIR:-}" ]]; then ARGS+=(--output-dir "${ABC_STATS_OUTPUT_DIR}"); fi
if [[ -n "${ABC_STATS_WORK_DIR:-}" ]]; then ARGS+=(--work-dir "${ABC_STATS_WORK_DIR}"); fi
exec "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/compute_abc_task_norm_stats.py" "${ARGS[@]}" "$@"
