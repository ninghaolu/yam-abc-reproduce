#!/usr/bin/env bash

#SBATCH --job-name=pi05_abc_put_lr1e-5
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

# Fine-tune only "put the plastic bottles in the bin" episodes from ABC-130K.
# Use /home/alex/policy-finetuning/norm_stats/norm_stats_abc/
# put_the_plastic_bottles_in_the_bin/norm_stats.json; do not recompute statistics.
# Warm up for 1,000 optimizer steps, then hold the learning rate at 1e-5.
# Default global batch: 256 across four H100 80 GB GPUs.
#
# Before submission:
#   module load slurm/slurm/23.02.8
#   mkdir -p /home/alex/policy-finetuning/policy-finetuning-logs/pi05-abc-task
#   Export WANDB_API_KEY or use the repo's .secrets/wandb_alex_api_key.
#
# Submit: sbatch slurm/finetune_abc_pi05_yam_lr_1e-5.bash
# Inspect task/statistics only: DRY_RUN=1 bash slurm/finetune_abc_pi05_yam_lr_1e-5.bash
# Resume: RUN_MODE=resume EXP_NAME=<original-name> sbatch slurm/finetune_abc_pi05_yam_lr_1e-5.bash
# TASK_NAME / TASK_NORM_STATS_PATH and the usual training overrides are
# handled by the shared task launcher.

set -euo pipefail

REPO_ROOT="/home/alex/policy-finetuning/yam-abc-reproduce"
export LEARNING_RATE="1e-5"
export WARMUP_STEPS=1000
exec bash "${REPO_ROOT}/slurm/finetune_abc_pi05_task.bash"
