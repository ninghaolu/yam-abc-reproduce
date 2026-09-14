#!/usr/bin/env bash

#SBATCH --job-name=pi05_abc130k_all_lr1e-4
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

# All 129,225 ABC-130K episodes with full-dataset normalization statistics.
# Global batch 256 on four H100 80 GB GPUs; THP disabled for training/workers.
# Warm up for 1,000 steps, then hold LR at 1e-4; default 200,000 updates.
#
# Before submission:
#   module load slurm/slurm/23.02.8
#   mkdir -p /home/alex/policy-finetuning/policy-finetuning-logs/pi05-abc130k-all-data
#   Export WANDB_API_KEY or use the repo's .secrets/wandb_alex_api_key.
# Submit: sbatch slurm/finetune_abc_pi05_all_data_lr_1e-4.bash
# Inspect: DRY_RUN=1 bash slurm/finetune_abc_pi05_all_data_lr_1e-4.bash
# Resume: RUN_MODE=resume EXP_NAME=<original-name> sbatch slurm/finetune_abc_pi05_all_data_lr_1e-4.bash

set -euo pipefail
REPO_ROOT="/home/alex/policy-finetuning/yam-abc-reproduce"
export LEARNING_RATE="1e-4"
export WARMUP_STEPS=1000
exec bash "${REPO_ROOT}/slurm/finetune_abc_pi05_all_data.bash"
