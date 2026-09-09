#!/usr/bin/env bash
#SBATCH --job-name=pi05_abc_no_thp
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce/third_party/policy/openpi
#SBATCH --account=torch_pr_147_courant
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc-task/slurm-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc-task/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500GB
#SBATCH --time=48:00:00
#SBATCH --export=ALL

set -euo pipefail
REPO_ROOT=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
# Execute the original launcher with THP disabled only for this process tree.
exec /usr/bin/python3 "${REPO_ROOT}/slurm/exec_without_thp.py" \
    /usr/bin/bash "${REPO_ROOT}/slurm/finetune_abc_pi05_task.bash"
