#!/usr/bin/env bash
#SBATCH --job-name=abc_ckpt_diag
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce/third_party/policy/openpi
#SBATCH --account=torch_pr_147_courant
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc-task/slurm-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/pi05-abc-task/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500GB
#SBATCH --time=02:00:00
#SBATCH --export=ALL

set -euo pipefail
REPO_ROOT=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
export EXP_NAME="pi05_abc_ckpt_diag_${SLURM_JOB_ID}"
export RUN_MODE=fresh
# The training loop labels steps from zero: 11 iterations trigger the step-10 save.
export NUM_TRAIN_STEPS=11 SAVE_INTERVAL=10 KEEP_PERIOD=10 LOG_INTERVAL=1
export GLOBAL_BATCH_SIZE=256 NUM_WORKERS=2 FSDP_DEVICES=1
export CHECKPOINT_DIAGNOSTICS=1
MONITOR_LOG="${REPO_ROOT}/../policy-finetuning-logs/pi05-abc-task/ckpt-diag-${SLURM_JOB_ID}.monitor.log"
monitor() {
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
        free -m
        ps -u "$(id -u)" -o pid,ppid,stat,pcpu,rss,comm --sort=-rss | head -n 22 || true
        sleep 30
    done
}
monitor > "${MONITOR_LOG}" 2>&1 &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
echo "Checkpoint diagnostic: step 10, batch=${GLOBAL_BATCH_SIZE}, workers=${NUM_WORKERS}; monitor=${MONITOR_LOG}"
bash "${REPO_ROOT}/slurm/finetune_abc_pi05_task.bash"
