#!/usr/bin/env bash
#SBATCH --job-name=put_val_loss
#SBATCH --partition=defq
#SBATCH --exclude=cld2-bom-comp004
#SBATCH --chdir=/home/alex/policy-finetuning/yam-abc-reproduce
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=08:00:00
#SBATCH --output=/home/alex/policy-finetuning/policy-finetuning-logs/validation-put-%A_%a.out
#SBATCH --error=/home/alex/policy-finetuning/policy-finetuning-logs/validation-put-%A_%a.err
#SBATCH --export=ALL

# Uses the same project venv/FFmpeg activation as all-data training via the
# single-checkpoint launcher. defq advertises generic GPUs (the cluster has H100s).
# Prepare a fixed checkpoint snapshot first, from the repository root:
# python3 scripts/validation_sweep.py prepare \
#   --output-dir outputs/pi05_all_data_lr1e-5_91_bottles_8k \
#   --checkpoint-run /projects/data/datasets/checkpoints/pi05_abc130k/pi05_abc130k_all_data_lr1e-5_91 \
#   --interval 8000
# Smoke: sbatch --array=0 slurm/eval_put_validation_sweep.bash <output-dir> --max-batches 2
# Full:  sbatch --array=0-10%1 slurm/eval_put_validation_sweep.bash <output-dir>
# Set the last array index to the manifest checkpoint count minus one.
set -euo pipefail
OUTPUT_DIR="${1:?Pass the directory containing manifest.json}"
shift
exec /usr/bin/python3 /home/alex/policy-finetuning/yam-abc-reproduce/scripts/validation_sweep.py \
    run --output-dir "${OUTPUT_DIR}" "$@"
