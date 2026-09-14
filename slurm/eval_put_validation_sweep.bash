#!/usr/bin/env bash
#SBATCH --job-name=put_val_loss
#SBATCH --account=torch_pr_147_courant
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=08:00:00
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/validation-put-%A_%a.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/validation-put-%A_%a.err
#SBATCH --export=ALL

# Prepare the manifest with scripts/validation_sweep.py prepare first.
# Smoke: sbatch --array=0,7,12 slurm/eval_put_validation_sweep.bash <output-dir> --max-batches 2
# Full:  sbatch --array=0-12%3 slurm/eval_put_validation_sweep.bash <output-dir>
set -euo pipefail
OUTPUT_DIR="${1:?Pass the directory containing manifest.json}"
shift
exec /usr/bin/python3 /projects/work/yang-lab/projects/policy-finetuning/yam-abc-reproduce/scripts/validation_sweep.py \
    run --output-dir "${OUTPUT_DIR}" "$@"
