#!/usr/bin/env bash
#SBATCH --job-name=iodiag
#SBATCH --account=torch_pr_147_courant
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/iodiag/slurm-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/iodiag/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=00:25:00
set -uo pipefail

PROJ=/projects/work/yang-lab/projects/policy-finetuning/.iodiag.$SLURM_JOB_ID
VID=/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_train/videos
LOCAL=/state/partition1/iodiag.$SLURM_JOB_ID
mkdir -p "$PROJ" "$LOCAL" 2>/dev/null

echo "host=$(hostname)  date=$(date)"
echo "--- node memory ---"; free -g | head -2
echo
echo "### 1. WRITE 8GB -> /projects (idle)"
dd if=/dev/zero of="$PROJ/a.bin" bs=1M count=8192 conv=fsync 2>&1 | tail -1
echo
echo "### 2. WRITE 8GB -> /state/partition1 (node-local NVMe)"
dd if=/dev/zero of="$LOCAL/a.bin" bs=1M count=8192 conv=fsync 2>&1 | tail -1
echo
echo "### 3. Start 16 random video readers, then re-measure /projects write"
mapfile -t FILES < <(find "$VID" -name '*.mp4' | shuf -n 64)
for i in $(seq 0 15); do
  ( for r in $(seq 1 400); do
      f=${FILES[$((RANDOM % ${#FILES[@]}))]}
      sz=$(stat -c%s "$f"); off=$((RANDOM * (sz/32768) / 32768))
      dd if="$f" of=/dev/null bs=1M count=4 skip=$((off/1048576)) iflag=skip_bytes,count_bytes 2>/dev/null
    done ) &
done
sleep 20
echo "--- /projects write UNDER random-read load:"
dd if=/dev/zero of="$PROJ/b.bin" bs=1M count=8192 conv=fsync 2>&1 | tail -1
echo "--- /state/partition1 write UNDER random-read load:"
dd if=/dev/zero of="$LOCAL/b.bin" bs=1M count=8192 conv=fsync 2>&1 | tail -1
wait
echo
echo "### cleanup"; rm -rf "$PROJ" "$LOCAL"; echo done
