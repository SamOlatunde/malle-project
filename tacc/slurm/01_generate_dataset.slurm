#!/bin/bash
# tacc/slurm/01_generate_dataset.slurm
#
# Generates malle_dataset/modified_images_x<N>/ combinatorial variants on
# Vista instead of your laptop. This is CPU-bound (PIL/torchvision image
# transforms, no GPU used) — runs on Vista's Grace-Grace (CPU-only) nodes,
# saving GPU allocation hours for the embedding jobs.
#
# One array task per modification count (x1, x2, x3, ...). Recall the full
# x1..x11 sweep is 2**11 - 1 = 2047 combinations/image; with 200 originals
# that's 409,400 output images for x1..x11 combined. Start with a small
# --array range (e.g. 1-4) and check output size/runtime before going wider.
#
# Prereqs:
#   - malle_dataset/original_images/ already present under $MALLE_REPO
#     (rsync/scp it from your laptop — it's ~a few hundred KB-MB, no need to
#     regenerate from imagenet-mini on Vista).
#   - Container pulled per tacc/env.sh comments.

#SBATCH -J malle-gen-dataset
#SBATCH -o logs/gen_%A_%a.out
#SBATCH -e logs/gen_%A_%a.err
#SBATCH -p gg                      # CPU-only Grace-Grace queue — verify with `sinfo`; fall back to `gh` (without --nv) if unavailable under your allocation
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --array=1-4                # one task per modification count

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
source tacc/env.sh
mkdir -p logs

export MALLE_MOD_COUNTS="$SLURM_ARRAY_TASK_ID"

malle_run python "$MALLE_REPO/generate_combination_modified.py"
