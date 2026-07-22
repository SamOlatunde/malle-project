#!/bin/bash
# tacc/slurm/02_embed_resnet50.slurm
#
# Baseline ResNet50 embedding run — same model.py/embed.py path you use
# locally, just moved onto a Vista GPU node. Embeds config.INDEX_DIR and
# config.QUERY_DIR (MALLE_DATASET_VARIANT selects which modified_images_x<N>
# folder) and writes embeddings/*.npy, matching your local pipeline exactly.

#SBATCH -J malle-embed-resnet50
#SBATCH -o logs/embed_resnet50_%j.out
#SBATCH -e logs/embed_resnet50_%j.err
#SBATCH -p gh                      # GPU queue (GH200 / H200) — verify with `sinfo`
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
source tacc/env.sh
mkdir -p logs

export MALLE_MODEL_FAMILY=resnet50
export MALLE_DATASET_VARIANT="${MALLE_DATASET_VARIANT:-x1}"

malle_run python "$MALLE_REPO/embed.py"
