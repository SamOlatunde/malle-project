#!/bin/bash
# tacc/slurm/03b_embed_dinov3.slurm
#
# Embeds config.INDEX_DIR/QUERY_DIR with a DINOv3 backbone via Hugging Face
# Transformers, using the pooled ([CLS]-style) output as the embedding.
#
# Prereq: run once on a login node (compute nodes generally lack outbound
# internet) —
#   1. Request access + accept the license:
#      https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
#      and accept the license on the matching huggingface.co model page.
#   2. export HF_TOKEN=<your HF access token>
#   3. python tacc/fetch_checkpoints.py --dinov3 \
#        --dinov3-model facebook/dinov3-vitb16-pretrain-lvd1689m

#SBATCH -J malle-embed-dinov3
#SBATCH -o logs/embed_dinov3_%j.out
#SBATCH -e logs/embed_dinov3_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
source tacc/env.sh
mkdir -p logs

export MALLE_MODEL_FAMILY=dinov3
export MALLE_DATASET_VARIANT="${MALLE_DATASET_VARIANT:-x1}"
export MALLE_DINOV3_MODEL="${MALLE_DINOV3_MODEL:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
# HF_HOME is set by tacc/env.sh to the cache tacc/fetch_checkpoints.py
# populated on the login node — this job should not need to hit the network.

malle_run python "$MALLE_REPO/embed.py"
