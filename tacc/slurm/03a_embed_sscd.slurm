#!/bin/bash
# tacc/slurm/03a_embed_sscd.slurm
#
# Embeds config.INDEX_DIR/QUERY_DIR with the SSCD (Self-Supervised
# Descriptor for Copy Detection) torchscript checkpoint instead of ResNet50.
# SSCD is purpose-built for exactly this task — image copy/duplicate
# detection under transformations — so it's worth comparing directly against
# the ResNet50 baseline's Recall@k.
#
# Prereq: run once on a login node —
#   python tacc/fetch_checkpoints.py --sscd
# (open license, no gating, direct download from dl.fbaipublicfiles.com).

#SBATCH -J malle-embed-sscd
#SBATCH -o logs/embed_sscd_%j.out
#SBATCH -e logs/embed_sscd_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
source tacc/env.sh
mkdir -p logs

export MALLE_MODEL_FAMILY=sscd
export MALLE_DATASET_VARIANT="${MALLE_DATASET_VARIANT:-x1}"
# MALLE_SSCD_CHECKPOINT already set by tacc/env.sh; override here if you
# fetched a different checkpoint (e.g. sscd_disc_large):
# export MALLE_SSCD_CHECKPOINT="$MALLE_CHECKPOINTS/sscd/sscd_disc_large.torchscript.pt"

malle_run python "$MALLE_REPO/embed.py"
