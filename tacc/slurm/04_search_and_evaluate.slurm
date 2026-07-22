#!/bin/bash
# tacc/slurm/04_search_and_evaluate.slurm
#
# Builds the FAISS index, runs k-NN search, and computes Recall@k for
# whichever MODEL_FAMILY/DATASET_VARIANT combination you point it at. CPU
# only — no need for a GPU node here.
#
# Usage:
#   sbatch --export=MALLE_MODEL_FAMILY=sscd,MALLE_DATASET_VARIANT=x2 \
#       tacc/slurm/04_search_and_evaluate.slurm

#SBATCH -J malle-search-eval
#SBATCH -o logs/search_eval_%j.out
#SBATCH -e logs/search_eval_%j.err
#SBATCH -p gg                      # CPU-only queue — verify with `sinfo`; falls back to `gh` if unavailable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
source tacc/env.sh
mkdir -p logs

export MALLE_MODEL_FAMILY="${MALLE_MODEL_FAMILY:-resnet50}"
export MALLE_DATASET_VARIANT="${MALLE_DATASET_VARIANT:-x1}"

malle_run python "$MALLE_REPO/faiss_index_and_search.py"
malle_run python "$MALLE_REPO/evaluate.py"
