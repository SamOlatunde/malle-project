#!/bin/bash
# tacc/env.sh — sourced by every SLURM script in tacc/slurm/.
#
# Centralizes the handful of things that differ between your laptop and a
# Vista compute node: container path, cache locations (so nothing writes to
# the small $HOME filesystem), and checkpoint locations produced by
# tacc/fetch_checkpoints.py. Edit the paths below once; every job script
# picks them up by `source tacc/env.sh`.
#
# Verify against `docs.tacc.utexas.edu/hpc/vista` before first use — exact
# partition names, module names, and container tags can change.

set -euo pipefail

# ── Allocation ────────────────────────────────────────────────────────────
# Only needed if your TACC account has more than one active allocation.
# export SBATCH_ACCOUNT=<YourAllocationName>

# ── Repo + data locations ────────────────────────────────────────────────
export MALLE_REPO="${MALLE_REPO:-$WORK/malle-project}"
export MALLE_CACHE_ROOT="${MALLE_CACHE_ROOT:-$WORK}"
export MALLE_CHECKPOINTS="$MALLE_CACHE_ROOT/malle_checkpoints"

# ── Container ─────────────────────────────────────────────────────────────
# Built once, on a login node, via:
#   export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
#   cd $WORK && apptainer pull malle_pytorch.sif docker://nvcr.io/nvidia/pytorch:<TAG>
# Pick <TAG> from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# — Vista's GH200 nodes are aarch64 (Grace CPU), so confirm the tag has an
# arm64/aarch64 manifest before pulling.
export MALLE_CONTAINER="${MALLE_CONTAINER:-$WORK/malle_pytorch.sif}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/apptainer_cache}"

# ── Model weight caches (populated by tacc/fetch_checkpoints.py) ───────────
export TORCH_HOME="$MALLE_CHECKPOINTS/torch_home"
export HF_HOME="$MALLE_CHECKPOINTS/hf_home"
export MALLE_SSCD_CHECKPOINT="${MALLE_SSCD_CHECKPOINT:-$MALLE_CHECKPOINTS/sscd/sscd_disc_mixup.torchscript.pt}"

# Convenience wrapper: run a command inside the container with the repo and
# checkpoint dirs bound in, GPU enabled.
malle_run() {
  apptainer exec --nv \
    --bind "$MALLE_REPO:$MALLE_REPO" \
    --bind "$MALLE_CACHE_ROOT:$MALLE_CACHE_ROOT" \
    --env "TORCH_HOME=$TORCH_HOME,HF_HOME=$HF_HOME" \
    "$MALLE_CONTAINER" "$@"
}
