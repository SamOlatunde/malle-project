# Running Malle Project on TACC Vista

This directory automates two things on Vista:

1. **Dataset generation** — `generate_combination_modified.py` moved off your laptop and onto Vista's CPU nodes.
2. **Embedding extraction with new models** — ResNet50 (existing baseline), plus SSCD and DINOv3, all driven by the same `config.py` / `embed.py` pipeline via a new `MODEL_FAMILY` knob.

Read this once end-to-end before running anything — a few Vista-specific gotchas (architecture, gated weights, no internet on compute nodes) will save you a failed job otherwise.

## What changed in the repo

- `config.py` — added `MODEL_FAMILY` (`resnet50` | `sscd` | `dinov3`), `SSCD_CHECKPOINT`, `DINOV3_HF_MODEL`. `DATASET_VARIANT` and `MODEL_FAMILY` are now overridable via `MALLE_DATASET_VARIANT` / `MALLE_MODEL_FAMILY` env vars, so SLURM jobs can sweep both without editing the file.
- `model.py` — `build_feature_extractor()` now dispatches on `MODEL_FAMILY` and returns a uniform `forward(batch) -> (B, D)` callable for all three model families (previously ResNet50-only, returning a trunk + separate pool function).
- `embed.py` — picks the right preprocessing transform per family (each model's published inference recipe) and calls the new uniform `extractor` callable. Behavior for `MODEL_FAMILY=resnet50` (the default) is unchanged from before.
- `generate_combination_modified.py` — `MODIFICATION_COUNTS` now reads from `MALLE_MOD_COUNTS` (comma-separated) if set, so a SLURM array job can assign one modification count per task.

Nothing here changes local, non-TACC usage — every new knob defaults to today's behavior.

## Vista basics you need before starting

- **Architecture: Vista's GH200 nodes are aarch64 (ARM Grace CPU + H200 GPU), not x86_64.** Anything you `pip install`ed for x86 on your laptop (including your local `requirements.txt` wheels) will not run as-is. Use a container with an aarch64-built PyTorch/CUDA stack instead of hand-installing packages — see "Container" below.
- **Partitions:** `gh` for GPU work (GH200/H200 nodes), and what appears to be a `gg` (Grace-Grace, CPU-only) partition for CPU work. Confirm exact names with `sinfo` when you log in — they're used as placeholders in the scripts here (`tacc/slurm/*.slurm`), swap if TACC has renamed/restructured them since.
- **Storage tiers:**
  - `$HOME` — small, permanent. Don't put datasets, checkpoints, or containers here.
  - `$WORK` — Lustre, persistent, large-file-friendly. Put the repo, containers, and model checkpoints here.
  - `$SCRATCH` — fast, not quota'd, but **purged after ~10 days of inactivity**. Fine for transient job I/O, not for anything you need to keep.
- **Compute nodes generally have no outbound internet access.** Anything that needs to be downloaded (pretrained weights, containers, `pip install`) has to happen on a **login node** first, cached somewhere under `$WORK`, and pointed to from the compute-node job via env vars. `tacc/fetch_checkpoints.py` and the container pull below both need to run on a login node.
- **Data transfer:** for `malle_dataset/original_images/` (currently ~142 MB total including x1 variants — small), plain `rsync`/`scp` to `$WORK` is fine. If you scale the corpus up significantly later, use [Globus](https://docs.tacc.utexas.edu/datatransfer/globus/) instead — it's TACC's recommended path for large transfers and handles interruptions/retries better than scp.

## One-time setup (run on a Vista login node)

```bash
# 1. Get the repo onto Vista
rsync -avz --exclude .venv --exclude __pycache__ /path/to/malle-project/ vista.tacc.utexas.edu:$WORK/malle-project/
ssh vista.tacc.utexas.edu

# 2. Send your existing original_images (no need to regenerate from imagenet-mini on Vista)
#    (run from your laptop, not on Vista)
rsync -avz malle_dataset/original_images/ vista.tacc.utexas.edu:$WORK/malle-project/malle_dataset/original_images/

# 3. Pull an aarch64-compatible PyTorch container
export APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
cd $WORK
apptainer pull malle_pytorch.sif docker://nvcr.io/nvidia/pytorch:<TAG>
# Pick <TAG> from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch —
# confirm it publishes an arm64/aarch64 manifest before pulling (most recent
# NGC PyTorch tags do, but check rather than assume).

# 4. Add the packages the container doesn't ship with (faiss, transformers, streamlit)
apptainer exec malle_pytorch.sif pip install --user faiss-cpu transformers huggingface_hub streamlit
# If faiss-cpu has no aarch64 wheel on PyPI at the time you do this, install
# it via conda-forge instead (it does publish linux-aarch64 builds):
#   apptainer exec malle_pytorch.sif bash -c \
#     "curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba && \
#      ./bin/micromamba install -y -c conda-forge faiss-cpu"

# 5. Pre-fetch model weights (compute nodes can't do this themselves)
cd malle-project
python tacc/fetch_checkpoints.py --resnet50
python tacc/fetch_checkpoints.py --sscd
# DINOv3 is gated — see tacc/slurm/03b_embed_dinov3.slurm for the access-request
# steps, then:
python tacc/fetch_checkpoints.py --dinov3 --dinov3-model facebook/dinov3-vitb16-pretrain-lvd1689m
```

## Pipeline order

```
tacc/slurm/01_generate_dataset.slurm        # malle_dataset/modified_images_x<N>/
        │
        ▼
tacc/slurm/02_embed_resnet50.slurm          # baseline — same as your local pipeline
tacc/slurm/03a_embed_sscd.slurm             # SSCD embeddings
tacc/slurm/03b_embed_dinov3.slurm           # DINOv3 embeddings
        │
        ▼
tacc/slurm/04_search_and_evaluate.slurm     # FAISS index + Recall@k, per model/variant
```

Submit with dependencies so search/eval doesn't run before its embeddings exist, e.g.:

```bash
cd $WORK/malle-project
mkdir -p logs
GEN=$(sbatch --parsable tacc/slurm/01_generate_dataset.slurm)
EMB=$(sbatch --parsable --dependency=afterok:$GEN tacc/slurm/02_embed_resnet50.slurm)
sbatch --dependency=afterok:$EMB tacc/slurm/04_search_and_evaluate.slurm
```

Repeat the embed → search/eval pair per model family (`03a_embed_sscd.slurm`, `03b_embed_dinov3.slurm`) and per `MALLE_DATASET_VARIANT` you care about, using `sbatch --export=MALLE_MODEL_FAMILY=...,MALLE_DATASET_VARIANT=...` to target `04_search_and_evaluate.slurm` at the right run.

## On the "full dataset" scope

You said: same source (the 200-image imagenet-mini sample), but generate more combinatorial variants instead of just `x1`. Two things worth deciding before you launch a big `--array` range:

- **Combinatorial growth is steep.** `x1..x4` is `C(11,1)+C(11,2)+C(11,3)+C(11,4) = 11+55+165+330 = 561` combinations/image → 112,200 images at x1-x4 combined (vs. 2,200 at x1 alone). Full `x1..x11` is 2,047 combos/image → 409,400 images. Storage and embedding time both scale linearly with this, so pick a ceiling deliberately rather than defaulting to `range(1,12)`.
- **`01_generate_dataset.slurm`'s `--array=1-4`** generates x1 through x4 as an example starting point — widen it once you've seen the size/runtime of the first few counts.

## Known risks / things to verify once you're actually on Vista

- Exact `gh`/`gg` partition names, queue limits, and allocation flag (`-A`) — confirm via `sinfo` and `showq -u` rather than trusting the placeholders here verbatim.
- `faiss-cpu` aarch64 wheel availability on PyPI at the time you set this up (fallback via conda-forge noted above).
- NGC PyTorch container tag with a published aarch64/arm64 manifest — check the catalog page rather than assuming the newest tag has one.
- DINOv3 license acceptance + `HF_TOKEN` must be done before `fetch_checkpoints.py --dinov3` will succeed; it's a gated model, not an open download like SSCD.
