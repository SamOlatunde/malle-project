# =============================================================================
# config.py  —  single source of truth for every experiment knob
# =============================================================================
# Change the values below; everything else is derived automatically.
#
# MODEL_FAMILY       : "resnet50" | "sscd" | "dinov3"
#   Selects which feature extractor model.py builds. Overridable via the
#   MALLE_MODEL_FAMILY env var so SLURM jobs on TACC can switch models
#   without editing this file (see tacc/README.md).
#
# Valid LAYER values (resnet50 only) :
#                      "conv1" | "bn1" | "relu" | "maxpool" |
#                      "layer1" | "layer2" | "layer3" | "layer4" |
#                      "avgpool" | "fc"
# Valid POOL values  (resnet50 only) : "none" | "gap" | "3x3"
# DATASET_VARIANT    : "x0" … "x11"  (matches modified_images_x<N> folder)
# =============================================================================

import os

# Overridable via MALLE_DATASET_VARIANT so a SLURM job array can embed one
# modified_images_x<N> folder per task without editing this file.
DATASET_VARIANT = os.environ.get("MALLE_DATASET_VARIANT", "x1")
LAYER           = "fc"
POOL            = "3x3"

MODEL_FAMILY = os.environ.get("MALLE_MODEL_FAMILY", "resnet50")

# ── sscd knobs ───────────────────────────────────────────────────────────────
# Torchscript checkpoint from facebookresearch/sscd-copy-detection. Open
# license, direct download — see tacc/fetch_checkpoints.py.
SSCD_CHECKPOINT = os.environ.get(
    "MALLE_SSCD_CHECKPOINT", "checkpoints/sscd/sscd_disc_mixup.torchscript.pt"
)

# ── dinov3 knobs ─────────────────────────────────────────────────────────────
# Hugging Face model id. DINOv3 weights are gated — you must accept the
# license on the model page and authenticate (HF_TOKEN) before this can
# download, or point HF_HOME at a cache pre-populated on the login node.
DINOV3_HF_MODEL = os.environ.get(
    "MALLE_DINOV3_MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m"
)

# ── Derived paths ─────────────────────────────────────────────────────────────
# Do not edit below this line.

MALLE_PATH = "malle_dataset/"
INDEX_DIR  = f"{MALLE_PATH}original_images/"
QUERY_DIR  = f"{MALLE_PATH}modified_images_{DATASET_VARIANT}/"

if MODEL_FAMILY == "resnet50":
    RUN_TAG = f"resnet50_{LAYER}_{POOL}"
elif MODEL_FAMILY == "sscd":
    RUN_TAG = "sscd_disc_mixup"
elif MODEL_FAMILY == "dinov3":
    RUN_TAG = DINOV3_HF_MODEL.rsplit("/", 1)[-1]  # e.g. "dinov3-vitb16-pretrain-lvd1689m"
else:
    raise ValueError(
        f"Unknown MODEL_FAMILY {MODEL_FAMILY!r}. Choose from: resnet50, sscd, dinov3"
    )

EMBED_INDEX  = f"embeddings/{RUN_TAG}_index.npy"
EMBED_QUERY  = f"embeddings/{RUN_TAG}_{DATASET_VARIANT}_query.npy"

INDEX_META = "metadata/index_metadata.jsonl"
QUERY_META = f"metadata/queries_{DATASET_VARIANT}_metadata.jsonl"

FAISS_INDEX = f"index/faiss_{RUN_TAG}.index"
RESULTS     = f"results/{RUN_TAG}_{DATASET_VARIANT}_results.jsonl"
