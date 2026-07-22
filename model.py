# =============================================================================
# model.py  —  feature extractor builders (resnet50 | sscd | dinov3)
# =============================================================================
# ResNet50 named children (index → name → output shape for 224×224 input):
#   [0] conv1    Conv2d              (B,   64, 112, 112)
#   [1] bn1      BatchNorm2d         (B,   64, 112, 112)
#   [2] relu     ReLU                (B,   64, 112, 112)
#   [3] maxpool  MaxPool2d           (B,   64,  56,  56)
#   [4] layer1   Sequential (3 blk)  (B,  256,  56,  56)
#   [5] layer2   Sequential (4 blk)  (B,  512,  28,  28)
#   [6] layer3   Sequential (6 blk)  (B, 1024,  14,  14)
#   [7] layer4   Sequential (3 blk)  (B, 2048,   7,   7)
#   [8] avgpool  AdaptiveAvgPool2d   (B, 2048,   1,   1)
#   [9] fc       Linear              (B, 1000)          ← output is already flat
#
# Note on "fc": ResNet50's forward() calls torch.flatten(x, 1) between avgpool
# and fc inline — that op is not a child module, so nn.Sequential would skip it
# and pass (B, 2048, 1, 1) directly to the Linear layer, causing a shape error.
# build_resnet_extractor inserts nn.Flatten(1) explicitly when layer="fc".
#
# Note on POOL with "avgpool"/"fc": these layers already collapse spatial dims,
# so pool_fn is a no-op at that point. build_resnet_extractor's forward()
# guards against calling adaptive_avg_pool2d on a non-4D tensor.
#
# sscd and dinov3 extractors produce an already-flat (B, D) embedding, so their
# forward() closures skip pooling entirely — build_feature_extractor gives
# embed.py a single, uniform `forward(batch) -> (B, D) tensor` callable
# regardless of MODEL_FAMILY.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# Maps layer name → number of children to keep (children[:N])
_LAYER_CHILDREN_COUNT = {
    "conv1":   1,
    "bn1":     2,
    "relu":    3,
    "maxpool": 4,
    "layer1":  5,
    "layer2":  6,
    "layer3":  7,
    "layer4":  8,
    "avgpool": 9,
    "fc":      10,
}

# Maps pool name → function applied to feature map before flattening
# "none" keeps the full spatial map and flattens it.
_POOL_FN = {
    "none": lambda x: x,
    "gap":  lambda x: F.adaptive_avg_pool2d(x, (1, 1)),
    "3x3":  lambda x: F.adaptive_avg_pool2d(x, (3, 3)),
}


def build_resnet_extractor(
    layer: str,
    pool: str,
    device: torch.device,
) -> callable:
    """Build a frozen ResNet50 trunk and wrap it in a uniform forward() closure.

    Args:
        layer: Name of the last layer to include. Must be one of:
               "conv1", "bn1", "relu", "maxpool",
               "layer1", "layer2", "layer3", "layer4",
               "avgpool", "fc".
        pool:  Pooling strategy applied after the trunk. Must be one of:
               "none"  — no pooling; output is flattened spatially.
               "gap"   — global average pool to (1×1) then flatten.
               "3x3"   — adaptive average pool to (3×3) then flatten.
        device: Torch device to move the trunk to.

    Returns:
        forward — callable(batch: torch.Tensor) -> torch.Tensor of shape (B, D).

    Raises:
        ValueError: If `layer` or `pool` is not recognised.
    """
    if layer not in _LAYER_CHILDREN_COUNT:
        raise ValueError(
            f"Unknown layer {layer!r}. "
            f"Choose from: {list(_LAYER_CHILDREN_COUNT)}"
        )
    if pool not in _POOL_FN:
        raise ValueError(
            f"Unknown pool {pool!r}. "
            f"Choose from: {list(_POOL_FN)}"
        )

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    children = list(backbone.children())
    n = _LAYER_CHILDREN_COUNT[layer]

    if layer == "fc":
        # ResNet50's forward() flattens between avgpool and fc inline; that op
        # is not a child module, so we insert it explicitly here.
        trunk = nn.Sequential(*children[:9], nn.Flatten(1), children[9])
    else:
        trunk = nn.Sequential(*children[:n])

    trunk = trunk.to(device)
    trunk.eval()

    pool_fn = _POOL_FN[pool]

    def forward(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feature_map = trunk(batch.to(device))
            # pool_fn (gap / 3x3) requires a 4D spatial tensor. "avgpool"
            # outputs (B, 2048, 1, 1) and "fc" outputs (B, 1000) — both are
            # already non-spatial, so skip pooling in those cases.
            if feature_map.dim() == 4:
                feature_map = pool_fn(feature_map)
            return feature_map.flatten(1)

    return forward


def build_sscd_extractor(checkpoint_path: str, device: torch.device) -> callable:
    """Load a facebookresearch/sscd-copy-detection torchscript checkpoint.

    Torchscript SSCD checkpoints are standalone — no sscd-copy-detection
    install required. They output 512-d (or 1024-d for sscd_disc_large)
    L2-normalized descriptors directly, so no separate pooling step is
    needed. See tacc/fetch_checkpoints.py to download a checkpoint.

    Args:
        checkpoint_path: Path to a `*.torchscript.pt` file.
        device: Torch device to run inference on.

    Returns:
        forward — callable(batch: torch.Tensor) -> torch.Tensor of shape (B, D).
    """
    sscd_model = torch.jit.load(checkpoint_path, map_location=device)
    sscd_model.eval()

    def forward(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return sscd_model(batch.to(device))

    return forward


def build_dinov3_extractor(hf_model_id: str, device: torch.device) -> callable:
    """Load a DINOv3 backbone via Hugging Face Transformers.

    Requires the `transformers` package and access to the (gated) DINOv3
    weights — accept the license on the model's Hugging Face page and
    authenticate (HF_TOKEN) or pre-populate HF_HOME on the login node before
    running this on a compute node without internet access.

    Args:
        hf_model_id: Hugging Face model id, e.g.
            "facebook/dinov3-vitb16-pretrain-lvd1689m".
        device: Torch device to run inference on.

    Returns:
        forward — callable(batch: torch.Tensor) -> torch.Tensor of shape (B, D),
                  taken from the model's pooled output.
    """
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(hf_model_id)
    backbone = backbone.to(device)
    backbone.eval()

    def forward(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = backbone(pixel_values=batch.to(device))
            return output.pooler_output

    return forward


def build_feature_extractor(
    family: str,
    device: torch.device,
    *,
    layer: str | None = None,
    pool: str | None = None,
    sscd_checkpoint: str | None = None,
    dinov3_model: str | None = None,
) -> callable:
    """Dispatch to the right builder based on MODEL_FAMILY.

    This is the single entry point embed.py should use — it hides the
    per-model differences (pooling, checkpoint loading, HF vs torchscript)
    behind one `forward(batch) -> (B, D)` callable.

    Args:
        family: "resnet50" | "sscd" | "dinov3" (see config.MODEL_FAMILY).
        device: Torch device to run inference on.
        layer, pool: Required when family == "resnet50".
        sscd_checkpoint: Required when family == "sscd".
        dinov3_model: Required when family == "dinov3".

    Returns:
        forward — callable(batch: torch.Tensor) -> torch.Tensor of shape (B, D).

    Raises:
        ValueError: If `family` is not recognised.
    """
    if family == "resnet50":
        return build_resnet_extractor(layer, pool, device)
    if family == "sscd":
        return build_sscd_extractor(sscd_checkpoint, device)
    if family == "dinov3":
        return build_dinov3_extractor(dinov3_model, device)
    raise ValueError(
        f"Unknown model family {family!r}. Choose from: resnet50, sscd, dinov3"
    )
