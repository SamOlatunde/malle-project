"""
tacc/fetch_checkpoints.py

Pre-downloads model weights on a TACC login node (which has internet access)
so that compute-node SLURM jobs — which typically do NOT have outbound
internet access — can load everything from local disk.

Run this once per model family you plan to use, from a Vista login node:

    python tacc/fetch_checkpoints.py --resnet50
    python tacc/fetch_checkpoints.py --sscd
    python tacc/fetch_checkpoints.py --dinov3 --dinov3-model facebook/dinov3-vitb16-pretrain-lvd1689m

Everything is written under $WORK by default (persistent, large-file-friendly
Lustre filesystem) — point MALLE_CACHE_ROOT elsewhere if you'd rather use a
different location.

DINOv3 weights are gated: request access at
https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/, accept
the license on the corresponding Hugging Face model page, and set HF_TOKEN
(a Hugging Face access token) in your environment before running --dinov3.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

CACHE_ROOT = Path(os.environ.get("MALLE_CACHE_ROOT", os.environ.get("WORK", ".")))
CHECKPOINT_DIR = CACHE_ROOT / "malle_checkpoints"

# Open license, direct download — see facebookresearch/sscd-copy-detection.
SSCD_URLS = {
    "sscd_disc_mixup": "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt",
    "sscd_disc_large": "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt",
}


def fetch_resnet50() -> None:
    """Warm the torchvision/torch.hub cache with ResNet50 ImageNet weights.

    Sets TORCH_HOME so the download lands under CHECKPOINT_DIR rather than
    $HOME (which is small and not meant for large caches). Compute-node jobs
    must set the same TORCH_HOME before importing torchvision.
    """
    torch_home = CHECKPOINT_DIR / "torch_home"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)

    from torchvision.models import resnet50, ResNet50_Weights

    print(f"Downloading ResNet50 ImageNet1K_V2 weights into {torch_home} ...")
    resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    print("Done. Set TORCH_HOME=%s in your SLURM scripts." % torch_home)


def fetch_sscd(model_name: str) -> None:
    """Download an SSCD torchscript checkpoint via wget."""
    if model_name not in SSCD_URLS:
        raise ValueError(f"Unknown SSCD model {model_name!r}. Choose from: {list(SSCD_URLS)}")

    out_dir = CHECKPOINT_DIR / "sscd"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = SSCD_URLS[model_name]
    out_path = out_dir / f"{model_name}.torchscript.pt"

    if out_path.exists():
        print(f"{out_path} already exists, skipping download.")
        return

    print(f"Downloading {url} -> {out_path} ...")
    subprocess.run(["wget", "-O", str(out_path), url], check=True)
    print(f"Done. Set MALLE_SSCD_CHECKPOINT={out_path} in your SLURM scripts.")


def fetch_dinov3(hf_model_id: str) -> None:
    """Download a gated DINOv3 checkpoint via the Hugging Face Hub.

    Requires HF_TOKEN in the environment and that you've accepted the model's
    license on huggingface.co first — otherwise this will fail with a 403.
    """
    hf_home = CHECKPOINT_DIR / "hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    if "HF_TOKEN" not in os.environ:
        raise SystemExit(
            "HF_TOKEN is not set. Accept the license for "
            f"{hf_model_id} on huggingface.co, create an access token, and "
            "export HF_TOKEN=... before re-running this."
        )

    from huggingface_hub import snapshot_download

    print(f"Downloading {hf_model_id} into {hf_home} ...")
    snapshot_download(repo_id=hf_model_id)
    print(f"Done. Set HF_HOME={hf_home} in your SLURM scripts.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resnet50", action="store_true", help="Warm ResNet50 weights.")
    parser.add_argument("--sscd", action="store_true", help="Download an SSCD checkpoint.")
    parser.add_argument(
        "--sscd-model", default="sscd_disc_mixup", choices=list(SSCD_URLS),
        help="Which SSCD checkpoint to fetch (default: sscd_disc_mixup).",
    )
    parser.add_argument("--dinov3", action="store_true", help="Download DINOv3 weights via HF Hub.")
    parser.add_argument(
        "--dinov3-model", default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face model id for DINOv3 (default: facebook/dinov3-vitb16-pretrain-lvd1689m).",
    )
    args = parser.parse_args()

    if not (args.resnet50 or args.sscd or args.dinov3):
        parser.error("Pass at least one of --resnet50, --sscd, --dinov3.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.resnet50:
        fetch_resnet50()
    if args.sscd:
        fetch_sscd(args.sscd_model)
    if args.dinov3:
        fetch_dinov3(args.dinov3_model)


if __name__ == "__main__":
    main()
