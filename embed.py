"""
Module: embed.py

Embedding pipeline for the Malle dataset.
All experiment knobs (dataset variant, ResNet layer, pooling strategy) are
controlled exclusively via config.py — no edits needed here.
"""

import json
import os

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

import config
from dataset import Dataset, safe_collatefn
from model import build_feature_extractor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

extractor = build_feature_extractor(
    config.MODEL_FAMILY,
    device,
    layer=getattr(config, 'LAYER', None),
    pool=getattr(config, 'POOL', None),
    sscd_checkpoint=getattr(config, 'SSCD_CHECKPOINT', None),
    dinov3_model=getattr(config, 'DINOV3_HF_MODEL', None),
)

# Image preprocessing — one recipe per model family, matching each model's
# published inference transform. All three normalize with ImageNet mean/std;
# they differ only in resize/crop behaviour:
#   resnet50 — Resize(256) + CenterCrop(224)   (torchvision ImageNet eval)
#   sscd     — Resize([320, 320])              (square/"skew", keeps batching
#                                                simple; README also allows a
#                                                288px-short-edge crop-free
#                                                variant for single images)
#   dinov3   — Resize((256, 256))              (facebookresearch/dinov3 README)
_IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)

_TRANSFORMS = {
    'resnet50': transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        _IMAGENET_NORMALIZE,
    ]),
    'sscd': transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize([320, 320]),
        _IMAGENET_NORMALIZE,
    ]),
    'dinov3': transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize([256, 256], antialias=True),
        _IMAGENET_NORMALIZE,
    ]),
}

transform = _TRANSFORMS[config.MODEL_FAMILY]


def embed_batch(batch: torch.Tensor) -> torch.Tensor:
    """Run one batch through the configured feature extractor.

    Args:
        batch: Float tensor of shape (B, C, H, W).

    Returns:
        Flat embedding tensor of shape (B, D). Pooling (for resnet50) or
        model-specific head logic (for sscd/dinov3) is handled inside the
        `extractor` closure built by model.build_feature_extractor — this
        function is family-agnostic.
    """
    return extractor(batch)


def load_embeddings(filename: str) -> np.ndarray:
    """Load embeddings from a .npy file.

    Args:
        filename: Path to the .npy file (including extension).

    Returns:
        Numpy array of embeddings.
    """
    return np.load(filename)


def save_embeddings(filename: str, embeddings: np.ndarray) -> None:
    """Save embeddings to a .npy file.

    Args:
        filename: Destination path (including .npy extension).
        embeddings: Array to save.
    """
    np.save(filename, embeddings)


def embed_folder(input_dir: str, outfile: str, batch_size: int) -> None:
    """Embed all images in a folder and write normalised vectors to disk.

    Args:
        input_dir: Directory of images to embed.
        outfile:   Destination .npy path for the resulting embeddings.
        batch_size: DataLoader batch size.
    """
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    dataset    = Dataset(input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=safe_collatefn)

    batches = []
    for batch, _ in dataloader:
        batches.append(embed_batch(batch))

    embeddings = torch.cat(batches, dim=0)
    embeddings = embeddings.squeeze()

    normalised = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    save_embeddings(outfile, normalised.cpu().numpy())


def stream_jsonl(filename: str):
    """Yield one JSON object at a time from a .jsonl file."""
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def load_jsonl(filename: str) -> list[dict]:
    """Load all records from a .jsonl file.

    Args:
        filename: Path to the .jsonl file.

    Returns:
        List of dicts, one per line.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def extract_query_metadata(query_input_dir: str, query_outfile: str) -> None:
    """Extract metadata for modified (query) images and write to a .jsonl file.

    Each line: {"id", "class", "instance_id", "modifications", "path"}

    Args:
        query_input_dir: Folder containing modified images.
        query_outfile:   Destination .jsonl path.
    """
    os.makedirs(os.path.dirname(query_outfile), exist_ok=True)
    with open(query_outfile, 'a', encoding='utf-8') as f:
        for idx, img_name in enumerate(os.listdir(query_input_dir)):
            parts     = img_name.split('_')
            img_class = parts[0]
            image_path = os.path.join(query_input_dir, img_name)
            mods = parts[2:]
            mods[-1] = mods[-1].split('.')[0]
            f.write(json.dumps({
                'id': idx,
                'class': img_class,
                'instance_id': parts[1],
                'modifications': mods,
                'path': image_path,
            }) + '\n')


def extract_index_metadata(index_input_dir: str, index_outfile: str) -> None:
    """Extract metadata for original (index) images and write to a .jsonl file.

    Each line: {"id", "class", "instance_id", "path"}

    Args:
        index_input_dir: Folder containing original images.
        index_outfile:   Destination .jsonl path.
    """
    os.makedirs(os.path.dirname(index_outfile), exist_ok=True)
    with open(index_outfile, 'a', encoding='utf-8') as f:
        for idx, img_name in enumerate(os.listdir(index_input_dir)):
            parts      = img_name.split('_')
            img_class  = parts[0]
            image_path = os.path.join(index_input_dir, img_name)
            instance_id, _ = parts[-1].rsplit('.', 1)
            f.write(json.dumps({
                'id': idx,
                'class': img_class,
                'instance_id': instance_id,
                'path': image_path,
            }) + '\n')


if __name__ == '__main__':
    # Embed index (original) images
    embed_folder(config.INDEX_DIR, config.EMBED_INDEX, batch_size=16)

    # Embed query (modified) images
    embed_folder(config.QUERY_DIR, config.EMBED_QUERY, batch_size=16)

    # Generate metadata (only needed once per dataset variant)
    extract_index_metadata(config.INDEX_DIR, config.INDEX_META)
    extract_query_metadata(config.QUERY_DIR, config.QUERY_META)
