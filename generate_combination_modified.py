"""Generate combination-based modified image sets for the Malle dataset.

For each modification count `x` from 1 through 11, this script creates a
separate output directory and writes one output image per unique combination of
`x` distinct modifications from the 11 supported modification types.

Each combination is applied exactly once to every image in
`malle_dataset/original_images`. Output file names follow the existing naming
style from `malle_dataset/modified_images`, appending the performed
modifications with underscores.
"""

from __future__ import annotations

import os
import random
from itertools import combinations

from PIL import Image
import torch

from generate_modified import (
    apply_modification,
    list_original_images,
    to_pil,
    to_tensor,
)

MALLE_PATH = "malle_dataset/"


def _default_modification_counts() -> list[int]:
    """Read modification counts from MALLE_MOD_COUNTS, else default to [1].

    Set MALLE_MOD_COUNTS to a comma-separated list (e.g. "3" or "1,2,3") to
    override without editing this file — lets a SLURM job array assign one
    count per task via `MALLE_MOD_COUNTS=$SLURM_ARRAY_TASK_ID`. See
    tacc/slurm/01_generate_dataset.slurm.
    """
    raw = os.environ.get("MALLE_MOD_COUNTS")
    if not raw:
        return [1]
    return [int(x) for x in raw.split(",") if x.strip()]


# Modification counts to generate. Each value becomes its own output folder.
# Full range is list(range(1, 12)), but that's 2**11 - 1 = 2047 combinations
# per image — generate a subset (e.g. via MALLE_MOD_COUNTS) unless you've
# budgeted the storage/compute for the full combinatorial explosion.
MODIFICATION_COUNTS = _default_modification_counts()

# Exhaustive set of modification types available for combination generation.
MODIFICATION_NAMES = [
    "cropping",
    "resizing",
    "rotation",
    "horizontal_flip",
    "vertical_flip",
    "blur",
    "brightness_contrast",
    "color",
    "watermark",
    "compression",
    "occlusion",
]


def build_output_directory(malle_path: str, modification_count: int) -> str:
    """Build the output directory path for a given modification count.

    Args:
        malle_path: Base directory containing the dataset folders.
        modification_count: Number of distinct modifications in each output
            combination.

    Returns:
        The output directory path for the given count.
    """
    return os.path.join(malle_path, f"modified_images_x{modification_count}")


def apply_modification_combo(
    image: Image.Image, modification_combo: tuple[str, ...]
) -> tuple[torch.Tensor, str]:
    """Apply a distinct modification combination to an image.

    Modifications are applied in the order provided by `modification_combo`.
    Because combinations are generated from `MODIFICATION_NAMES`, that order is
    the same relative order used by `MODIFICATION_NAMES`.

    Args:
        image: Source PIL image.
        modification_combo: Ordered tuple of modification names to apply.

    Returns:
        A tuple containing the modified tensor image and the output-name suffix.
    """
    modified_image = to_tensor(image)
    suffix_parts: list[str] = []

    for modification_name in modification_combo:
        modified_image, suffix = apply_modification(modified_image, modification_name)
        suffix_parts.append(suffix)

    return modified_image, "".join(suffix_parts)


def generate_combination_modified_images(
    malle_path: str = MALLE_PATH,
    seed: int | None = 17,
    modification_counts: list[int] | None = None,
) -> None:
    """Generate modified images for every unique modification combination.

    Args:
        malle_path: Base directory containing the dataset folders.
        seed: Optional random seed for reproducible output.
        modification_counts: Counts to generate. Defaults to 1 through 11.
    """
    if seed is not None:
        random.seed(seed)

    if modification_counts is None:
        modification_counts = MODIFICATION_COUNTS

    original_images = sorted(list_original_images(malle_path))

    for modification_count in modification_counts:
        output_dir = build_output_directory(malle_path, modification_count)
        os.makedirs(output_dir, exist_ok=True)

        for file_name in original_images:
            source_path = os.path.join(malle_path, "original_images", file_name)
            stem, ext = os.path.splitext(file_name)

            with Image.open(source_path) as source_image:
                source_image = source_image.convert("RGB")

                for modification_combo in combinations(
                    MODIFICATION_NAMES, modification_count
                ):
                    modified_image, suffix = apply_modification_combo(
                        source_image, modification_combo
                    )
                    output_image = to_pil(modified_image)
                    output_name = f"{stem}{suffix}{ext}"
                    output_path = os.path.join(output_dir, output_name)
                    output_image.save(output_path)


def main() -> None:
    """Run the combination-based modified-image generator."""
    generate_combination_modified_images()


if __name__ == "__main__":
    main()
