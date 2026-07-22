"""Generate modified versions of images in the Malle dataset.

This module provides importable helper functions for each supported image
modification as well as a small pipeline for generating randomized variants.
The script only executes its batch generation loop when run directly.
"""

from __future__ import annotations

import os
import random

from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms.v2 as transforms

MALLE_PATH = "malle_dataset/"

IMG_OPS = {
    "cropping": {
        "center_crop": {"params": {"size": [256, 384]}},
        "random_crop": {"params": {"size": [256, 384]}},
    },
    "resizing": {"params": {"min_size": 256, "max_size": 384}},
    "rotation": {"params": {"degrees": (-15, 15)}},
    "horizontal_flip": {"params": {"p": 1.0}},
    "vertical_flip": {"params": {"p": 1.0}},
    "blur": {"params": {"kernel_size": [7, 11, 15], "sigma": (1.0, 5.0)}},
    "brightness_contrast": {
        "params": {"brightness_factor": (0.8, 1.2), "contrast_factor": (0.6, 1.4)}
    },
    "color": {"params": {"hue_shift": (-0.5, 0.5), "saturation_factor": (0.6, 1.4)}},
    "watermark": {
        "params": {
            "text": "Malle Project",
            "position": (30, 30),
            "font_size": 80,
            "opacity": 120,
        }
    },
    "compression": {"params": {"quality": (30, 80)}},
    "occlusion": {"params": {"mask_size": (80, 80), "mask_position": (90, 90)}},
}

POSS_NUM_MODS_PER_IMG = [4, 8]

# Canonical conversions:
# PIL -> float tensor in [0, 1].
PIL_TO_TENSOR = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)

# Tensor in [0, 1] -> PIL.
TENSOR_TO_PIL = transforms.ToPILImage()


def list_original_images(malle_path: str = MALLE_PATH) -> list[str]:
    """Return the original image file names.

    Args:
        malle_path: Base directory containing the dataset folders.

    Returns:
        A list of file names found in the `original_images` directory.
    """
    return os.listdir(os.path.join(malle_path, "original_images"))


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float tensor in [0, 1].

    Args:
        image: Input PIL image.

    Returns:
        The image as a float tensor.
    """
    return PIL_TO_TENSOR(image)


def to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a tensor image in [0, 1] to a PIL image.

    Args:
        image: Input tensor image.

    Returns:
        The image as a PIL image.
    """
    return TENSOR_TO_PIL(image)


def apply_center_crop(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Apply a center crop to an image tensor.

    Args:
        image: Input tensor image.
        crop_size: Crop size to apply.

    Returns:
        The cropped image tensor.
    """
    return transforms.CenterCrop(crop_size)(image)


def apply_random_crop(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    """Apply a random crop to an image tensor.

    Args:
        image: Input tensor image.
        crop_size: Crop size to apply.

    Returns:
        The cropped image tensor.
    """
    return transforms.RandomCrop(
        crop_size, pad_if_needed=True, padding_mode="constant"
    )(image)


def apply_resizing(image: torch.Tensor) -> torch.Tensor:
    """Apply a random resize to an image tensor.

    Args:
        image: Input tensor image.

    Returns:
        The resized image tensor.
    """
    return transforms.RandomResize(**IMG_OPS["resizing"]["params"])(image)


def apply_rotation(image: torch.Tensor) -> torch.Tensor:
    """Apply a random rotation to an image tensor.

    Args:
        image: Input tensor image.

    Returns:
        The rotated image tensor.
    """
    return transforms.RandomRotation(IMG_OPS["rotation"]["params"]["degrees"])(image)


def apply_horizontal_flip(image: torch.Tensor) -> torch.Tensor:
    """Flip an image tensor horizontally.

    Args:
        image: Input tensor image.

    Returns:
        The horizontally flipped image tensor.
    """
    return transforms.RandomHorizontalFlip(
        p=IMG_OPS["horizontal_flip"]["params"]["p"]
    )(image)


def apply_vertical_flip(image: torch.Tensor) -> torch.Tensor:
    """Flip an image tensor vertically.

    Args:
        image: Input tensor image.

    Returns:
        The vertically flipped image tensor.
    """
    return transforms.RandomVerticalFlip(
        p=IMG_OPS["vertical_flip"]["params"]["p"]
    )(image)


def apply_blur(image: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur to an image tensor.

    The kernel size is selected from the configured blur options based on the
    smallest image dimension.

    Args:
        image: Input tensor image.

    Returns:
        The blurred image tensor.
    """
    _, h, w = image.shape
    chosen_dim = min(h, w)

    if chosen_dim <= 128:
        kernel_size = IMG_OPS["blur"]["params"]["kernel_size"][0]
    elif chosen_dim < 256:
        kernel_size = IMG_OPS["blur"]["params"]["kernel_size"][1]
    else:
        kernel_size = IMG_OPS["blur"]["params"]["kernel_size"][2]

    sigma = IMG_OPS["blur"]["params"]["sigma"]
    return transforms.GaussianBlur(
        kernel_size=(kernel_size, kernel_size), sigma=sigma
    )(image)


def apply_brightness_contrast(image: torch.Tensor) -> torch.Tensor:
    """Apply brightness and contrast jitter to an image tensor.

    Args:
        image: Input tensor image.

    Returns:
        The adjusted image tensor.
    """
    color_jitter = transforms.ColorJitter(
        brightness=IMG_OPS["brightness_contrast"]["params"]["brightness_factor"],
        contrast=IMG_OPS["brightness_contrast"]["params"]["contrast_factor"],
    )
    return color_jitter(image)


def apply_color(image: torch.Tensor) -> torch.Tensor:
    """Apply color jitter to an image tensor.

    Args:
        image: Input tensor image.

    Returns:
        The adjusted image tensor.
    """
    color_jitter = transforms.ColorJitter(
        saturation=IMG_OPS["color"]["params"]["saturation_factor"],
        hue=IMG_OPS["color"]["params"]["hue_shift"],
    )
    return color_jitter(image)


def apply_watermark(image: torch.Tensor) -> torch.Tensor:
    """Overlay a watermark on an image tensor.

    Args:
        image: Input tensor image in the [0, 1] range.

    Returns:
        The watermarked image tensor.
    """
    pil_image = to_pil(image).convert("RGBA")
    watermark_base = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(watermark_base)

    try:
        font = ImageFont.truetype(
            "arial.ttf", IMG_OPS["watermark"]["params"]["font_size"]
        )
    except OSError:
        font = ImageFont.load_default()

    text = IMG_OPS["watermark"]["params"]["text"]
    position = IMG_OPS["watermark"]["params"]["position"]
    opacity = IMG_OPS["watermark"]["params"]["opacity"]

    draw.text(
        position,
        text,
        fill=(255, 255, 255, opacity),
        font=font,
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )

    watermarked = Image.alpha_composite(pil_image, watermark_base).convert("RGB")
    return to_tensor(watermarked)


def apply_compression(image: torch.Tensor) -> torch.Tensor:
    """Apply JPEG compression to an image tensor.

    Args:
        image: Input tensor image in the [0, 1] range.

    Returns:
        The compressed image tensor.
    """
    pil_image = to_pil(image).convert("RGB")
    q_min, q_max = IMG_OPS["compression"]["params"]["quality"]
    quality = random.randint(q_min, q_max)
    return to_tensor(transforms.JPEG(quality=quality)(pil_image))


def apply_occlusion(image: torch.Tensor) -> torch.Tensor:
    """Apply a black rectangular occlusion to an image tensor.

    Args:
        image: Input tensor image in the [0, 1] range.

    Returns:
        The occluded image tensor.
    """
    x_max, y_max = IMG_OPS["occlusion"]["params"]["mask_position"]
    x = random.randint(0, x_max)
    y = random.randint(0, y_max)
    height, width = IMG_OPS["occlusion"]["params"]["mask_size"]

    image[:, y : y + height, x : x + width] = 0.0
    return image


def apply_modification(image: torch.Tensor, modification: str) -> tuple[torch.Tensor, str]:
    """Apply one named modification to an image tensor.

    Args:
        image: Input tensor image in the [0, 1] range.
        modification: Name of the modification to apply.

    Returns:
        A tuple containing the modified image tensor and the suffix fragment to
        append to the output file name.

    Raises:
        ValueError: If the modification name is not supported.
    """
    if modification == "cropping":
        crop_flag = random.randint(0, 1)
        if crop_flag == 0:
            crop_size = random.choice(
                IMG_OPS["cropping"]["center_crop"]["params"]["size"]
            )
            return apply_center_crop(image, crop_size), f"_centCrop{crop_size}"

        crop_size = random.choice(
            IMG_OPS["cropping"]["random_crop"]["params"]["size"]
        )
        return apply_random_crop(image, crop_size), f"_randCrop{crop_size}"

    if modification == "resizing":
        return apply_resizing(image), "_resize"

    if modification == "rotation":
        return apply_rotation(image), "_rotate"

    if modification == "horizontal_flip":
        return apply_horizontal_flip(image), "_hflip"

    if modification == "vertical_flip":
        return apply_vertical_flip(image), "_vflip"

    if modification == "blur":
        return apply_blur(image), "_blur"

    if modification == "brightness_contrast":
        return apply_brightness_contrast(image), "_brightness_contrast"

    if modification == "color":
        return apply_color(image), "_color"

    if modification == "watermark":
        return apply_watermark(image), "_watermark"

    if modification == "compression":
        return apply_compression(image), "_compression"

    if modification == "occlusion":
        return apply_occlusion(image), "_occlusion"

    raise ValueError(f"Unsupported modification: {modification}")


def generate_modified_images(
    malle_path: str = MALLE_PATH,
    seed: int | None = 17,
    num_mods_per_img: list[int] | None = None,
) -> None:
    """Generate randomized modified variants for every original image.

    Args:
        malle_path: Base directory containing the dataset folders.
        seed: Optional random seed for reproducible output.
        num_mods_per_img: Number of variants to generate per source image. If
            omitted, the module default options are used.
    """
    if seed is not None:
        random.seed(seed)

    if num_mods_per_img is None:
        num_mods_per_img = POSS_NUM_MODS_PER_IMG

    original_images = list_original_images(malle_path)

    for pic in original_images:
        num_mods = random.choice(num_mods_per_img)
        source_path = os.path.join(malle_path, "original_images", pic)
        with Image.open(source_path) as pic_img:
            pic_img = pic_img.convert("RGB")

            for _ in range(num_mods):
                modified_pic = to_tensor(pic_img.copy())
                mods_str = ""
                mod_depth = random.randint(1, 7)
                mod_list = random.sample(list(IMG_OPS.keys()), k=mod_depth)

                for mod in mod_list:
                    modified_pic, suffix = apply_modification(modified_pic, mod)
                    mods_str += suffix

                final_pil = to_pil(modified_pic)
                name, ext = pic.rsplit(".", 1)
                out_path = os.path.join(
                    malle_path, "modified_images", f"{name}{mods_str}.{ext}"
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                final_pil.save(out_path)


def main() -> None:
    """Run the batch modified-image generator."""
    generate_modified_images()


if __name__ == "__main__":
    main()
