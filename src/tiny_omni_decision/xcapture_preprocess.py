"""Fail-closed preprocessing for X-Capture RGB images and object masks."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def masked_object_crop(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    padding_fraction: float = 0.08,
    minimum_mask_coverage: float = 0.01,
    maximum_mask_coverage: float = 0.95,
) -> tuple[Image.Image, dict[str, int | float]]:
    """Return an RGB crop with background zeroed and audit measurements.

    The mask is required to be a nonempty binary 0/255 image aligned to the RGB
    image. Padding is relative to the larger foreground-box dimension.
    """
    if not 0.0 <= padding_fraction <= 0.5:
        raise ValueError("padding_fraction must be between 0 and 0.5")
    if not 0.0 < minimum_mask_coverage < maximum_mask_coverage <= 1.0:
        raise ValueError("mask coverage bounds must satisfy 0 < minimum < maximum <= 1")

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    with Image.open(mask_path) as source_mask:
        mask = source_mask.convert("L")

    if image.size != mask.size:
        raise ValueError(f"image/mask dimensions differ: {image.size} != {mask.size}")
    colors_and_counts = mask.getcolors(maxcolors=256) or []
    values = {value for _count, value in colors_and_counts}
    if not values or not values.issubset({0, 255}):
        raise ValueError(f"mask must contain only binary 0/255 values; found {sorted(values)}")

    width, height = image.size
    mask_pixels = sum(count for count, value in colors_and_counts if value == 255)
    coverage = mask_pixels / (width * height)
    if not minimum_mask_coverage <= coverage <= maximum_mask_coverage:
        raise ValueError(f"mask coverage {coverage:.6f} is outside configured bounds")

    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("mask has no foreground pixels")
    left, top, right, bottom = bbox
    padding = round(max(right - left, bottom - top) * padding_fraction)
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )

    masked_image = Image.composite(image, Image.new("RGB", image.size), mask)
    crop = masked_image.crop(crop_box)
    return crop, {
        "image_width": width,
        "image_height": height,
        "mask_pixels": mask_pixels,
        "mask_coverage": coverage,
        "padding_pixels": padding,
        "crop_left": crop_box[0],
        "crop_top": crop_box[1],
        "crop_right": crop_box[2],
        "crop_bottom": crop_box[3],
        "crop_width": crop.width,
        "crop_height": crop.height,
    }
