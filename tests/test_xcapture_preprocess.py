from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tiny_omni_decision.xcapture_preprocess import masked_object_crop


def _write_fixture(tmp_path: Path, *, mask_size: tuple[int, int] = (20, 12)) -> tuple[Path, Path]:
    image = Image.new("RGB", (20, 12), (255, 0, 0))
    mask = Image.new("L", mask_size, 0)
    ImageDraw.Draw(mask).rectangle((5, 3, 14, 8), fill=255)
    image_path, mask_path = tmp_path / "rgb.png", tmp_path / "mask.png"
    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def test_masked_object_crop_zeros_background_and_keeps_foreground(tmp_path: Path) -> None:
    image_path, mask_path = _write_fixture(tmp_path)

    crop, audit = masked_object_crop(image_path, mask_path, padding_fraction=0.0)

    assert crop.size == (10, 6)
    assert crop.getpixel((0, 0)) == (255, 0, 0)
    assert crop.getpixel((4, 2)) == (255, 0, 0)
    assert audit["mask_pixels"] == 60
    assert audit["mask_coverage"] == pytest.approx(0.25)


def test_masked_object_crop_rejects_misaligned_mask(tmp_path: Path) -> None:
    image_path, mask_path = _write_fixture(tmp_path, mask_size=(10, 12))

    with pytest.raises(ValueError, match="dimensions differ"):
        masked_object_crop(image_path, mask_path)


def test_masked_object_crop_rejects_nonbinary_mask(tmp_path: Path) -> None:
    image_path, mask_path = _write_fixture(tmp_path)
    with Image.open(mask_path) as source_mask:
        mask = source_mask.convert("L")
    ImageDraw.Draw(mask).point((0, 0), fill=127)
    mask.save(mask_path)

    with pytest.raises(ValueError, match="binary 0/255"):
        masked_object_crop(image_path, mask_path)


def test_masked_object_crop_rejects_empty_and_unreasonable_coverage(tmp_path: Path) -> None:
    image_path, mask_path = _write_fixture(tmp_path)
    Image.new("L", (20, 12), 0).save(mask_path)

    with pytest.raises(ValueError, match="coverage"):
        masked_object_crop(image_path, mask_path)


def test_masked_object_crop_allows_dataset_reviewed_large_object_mask(tmp_path: Path) -> None:
    image_path, mask_path = _write_fixture(tmp_path)
    mask = Image.new("L", (20, 12), 255)
    ImageDraw.Draw(mask).rectangle((0, 0, 9, 0), fill=0)
    mask.save(mask_path)

    with pytest.raises(ValueError, match="coverage"):
        masked_object_crop(image_path, mask_path)

    _crop, audit = masked_object_crop(
        image_path, mask_path, maximum_mask_coverage=0.98
    )

    assert audit["mask_coverage"] == pytest.approx(230 / 240)


def test_masked_object_crop_allows_reviewed_small_contact_region(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), (40, 80, 120))
    mask = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(mask).rectangle((1, 1, 8, 10), fill=255)
    image_path, mask_path = tmp_path / "small-rgb.png", tmp_path / "small-mask.png"
    image.save(image_path)
    mask.save(mask_path)

    with pytest.raises(ValueError, match="coverage"):
        masked_object_crop(image_path, mask_path)

    _crop, audit = masked_object_crop(
        image_path,
        mask_path,
        minimum_mask_coverage=0.003,
        maximum_mask_coverage=0.98,
    )

    assert audit["mask_coverage"] == pytest.approx(0.008)


def test_masked_object_crop_requires_valid_padding(tmp_path: Path) -> None:
    image_path, mask_path = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="padding_fraction"):
        masked_object_crop(image_path, mask_path, padding_fraction=0.8)
