import hashlib

import pytest
from PIL import Image

from scripts.download_inaturalist_cc0_bird_media import (
    _decode_and_strip,
    medium_url,
    validate_splits,
)


def _row(split: str, observation: str, photo: str, group: str) -> dict:
    return {
        "source": "inaturalist",
        "split": split,
        "observation_id": observation,
        "taxon_id": 1,
        "label": "bird",
        "observation_license": "cc0",
        "photo_id": photo,
        "photo_license": "cc0",
        "photo_url": f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo}/square.jpg",
        "observer_group_sha256": group,
    }


def test_medium_url_is_derived_only_for_matching_https_photo_identity():
    row = _row("train", "obs-1", "1234", "group-a")
    assert medium_url(row).endswith("/photos/1234/medium.jpg")
    row["photo_url"] = "https://inaturalist-open-data.s3.amazonaws.com/photos/1234/square.png"
    assert medium_url(row).endswith("/photos/1234/medium.png")
    row["photo_url"] = "https://example.test/photos/1234/square.jpg"
    with pytest.raises(ValueError, match="expected CC0"):
        medium_url(row)
    row["photo_url"] = "https://inaturalist-open-data.s3.amazonaws.com/photos/5678/square.jpg"
    with pytest.raises(ValueError, match="expected CC0"):
        medium_url(row)


def test_split_gate_rejects_shared_observer_photo_or_missing_cc0():
    train = [_row("train", "obs-1", "1234", "group-a")]
    validation = [_row("validation", "obs-2", "5678", "group-b")]
    validate_splits(train, validation)

    validation[0]["observer_group_sha256"] = "group-a"
    with pytest.raises(ValueError, match="observer_group_sha256 overlap"):
        validate_splits(train, validation)
    validation[0]["observer_group_sha256"] = "group-b"
    validation[0]["photo_id"] = "1234"
    validation[0]["photo_url"] = train[0]["photo_url"]
    with pytest.raises(ValueError, match="photo_id overlap"):
        validate_splits(train, validation)
    validation[0]["photo_id"] = "5678"
    validation[0]["photo_url"] = "https://inaturalist-open-data.s3.amazonaws.com/photos/5678/square.jpg"
    validation[0]["photo_license"] = "cc-by"
    with pytest.raises(ValueError, match="explicitly be CC0"):
        validate_splits(train, validation)


def test_image_normalization_applies_orientation_and_strips_exif(tmp_path):
    source = tmp_path / "source.jpg"
    image = Image.new("RGB", (80, 40), color=(10, 20, 30))
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)

    normalized, metadata = _decode_and_strip(source)
    destination = tmp_path / "clean.png"
    normalized.save(destination, format="PNG")

    assert normalized.size == (40, 80)
    assert metadata["width"] == 40
    assert metadata["height"] == 80
    with Image.open(destination) as saved:
        assert not saved.getexif()
        digest = hashlib.sha256(
            f"RGB:{saved.width}x{saved.height}:".encode() + saved.tobytes()
        ).hexdigest()
    assert digest == metadata["normalized_rgb_sha256"]
