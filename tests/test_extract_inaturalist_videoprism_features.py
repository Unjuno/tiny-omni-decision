import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.extract_inaturalist_videoprism_features import (
    prepare_split,
    sha256_file,
    validate_feature_splits,
)


def _source_row(split: str, observation: str, photo: str, group: str) -> dict:
    return {
        "source": "inaturalist",
        "split": split,
        "observation_id": observation,
        "taxon_id": 1,
        "label": "bird",
        "observation_license": "cc0",
        "photo_id": photo,
        "photo_license": "cc0",
        "photo_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/1/square.jpg",
        "observer_group_sha256": group,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_split_checks_ids_rights_and_media_bytes(tmp_path):
    metadata_dir = tmp_path / "metadata"
    media_dir = tmp_path / "media"
    (media_dir / "train").mkdir(parents=True)
    metadata_dir.mkdir()
    image_path = media_dir / "train" / "123.png"
    Image.new("RGB", (40, 40), color="blue").save(image_path)
    source = _source_row("train", "obs-1", "123", "group-1")
    media = source | {
        "status": "saved",
        "split": "train",
        "local_path": "123.png",
        "download_sha256": "d" * 64,
        "media_sha256": sha256_file(image_path),
        "normalized_rgb_sha256": "a" * 64,
        "perceptual_hash_64": "0" * 16,
        "width": 40,
        "height": 40,
        "exif_stripped": True,
    }
    _write_jsonl(metadata_dir / "train.jsonl", [source])
    _write_jsonl(media_dir / "train-media-manifest.jsonl", [media])

    prepared = prepare_split(metadata_dir, media_dir, "train")

    assert len(prepared) == 1
    assert prepared[0]["sample_id"] == "inat:1:obs-1"
    assert Path(prepared[0]["media_path"]) == image_path.resolve()

    media["media_sha256"] = "0" * 64
    _write_jsonl(media_dir / "train-media-manifest.jsonl", [media])
    with pytest.raises(ValueError, match="frozen SHA-256"):
        prepare_split(metadata_dir, media_dir, "train")


def test_validate_feature_splits_rejects_group_and_near_duplicate_overlap():
    train = [{
        "observation_id": "obs-1",
        "photo_id": "photo-1",
        "observer_group_sha256": "group-1",
        "download_sha256": "d" * 64,
        "media_sha256": "m" * 64,
        "normalized_rgb_sha256": "r" * 64,
        "perceptual_hash_64": "0000000000000000",
    }]
    validation = [{
        "observation_id": "obs-2",
        "photo_id": "photo-2",
        "observer_group_sha256": "group-2",
        "download_sha256": "e" * 64,
        "media_sha256": "n" * 64,
        "normalized_rgb_sha256": "s" * 64,
        "perceptual_hash_64": "ffffffffffffffff",
    }]
    validate_feature_splits(train, validation)

    validation[0]["observer_group_sha256"] = "group-1"
    with pytest.raises(ValueError, match="observer_group_sha256 overlap"):
        validate_feature_splits(train, validation)
    validation[0]["observer_group_sha256"] = "group-2"
    validation[0]["perceptual_hash_64"] = "0000000000000001"
    with pytest.raises(ValueError, match="near-duplicate"):
        validate_feature_splits(train, validation)
