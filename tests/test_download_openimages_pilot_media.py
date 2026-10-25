import base64
from email.message import Message
from pathlib import Path

import pytest
from PIL import Image

from scripts.download_openimages_pilot_media import (
    audit_cross_split_duplicates,
    download_one,
    perceptual_hash,
    prepare_download_records,
)


def _receipt(image_id, split, *, match=True):
    return {
        "ImageID": image_id,
        "Subset": split,
        "landing_page_check_status": (
            "metadata_confirmed_on_landing_page" if match else "license_mismatch"
        ),
        "Thumbnail300KURL": f"https://c1.staticflickr.com/1/{image_id}_abc_n.jpg",
        "OriginalURL": f"https://farm1.staticflickr.com/1/{image_id}_abc.jpg",
        "OriginalLandingURL": f"https://www.flickr.com/photos/person/{image_id}",
        "OriginalMD5": base64.b64encode(bytes.fromhex("00" * 16)).decode("ascii"),
        "landing_page_metadata": {"license_url": "https://creativecommons.org/licenses/by/2.0/"},
    }


def _task(image_id, split):
    return {
        "sample_id": f"{split}-{image_id}",
        "image_id": image_id,
        "target_label": "/cat",
        "target": "Cat",
        "options": ["Cat", "Dog", "Bird", "Horse"],
        "target_position": 0,
    }


def test_prepare_records_requires_page_match_head_success_and_same_photo_id():
    records = prepare_download_records(
        split="train",
        task_manifest={"split": "train", "examples": [_task("123", "train")]},
        page_receipts=[_receipt("123", "train"), _receipt("456", "train", match=False)],
        head_records=[
            {
                "split": "train",
                "image_id": "123",
                "status": "ok",
                "content_type": "image/jpeg",
            },
        ],
    )

    assert len(records) == 1
    assert records[0]["image_id"] == "123"
    assert records[0]["individual_rights_verified"] is False


def test_prepare_records_rejects_wrong_photo_id_in_media_url():
    receipt = _receipt("123", "train")
    receipt["Thumbnail300KURL"] = "https://c1.staticflickr.com/1/999_abc_n.jpg"
    with pytest.raises(ValueError, match="does not identify"):
        prepare_download_records(
            split="train",
            task_manifest={"split": "train", "examples": [_task("123", "train")]},
            page_receipts=[receipt],
            head_records=[
                {
                    "split": "train",
                    "image_id": "123",
                    "status": "ok",
                    "content_type": "image/jpeg",
                }
            ],
        )


def test_perceptual_hash_is_stable_for_identical_images():
    image = Image.new("RGB", (40, 30), (12, 90, 170))
    assert perceptual_hash(image) == perceptual_hash(image.copy())


def test_cross_split_audit_finds_exact_and_near_perceptual_matches():
    rows = [
        {
            "split": "train",
            "status": "saved",
            "image_id": "a",
            "normalized_rgb_sha256": "same",
            "perceptual_hash_64": "0000000000000000",
        },
        {
            "split": "validation",
            "status": "saved",
            "image_id": "b",
            "normalized_rgb_sha256": "same",
            "perceptual_hash_64": "0000000000000001",
        },
    ]

    result = audit_cross_split_duplicates(rows)

    assert result["train_validation_normalized_rgb_overlap"] == 1
    assert result["train_validation_exact_phash_overlap"] == 0
    assert result["train_validation_phash_hamming_at_most_4"][0]["hamming_distance"] == 1
    assert result["individual_rights_verified"] is False


def test_download_rejects_unavailable_or_non_jpeg_media(tmp_path: Path, monkeypatch):
    class FakeResponse:
        status = 200
        headers = Message()
        headers["Content-Type"] = "text/html"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://c1.staticflickr.com/1/123_abc_n.jpg"

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        "scripts.download_openimages_pilot_media.build_opener", lambda *_: FakeOpener()
    )
    result = download_one(
        {
            "image_id": "123",
            "url": "https://c1.staticflickr.com/1/123_abc_n.jpg",
            "split": "train",
        },
        split_dir=tmp_path,
    )

    assert result["status"] == "failed_ValueError"
    assert list(tmp_path.glob("*.jpg")) == []
