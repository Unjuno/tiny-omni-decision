from __future__ import annotations

import hashlib
import json
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.audit_clevr4_content_overlap import (
    normalized_rgb64_sha256,
    phash_bytes,
    phash_distances,
    run_audit,
)
from scripts.audit_clevr4_near_pair_pixel_distance import run_audit as run_near_pair_audit
from scripts.audit_clevr4_scene_config_overlap import (
    _StreamingJSON,
    scene_fingerprint,
)
from scripts.audit_clevr4_scene_config_overlap import (
    run_audit as run_scene_audit,
)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (48, 48), "white")
    ImageDraw.Draw(image).rectangle((8, 8, 34, 34), fill="red")
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_phash_is_repeatable_and_returns_63_bits():
    value = phash_bytes(_png_bytes())
    assert value == phash_bytes(_png_bytes())
    assert 0 <= value < 2**63


def test_phash_distance_is_zero_for_same_image():
    value = phash_bytes(_png_bytes())
    distances = phash_distances(value, np.asarray([value], dtype=np.uint64))
    assert distances.tolist() == [0]


def test_normalized_pixel_hash_is_repeatable():
    image = _png_bytes()
    assert normalized_rgb64_sha256(image) == normalized_rgb64_sha256(image)


def test_run_audit_resolves_prior_manifests_and_records_hashes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image_bytes = _png_bytes()
    archive = source / "candidate.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("images/item.png", image_bytes)
    candidate_manifest = source / "candidate.jsonl"
    candidate_manifest.write_text(
        json.dumps(
            {
                "id": "candidate-1",
                "archive_member": "images/item.png",
                "media_sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    previous_root = tmp_path / "previous"
    train_images = previous_root / "train" / "images"
    validation_images = previous_root / "val" / "images"
    train_images.mkdir(parents=True)
    validation_images.mkdir(parents=True)
    (train_images / "old-train.png").write_bytes(image_bytes)
    (validation_images / "old-val.png").write_bytes(image_bytes)
    (previous_root / "train-image-manifest.jsonl").write_text("{}\n", encoding="utf-8")
    (previous_root / "val-image-manifest.jsonl").write_text("{}\n", encoding="utf-8")

    result = run_audit(
        archive=archive,
        candidate_manifest=candidate_manifest,
        previous_train=train_images,
        previous_validation=validation_images,
        output=tmp_path / "result",
        expected_archive_bytes=archive.stat().st_size,
        expected_archive_sha512="pinned-in-fixture",
    )

    assert result["candidate_count"] == 1
    assert result["candidate_old_media_sha256_overlap"] == 1
    assert result["phash"]["nearest_hamming_min"] == 0
    assert Path(tmp_path / "result" / "result.json").is_file()


def test_near_pair_audit_reports_rgb_and_edge_distance(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image_bytes = _png_bytes()
    archive = source / "candidate.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("images/item.png", image_bytes)
    candidate_manifest = source / "candidate.jsonl"
    candidate_manifest.write_text(
        json.dumps(
            {
                "id": "candidate-1",
                "archive_member": "images/item.png",
                "media_sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_images = tmp_path / "old" / "train" / "images"
    validation_images = tmp_path / "old" / "val" / "images"
    train_images.mkdir(parents=True)
    validation_images.mkdir(parents=True)
    (train_images / "old-train.png").write_bytes(image_bytes)
    (validation_images / "old-val.png").write_bytes(image_bytes)

    result = run_near_pair_audit(
        archive=archive,
        candidate_manifest=candidate_manifest,
        old_train_images=train_images,
        old_validation_images=validation_images,
        output=tmp_path / "near-result",
    )

    assert result["candidate_with_any_reference_within_threshold"] == 1
    pair = result["near_pairs"][0]
    assert pair["rgb64_mean_absolute_error_0_255"] == 0
    assert pair["edge64_mean_absolute_error_0_255"] == 0


def test_streaming_scene_reader_skips_unselected_nested_values():
    source = json.dumps({"ignored": {"text": '] } { "', "items": [1, 2]}, "keep": {"objects": []}})
    result = list(_StreamingJSON(StringIO(source)).iter_object({"keep"}))
    assert result == [("keep", {"objects": []})]


def test_scene_fingerprint_ignores_non_scene_source_fields():
    left = {"image_index": 1, "split": "train", "objects": [{"color": "red"}]}
    right = {"image_index": 2, "split": "val", "objects": [{"color": "red"}]}
    assert scene_fingerprint(left) == scene_fingerprint(right)


def test_scene_audit_reports_cross_split_exact_duplicate(tmp_path):
    archive = tmp_path / "scenes.zip"
    scene = {"objects": [{"shape": "cube", "x": 1.25}]}
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "whole_scene_data.json",
            json.dumps({"train-a": scene, "val-a": scene, "official-val": {"secret": True}}),
        )
    train_labels = tmp_path / "train.jsonl"
    validation_labels = tmp_path / "validation.jsonl"
    train_labels.write_text(
        json.dumps({"id": "train-a", "source_split": "train"}) + "\n",
        encoding="utf-8",
    )
    validation_labels.write_text(
        json.dumps({"id": "val-a", "source_split": "train"}) + "\n",
        encoding="utf-8",
    )
    train_media_manifest = tmp_path / "train-media.jsonl"
    validation_media_manifest = tmp_path / "validation-media.jsonl"
    train_media_manifest.write_text(json.dumps({"id": "train-a"}) + "\n", encoding="utf-8")
    validation_media_manifest.write_text(json.dumps({"id": "val-a"}) + "\n", encoding="utf-8")
    result = run_scene_audit(
        archive=archive,
        train_labels=train_labels,
        validation_labels=validation_labels,
        train_media_manifest=train_media_manifest,
        validation_media_manifest=validation_media_manifest,
        output=tmp_path / "scene-result",
    )
    assert result["exact_scene_config_cross_split_group_count"] == 1
    assert result["exact_scene_config_cross_split_asset_count"] == 2
