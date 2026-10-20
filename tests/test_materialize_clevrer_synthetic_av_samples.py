from __future__ import annotations

import hashlib
import json

import pytest

import scripts.materialize_clevrer_synthetic_av_samples as materializer
from scripts.create_clevrer_synthetic_av_protocol import DATASET, REVISION


def _write_fixture(tmp_path, monkeypatch):
    scenes = {"train": [1], "validation": [2]}
    samples = {"train": [], "validation": []}
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    entries = []
    for split, scene in (("train", 1), ("validation", 2)):
        media_name = f"video_{scene:05d}.mp4"
        media_bytes = f"synthetic-mp4-{scene}".encode()
        (media_dir / media_name).write_bytes(media_bytes)
        entries.append(
            {
                "scene_index": scene,
                "split": split,
                "path": f"media/{media_name}",
                "bytes": len(media_bytes),
                "sha256": hashlib.sha256(media_bytes).hexdigest(),
            }
        )
        pair_id = f"scene-{scene}-event-0"
        for target in (0, 1):
            samples[split].append(
                {
                    "sample_id": f"{pair_id}-{target}",
                    "pair_id": pair_id,
                    "event_id": pair_id,
                    "scene_index": scene,
                    "video_filename": media_name,
                    "question": "Did the sound precede the collision?",
                    "options": ["before", "after"],
                    "target": target,
                    "relation": "before" if target == 0 else "after",
                    "collision_frame": 50,
                    "object_ids": [0, 1],
                    "object_descriptors": [],
                    "scene_time_shift_frames": 12,
                    "video_start_frame": 12,
                    "video_end_frame": 139,
                    "sound_onset_frame": 34 + target * 32,
                    "sound_onset_sample": (34 + target * 32) * 625,
                    "split": split,
                }
            )
    protocol = {
        "protocol_id": "clevrer-synthetic-av-timing-v1",
        "created_from": {"source_dataset": DATASET, "source_revision": REVISION},
        "isolation": {"validation_test_or_sealed_data_opened": False},
        "split": {
            "scene_ids": scenes,
            "sample_id_order_sha256": {
                split: hashlib.sha256(
                    "\n".join(sorted(row["sample_id"] for row in rows)).encode()
                ).hexdigest()
                for split, rows in samples.items()
            },
        },
        "samples": samples,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode()
    protocol_path.write_bytes(protocol_bytes)
    monkeypatch.setattr(materializer, "PROTOCOL_SHA256", hashlib.sha256(protocol_bytes).hexdigest())
    receipt = {
        "status": "complete_train_only_media_fetch",
        "dataset": DATASET,
        "revision": REVISION,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "scene_count": 2,
        "train_scene_count": 1,
        "validation_scene_count": 1,
        "scene_overlap": 0,
        "media_sha256_overlap": 0,
        "validation_test_or_sealed_data_opened": False,
        "entries": entries,
    }
    receipt_path = tmp_path / "media-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return protocol_path, receipt_path, media_dir.parent


def test_materializer_checks_assets_and_emits_disjoint_paired_manifests(tmp_path, monkeypatch):
    protocol_path, receipt_path, output_dir = _write_fixture(tmp_path, monkeypatch)

    report = materializer.materialize_samples(protocol_path, receipt_path, output_dir)

    train = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
    validation = [
        json.loads(line) for line in (output_dir / "validation.jsonl").read_text().splitlines()
    ]
    assert report["train_samples"] == 2
    assert report["validation_samples"] == 2
    assert {row["scene_index"] for row in train}.isdisjoint(
        {row["scene_index"] for row in validation}
    )
    assert {row["target"] for row in train} == {"before", "after"}
    assert all(row["media_sha256"] for row in train + validation)
    assert report["media_hash_overlap"] == 0


def test_materializer_rejects_missing_or_hash_mismatched_media(tmp_path, monkeypatch):
    protocol_path, receipt_path, output_dir = _write_fixture(tmp_path, monkeypatch)
    media_path = output_dir / "media" / "video_00001.mp4"
    media_path.write_bytes(b"changed bytes")

    with pytest.raises(ValueError, match="media file hash mismatch"):
        materializer.materialize_samples(protocol_path, receipt_path, output_dir)
