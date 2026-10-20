from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest

import scripts.fetch_clevrer_synthetic_av_media as fetcher
from scripts.create_clevrer_synthetic_av_protocol import DATASET, REVISION


def _protocol() -> dict[str, object]:
    scenes = list(range(405))
    return {
        "protocol_id": "clevrer-synthetic-av-timing-v1",
        "created_from": {"source_dataset": DATASET, "source_revision": REVISION},
        "isolation": {"validation_test_or_sealed_data_opened": False},
        "split": {"scene_ids": {"train": scenes[:324], "validation": scenes[324:]}},
    }


def test_selected_scenes_rejects_split_overlap_and_nonfrozen_data() -> None:
    protocol = _protocol()
    split = protocol["split"]["scene_ids"]  # type: ignore[index]
    split["validation"] = split["train"][:1] + split["validation"][1:]  # type: ignore[index]

    with pytest.raises(ValueError, match="overlap"):
        fetcher._selected_scenes(protocol)


def test_fetch_media_checks_hashes_writes_new_hashed_train_video_set(tmp_path, monkeypatch) -> None:
    protocol = _protocol()
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode("utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(protocol_bytes)
    monkeypatch.setattr(fetcher, "PROTOCOL_SHA256", hashlib.sha256(protocol_bytes).hexdigest())
    monkeypatch.setattr(
        fetcher.shutil, "disk_usage", lambda _path: type("Usage", (), {"free": 4_000_000_000})()
    )

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_STORED) as archive:
        for scene in range(405):
            payload = b"\x00\x00\x00\x18ftypisom" + scene.to_bytes(4, "big")
            archive.writestr(f"videos/video_{scene:05d}.mp4", payload)
    archive_bytes.seek(0)

    output = tmp_path / "media-output"
    receipt = fetcher.fetch_media(protocol_path, output, archive_file=archive_bytes)

    assert receipt["scene_count"] == 405
    assert receipt["train_scene_count"] == 324
    assert receipt["validation_scene_count"] == 81
    assert receipt["payload_bytes"] > 0
    assert receipt["media_sha256_overlap"] == 0
    assert len(list((output / "media").glob("*.mp4"))) == 405
    assert (output / "media-manifest.jsonl").is_file()
    assert (output / "receipt.json").is_file()


def test_fetch_media_fails_closed_on_existing_output(tmp_path, monkeypatch) -> None:
    protocol = _protocol()
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode("utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(protocol_bytes)
    monkeypatch.setattr(fetcher, "PROTOCOL_SHA256", hashlib.sha256(protocol_bytes).hexdigest())
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        fetcher.fetch_media(protocol_path, output)
