from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import download_physicalai_validation as download_script


class FakeResponse:
    def __init__(self, *, payload: bytes | None = None, body: dict | None = None):
        self.content = payload
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


def test_downloads_only_validation_payload_and_checks_publisher_hashes(tmp_path: Path, monkeypatch):
    revision = "a" * 40
    dataset_id = "nvidia/PhysicalAI-Robotics-Manipulation-SingleArm"
    subset = "panda-open-cabinet-left"
    video_path = f"{subset}/videos/chunk-000/observation.images.world_camera/episode_000002.mp4"
    parquet_path = f"{subset}/data/chunk-000/episode_000002.parquet"
    payloads = {video_path: b"video", parquet_path: b"parquet"}
    siblings = [
        {
            "rfilename": path,
            "size": len(payload),
            "lfs": {"sha256": hashlib.sha256(payload).hexdigest()},
        }
        for path, payload in payloads.items()
    ]
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": {
                    "revision": revision,
                    "dataset_id": dataset_id,
                    "subset": subset,
                },
                "episodes": [
                    {"episode_index": 2, "split": "validation", "video_path": video_path},
                    {"episode_index": 3, "split": "reserved_holdout", "video_path": "unused"},
                ],
            }
        )
    )
    requested_urls: list[str] = []

    def fake_get(url: str, **_kwargs) -> FakeResponse:
        requested_urls.append(url)
        if url.startswith("https://huggingface.co/api/"):
            return FakeResponse(body={"sha": revision, "siblings": siblings})
        path = next(path for path in payloads if path in url)
        return FakeResponse(payload=payloads[path])

    monkeypatch.setattr(download_script.requests, "get", fake_get)
    output = tmp_path / "out"

    result = download_script.download_validation(manifest_path, output, workers=1)

    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert result["files"] == 2
    assert receipt["validation_episode_indices"] == [2]
    assert receipt["reserved_holdout_payload_downloaded"] is False
    assert len(requested_urls) == 3
    assert all("episode_000003" not in url for url in requested_urls)
    for item in receipt["files"]:
        local_path = Path(item["local_path"])
        assert local_path.stat().st_size == item["size"]
        assert hashlib.sha256(local_path.read_bytes()).hexdigest() == item["sha256"]


def test_download_split_supports_train_but_rejects_reserved_before_network(tmp_path, monkeypatch):
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps({"episodes": []}))
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("reserved split must be rejected before network access")

    monkeypatch.setattr(download_script.requests, "get", fail_if_called)
    with pytest.raises(ValueError, match="only train and validation"):
        download_script.download_split(manifest_path, tmp_path / "out", split="reserved_holdout")
    assert called is False
