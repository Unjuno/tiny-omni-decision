from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from scripts import download_physicalai_stack_platforms_pilot as downloader


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ):
        self.content = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise downloader.requests.HTTPError(f"HTTP {self.status_code}")
        return None


def make_manifest(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    revision = "a" * 40
    payloads = {
        "subset/videos/train.mp4": b"train video",
        "subset/data/train.parquet": b"train parquet",
        "subset/videos/validation.mp4": b"validation video",
        "subset/data/validation.parquet": b"validation parquet",
        "subset/videos/reserved.mp4": b"reserved video",
        "subset/data/reserved.parquet": b"reserved parquet",
    }

    def episode(index: int, split: str, prefix: str) -> dict[str, object]:
        video = f"subset/videos/{prefix}.mp4"
        parquet = f"subset/data/{prefix}.parquet"
        return {
            "episode_index": index,
            "split": split,
            "video_path": video,
            "video_bytes": len(payloads[video]),
            "video_sha256": hashlib.sha256(payloads[video]).hexdigest(),
            "parquet_path": parquet,
            "parquet_bytes": len(payloads[parquet]),
            "parquet_sha256": hashlib.sha256(payloads[parquet]).hexdigest(),
        }

    manifest = {
        "source": {
            "dataset_id": "nvidia/PhysicalAI-Robotics-Manipulation-SingleArm",
            "revision": revision,
            "subset": "panda-stack-platforms",
        },
        "pilot_selection": {
            "train_episode_indices": [11],
            "validation_episode_indices": [22],
            "train_episode_id_order_sha256": "train-order",
            "validation_episode_id_order_sha256": "validation-order",
        },
        "split_policy": {"allocation": "train/validation/reserved_holdout"},
        "grouping": {"scene_or_layout_identity_status": "UNKNOWN"},
        "episode_assignments": [
            episode(11, "train", "train"),
            episode(22, "validation", "validation"),
            episode(33, "reserved_holdout", "reserved"),
        ],
    }
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, payloads


def test_downloads_only_selected_train_validation_and_reuses_verified_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, payloads = make_manifest(tmp_path)
    requested_urls: list[str] = []

    def fake_get(url: str, **_kwargs: object) -> FakeResponse:
        requested_urls.append(url)
        match = next(path for path in payloads if path in url)
        return FakeResponse(payloads[match])

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    output = tmp_path / "pilot"

    result = downloader.download_pilot(manifest_path, output, workers=2)

    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert result["files"] == 4
    assert result["train_episode_count"] == 1
    assert result["validation_episode_count"] == 1
    assert receipt["reserved_holdout_payload_downloaded"] is False
    assert len(requested_urls) == 4
    assert all("reserved" not in url for url in requested_urls)
    assert all("/" + "a" * 40 + "/" in url for url in requested_urls)

    downloader.download_pilot(manifest_path, output, workers=2)
    assert len(requested_urls) == 4


def test_cached_huggingface_token_is_used_only_as_request_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path, payloads = make_manifest(tmp_path)
    token = "test-secret-token"
    hub = types.ModuleType("huggingface_hub")
    hub.get_token = lambda: token  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    observed_headers: list[dict[str, str] | None] = []
    requested_urls: list[str] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        requested_urls.append(url)
        observed_headers.append(kwargs.get("headers"))  # type: ignore[arg-type]
        match = next(path for path in payloads if path in url)
        return FakeResponse(payloads[match])

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    output = tmp_path / "authenticated-pilot"

    result = downloader.download_pilot(manifest_path, output, workers=1)

    assert observed_headers == [{"Authorization": f"Bearer {token}"}] * 4
    receipt = Path(result["receipt_path"]).read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert token not in receipt
    assert token not in captured.out
    assert token not in captured.err
    assert all("reserved" not in url for url in requested_urls)
    assert all("/" + "a" * 40 + "/" in url for url in requested_urls)


def test_download_fails_closed_on_publisher_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _payloads = make_manifest(tmp_path)
    monkeypatch.setattr(
        downloader.requests,
        "get",
        lambda *_args, **_kwargs: FakeResponse(b"tampered"),
    )

    with pytest.raises(ValueError, match="publisher size/hash mismatch"):
        downloader.download_pilot(manifest_path, tmp_path / "pilot", workers=1)

    assert list((tmp_path / "pilot").glob("*.mp4")) == []
    assert list((tmp_path / "pilot").glob("*.parquet")) == []


def test_rate_limit_retry_respects_retry_after_and_completes(tmp_path, monkeypatch):
    manifest_path, payloads = make_manifest(tmp_path)
    retries: dict[str, int] = {}
    delays: list[float] = []

    def fake_get(url: str, **_kwargs: object) -> FakeResponse:
        match = next(path for path in payloads if path in url)
        retries[match] = retries.get(match, 0) + 1
        if retries[match] == 1:
            return FakeResponse(
                b"", status_code=429, headers={"Retry-After": "0.25"}
            )
        return FakeResponse(payloads[match])

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "sleep", delays.append)

    result = downloader.download_pilot(
        manifest_path, tmp_path / "pilot", workers=2, max_retries=2
    )

    assert result["files"] == 4
    assert len(retries) == 4
    assert all(count == 2 for count in retries.values())
    assert delays == [0.25] * 4


def test_http_499_retry_respects_retry_after_and_completes(tmp_path, monkeypatch):
    manifest_path, payloads = make_manifest(tmp_path)
    retries: dict[str, int] = {}
    delays: list[float] = []

    def fake_get(url: str, **_kwargs: object) -> FakeResponse:
        match = next(path for path in payloads if path in url)
        retries[match] = retries.get(match, 0) + 1
        if retries[match] == 1:
            return FakeResponse(b"", status_code=499, headers={"Retry-After": "0.25"})
        return FakeResponse(payloads[match])

    monkeypatch.setattr(downloader.requests, "get", fake_get)
    monkeypatch.setattr(downloader.time, "sleep", delays.append)

    result = downloader.download_pilot(
        manifest_path, tmp_path / "pilot", workers=1, max_retries=1
    )

    assert result["files"] == 4
    assert len(retries) == 4
    assert all(count == 2 for count in retries.values())
    assert delays == [0.25] * 4


def test_nonretryable_http_404_fails_without_retry(tmp_path, monkeypatch):
    manifest_path, payloads = make_manifest(tmp_path)
    calls: dict[str, int] = {}

    def fake_get(url: str, **_kwargs: object) -> FakeResponse:
        match = next(path for path in payloads if path in url)
        calls[match] = calls.get(match, 0) + 1
        return FakeResponse(b"", status_code=404)

    monkeypatch.setattr(downloader.requests, "get", fake_get)

    with pytest.raises(downloader.requests.HTTPError, match="HTTP 404"):
        downloader.download_pilot(
            manifest_path, tmp_path / "pilot", workers=1, max_retries=3
        )

    assert calls
    assert all(count == 1 for count in calls.values())
    assert list((tmp_path / "pilot").glob("*.mp4")) == []
    assert list((tmp_path / "pilot").glob("*.parquet")) == []


def test_retry_after_exponential_fallback_is_bounded(monkeypatch):
    response = FakeResponse(b"", status_code=429)
    assert downloader._retry_after_seconds(response, 0) == 2.0
    assert downloader._retry_after_seconds(response, 3) == 16.0
    assert downloader._retry_after_seconds(response, 20) == 120.0


def test_duplicate_or_overlapping_selection_is_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _payloads = make_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pilot_selection"]["train_episode_indices"] = [11, 11]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    requested = False

    def fail_if_requested(*_args: object, **_kwargs: object) -> FakeResponse:
        nonlocal requested
        requested = True
        raise AssertionError("invalid selections must fail before network access")

    monkeypatch.setattr(downloader.requests, "get", fail_if_requested)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        downloader.download_pilot(manifest_path, tmp_path / "pilot", workers=1)
    assert requested is False
