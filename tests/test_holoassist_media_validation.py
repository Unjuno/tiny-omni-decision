from __future__ import annotations

import hashlib
import sys
from fractions import Fraction
from types import SimpleNamespace

import pytest

from scripts.validate_holoassist_selected_media import validate_selected_media


def media_facts(
    monkeypatch, *, video_duration: float = 3.25, audio_duration: float = 3.25
):
    def stream(kind: str, duration: float):
        return SimpleNamespace(
            type=kind,
            start_time=0,
            duration=int(duration * 1000),
            time_base=Fraction(1, 1000),
            average_rate=Fraction(30, 1) if kind == "video" else None,
            frames=94 if kind == "video" else 153,
            rate=48_000 if kind == "audio" else None,
            codec_context=SimpleNamespace(
                channels=1 if kind == "audio" else None,
                frame_size=1024 if kind == "audio" else None,
                name="aac" if kind == "audio" else "h264",
                width=640 if kind == "video" else None,
                height=360 if kind == "video" else None,
            ),
        )

    class Container:
        streams = [stream("video", video_duration), stream("audio", audio_duration)]

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _path: Container()))


def fixture(tmp_path):
    selection = {
        "selection": {
            "train": {"videos": [{"session_id": "tr-1", "size_bytes": 7}]},
            "validation": {"videos": [{"session_id": "va-1", "size_bytes": 7}]},
        },
    }
    files = []
    annotations = []
    for split, session in (("train", "tr-1"), ("validation", "va-1")):
        path = tmp_path / split / f"{session}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"media!!")
        files.append(
            {
                "split": split,
                "session_id": session,
                "size_bytes": 7,
                "sha256": hashlib.sha256(b"media!!").hexdigest(),
                "archive_etag": '"pinned"',
            }
        )
        annotations.append(
            {
                "video_name": session,
                "videoMetadata": {"duration": {"raw": "00:00:03.25"}},
            }
        )
    return (
        selection,
        {"selection_sha256": "selection", "archive_etag": '"pinned"', "files": files},
        annotations,
    )


def test_media_validation_requires_av_timing_and_exact_selected_ids(tmp_path, monkeypatch):
    media_facts(monkeypatch)
    selection, manifest, annotations = fixture(tmp_path)

    report = validate_selected_media(
        selection=selection,
        media_manifest=manifest,
        annotations=annotations,
        media_root=tmp_path,
        expected_archive_etag='"pinned"',
        selection_sha256="selection",
        media_manifest_sha256="manifest",
    )

    assert report["status"] == "PASS_HOLOASSIST_TRAINVAL_AV_MEDIA"
    assert report["train_session_count"] == 1
    assert report["validation_session_count"] == 1
    assert len(report["media"]) == 2


def test_media_validation_rejects_av_duration_mismatch(tmp_path, monkeypatch):
    media_facts(monkeypatch, audio_duration=3.5)
    selection, manifest, annotations = fixture(tmp_path)

    with pytest.raises(ValueError, match="audio/video duration mismatch"):
        validate_selected_media(
            selection=selection,
            media_manifest=manifest,
            annotations=annotations,
            media_root=tmp_path,
            expected_archive_etag='"pinned"',
            selection_sha256="selection",
            media_manifest_sha256="manifest",
        )


def test_media_validation_allows_two_video_frame_endpoint_difference(tmp_path, monkeypatch):
    media_facts(monkeypatch, audio_duration=3.314)
    selection, manifest, annotations = fixture(tmp_path)

    report = validate_selected_media(
        selection=selection,
        media_manifest=manifest,
        annotations=annotations,
        media_root=tmp_path,
        expected_archive_etag='"pinned"',
        selection_sha256="selection",
        media_manifest_sha256="manifest",
    )

    assert report["status"] == "PASS_HOLOASSIST_TRAINVAL_AV_MEDIA"
    assert report["media"][0]["audio_video_duration_difference_seconds"] == pytest.approx(0.064)
    assert report["media"][0]["audio_video_duration_tolerance_seconds"] == pytest.approx(
        2 / 30 + 1024 / 48_000
    )


def test_media_validation_allows_aac_packet_rounding_after_two_video_frames(
    tmp_path, monkeypatch
):
    media_facts(monkeypatch, audio_duration=3.318)
    selection, manifest, annotations = fixture(tmp_path)

    report = validate_selected_media(
        selection=selection,
        media_manifest=manifest,
        annotations=annotations,
        media_root=tmp_path,
        expected_archive_etag='"pinned"',
        selection_sha256="selection",
        media_manifest_sha256="manifest",
    )

    assert report["status"] == "PASS_HOLOASSIST_TRAINVAL_AV_MEDIA"
    assert report["media"][0]["audio_packet_interval_seconds"] == pytest.approx(1024 / 48_000)


def test_media_validation_compares_annotation_duration_to_audio_endpoint(tmp_path, monkeypatch):
    media_facts(monkeypatch, video_duration=3.2, audio_duration=3.25)
    selection, manifest, annotations = fixture(tmp_path)

    report = validate_selected_media(
        selection=selection,
        media_manifest=manifest,
        annotations=annotations,
        media_root=tmp_path,
        expected_archive_etag='"pinned"',
        selection_sha256="selection",
        media_manifest_sha256="manifest",
    )

    assert report["media"][0]["video_duration_seconds"] == pytest.approx(3.2)
    assert report["media"][0]["audio_duration_seconds"] == pytest.approx(3.25)


def test_media_validation_rejects_annotation_duration_mismatch(tmp_path, monkeypatch):
    media_facts(monkeypatch)
    selection, manifest, annotations = fixture(tmp_path)
    annotations[0]["videoMetadata"]["duration"]["raw"] = "00:00:03.00"

    with pytest.raises(ValueError, match="audio/annotation duration mismatch"):
        validate_selected_media(
            selection=selection,
            media_manifest=manifest,
            annotations=annotations,
            media_root=tmp_path,
            expected_archive_etag='"pinned"',
            selection_sha256="selection",
            media_manifest_sha256="manifest",
        )


def test_media_validation_rejects_manifest_for_other_selection(tmp_path, monkeypatch):
    media_facts(monkeypatch)
    selection, manifest, annotations = fixture(tmp_path)
    manifest["selection_sha256"] = "other-selection"

    with pytest.raises(ValueError, match="different immutable selection"):
        validate_selected_media(
            selection=selection,
            media_manifest=manifest,
            annotations=annotations,
            media_root=tmp_path,
            expected_archive_etag='"pinned"',
            selection_sha256="selection",
            media_manifest_sha256="manifest",
        )
