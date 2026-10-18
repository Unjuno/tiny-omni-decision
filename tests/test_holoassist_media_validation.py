from __future__ import annotations

import hashlib
import sys
from fractions import Fraction
from types import SimpleNamespace

import pytest

from scripts.validate_holoassist_selected_media import validate_selected_media


def media_facts(monkeypatch, *, audio_duration: float = 3.25):
    def stream(kind: str, duration: float):
        return SimpleNamespace(
            type=kind,
            start_time=0,
            duration=int(duration * 1000),
            time_base=Fraction(1, 1000),
            frames=94 if kind == "video" else 153,
            rate=48_000 if kind == "audio" else None,
            codec_context=SimpleNamespace(
                channels=1 if kind == "audio" else None,
                name="aac" if kind == "audio" else "h264",
                width=640 if kind == "video" else None,
                height=360 if kind == "video" else None,
            ),
        )

    class Container:
        streams = [stream("video", 3.25), stream("audio", audio_duration)]

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _path: Container()))


def fixture(tmp_path):
    selection = {
        "archive_etag": '"pinned"',
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
    return selection, {"selection_sha256": "selection", "files": files}, annotations


def test_media_validation_requires_av_timing_and_exact_selected_ids(tmp_path, monkeypatch):
    media_facts(monkeypatch)
    selection, manifest, annotations = fixture(tmp_path)

    report = validate_selected_media(
        selection=selection,
        media_manifest=manifest,
        annotations=annotations,
        media_root=tmp_path,
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
            selection_sha256="selection",
            media_manifest_sha256="manifest",
        )
