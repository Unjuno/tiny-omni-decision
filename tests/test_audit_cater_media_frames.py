from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest

from scripts.audit_cater_media_frames import audit_media


class _FakeContainer:
    def __init__(self, decoded: int, declared: int | None = None) -> None:
        expected = decoded if declared is None else declared
        self.streams = [
            SimpleNamespace(type="video", frames=expected, average_rate="24", duration=expected)
        ]
        self._frames = decoded

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def decode(self, _stream):
        return iter(SimpleNamespace(pts=index) for index in range(self._frames))


def test_audit_media_decodes_and_reports_exact_frame_count(tmp_path, monkeypatch) -> None:
    payload = b"pinned-media"
    path = tmp_path / "clip.avi"
    path.write_bytes(payload)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _path: _FakeContainer(4)))
    record = {
        "split": "train",
        "video_id": "clip.avi",
        "relative_path": "clip.avi",
        "media_bytes": len(payload),
        "media_sha256": hashlib.sha256(payload).hexdigest(),
    }

    result = audit_media(tmp_path, [record])

    assert result[0]["stream_frames"] == 4
    assert result[0]["decoded_frames"] == 4
    assert result[0]["first_pts"] == 0
    assert result[0]["last_pts"] == 3


def test_audit_media_rejects_declared_decoded_frame_mismatch(tmp_path, monkeypatch) -> None:
    payload = b"pinned-media"
    path = tmp_path / "clip.avi"
    path.write_bytes(payload)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _path: _FakeContainer(4, 5)))
    record = {
        "split": "train",
        "video_id": "clip.avi",
        "relative_path": "clip.avi",
        "media_bytes": len(payload),
        "media_sha256": hashlib.sha256(payload).hexdigest(),
    }

    with pytest.raises(ValueError, match="declared/decoded frame mismatch"):
        audit_media(tmp_path, [record])
