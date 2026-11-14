from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

from scripts.audit_cater_media_frames import audit_media


class _FakeContainer:
    def __init__(self, decoded: int, declared: int | None = None) -> None:
        expected = decoded if declared is None else declared
        self.streams = [
            SimpleNamespace(
                type="video", frames=expected, average_rate="24", duration=expected
            )
        ]
        self._frames = decoded

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def demux(self, _stream):
        return iter(_FakePacket(index) for index in range(self._frames))


class _FakePacket:
    size = 10

    def __init__(self, pts: int | None, invalid: bool = False) -> None:
        self.pts = pts
        self.invalid = invalid

    def decode(self):
        if self.invalid:
            raise _FakeInvalidDataError("bad packet")
        return [] if self.pts is None else [SimpleNamespace(pts=self.pts)]


class _FakeInvalidDataError(Exception):
    pass


def test_audit_media_decodes_and_reports_exact_frame_count(tmp_path, monkeypatch) -> None:
    payload = b"pinned-media"
    path = tmp_path / "clip.avi"
    path.write_bytes(payload)
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(
            open=lambda _path: _FakeContainer(4),
            error=SimpleNamespace(InvalidDataError=_FakeInvalidDataError),
        ),
    )
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
    assert result[0]["decoded_packet_pts"] == [0, 1, 2, 3]
    assert result[0]["decode_status"] == "PASS"


def test_audit_media_records_missing_declared_frame_packet(tmp_path, monkeypatch) -> None:
    payload = b"pinned-media"
    path = tmp_path / "clip.avi"
    path.write_bytes(payload)
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(
            open=lambda _path: _FakeContainer(4, 5),
            error=SimpleNamespace(InvalidDataError=_FakeInvalidDataError),
        ),
    )
    record = {
        "split": "train",
        "video_id": "clip.avi",
        "relative_path": "clip.avi",
        "media_bytes": len(payload),
        "media_sha256": hashlib.sha256(payload).hexdigest(),
    }

    result = audit_media(tmp_path, [record])[0]

    assert result["decoded_frames"] == 4
    assert result["unavailable_packet_pts"] == [4]
    assert result["decode_status"] == "INCOMPLETE"


def test_audit_media_records_bad_packet_pts_without_claiming_full_decode(
    tmp_path, monkeypatch
) -> None:
    payload = b"pinned-media-with-one-bad-packet"
    path = tmp_path / "clip.avi"
    path.write_bytes(payload)

    class InvalidDataError(Exception):
        pass

    class Packet:
        def __init__(self, pts: int, *, invalid: bool = False) -> None:
            self.pts = pts
            self.size = 10
            self.invalid = invalid

        def decode(self):
            if self.invalid:
                raise InvalidDataError("bad jpeg frame")
            return [SimpleNamespace(pts=self.pts)]

    class Container(_FakeContainer):
        def __init__(self):
            super().__init__(decoded=0, declared=3)

        def demux(self, _stream):
            return iter([Packet(0), Packet(1, invalid=True), Packet(2), Packet(None)])

    fake_av = SimpleNamespace(
        open=lambda _path: Container(),
        error=SimpleNamespace(InvalidDataError=InvalidDataError),
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)
    record = {
        "split": "train",
        "video_id": "clip.avi",
        "relative_path": "clip.avi",
        "media_bytes": len(payload),
        "media_sha256": hashlib.sha256(payload).hexdigest(),
    }

    result = audit_media(tmp_path, [record])[0]

    assert result["decoded_packet_pts"] == [0, 2]
    assert result["unavailable_packet_pts"] == [1]
    assert result["decoded_frames"] == 2
    assert result["decode_status"] == "INCOMPLETE"
