import sys
from types import SimpleNamespace

import numpy as np

from scripts.extract_perception_source_candidate_audio_features import (
    collect_unique_windows,
    decode_resampled_mono_audio,
)


def test_audio_feature_windows_deduplicate_but_keep_record_references():
    window = {
        "source_start_us": 1_000_000,
        "source_end_us": 3_000_000,
        "pad_left_us": 0,
        "pad_right_us": 0,
    }
    records = [
        {
            "record_id": "r1",
            "video_id": "v1",
            "audio_windows": [window],
            "shifted_audio_windows": [window],
        },
        {
            "record_id": "r2",
            "video_id": "v1",
            "audio_windows": [window],
            "shifted_audio_windows": None,
        },
    ]

    unique, references = collect_unique_windows(records)

    assert len(unique) == 1
    assert [row["feature_index"] for row in references] == [0, 0, 0]
    assert [(row["record_id"], row["variant"]) for row in references] == [
        ("r1", "source"),
        ("r1", "offset"),
        ("r2", "source"),
    ]


def test_audio_decode_preserves_nonzero_pts_origin(tmp_path, monkeypatch):
    media = tmp_path / "audio.mp4"
    media.write_bytes(b"media")
    source_stream = SimpleNamespace(type="audio")
    sample_array = np.asarray([[0.25, -0.25]], dtype=np.float32)

    class FakeResampler:
        def __init__(self, *, format, layout, rate):
            assert (format, layout, rate) == ("fltp", "mono", 10)

        def resample(self, frame):
            if frame is None:
                return []
            return [SimpleNamespace(time=0.5, to_ndarray=lambda: sample_array)]

    class FakeContainer:
        streams = [source_stream]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, _stream):
            return iter([object()])

    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(AudioResampler=FakeResampler, open=lambda _path: FakeContainer()),
    )
    waveform, start_us = decode_resampled_mono_audio(media, sample_rate_hz=10)

    assert start_us == 500_000
    assert waveform.tolist() == [0.25, -0.25]
