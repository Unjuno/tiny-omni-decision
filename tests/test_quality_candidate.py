from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest

from tiny_omni_decision.video_cache import install_video_decode_cache


class FakeVideoProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def sample_frames(self, *, num_frames: int) -> None:
        return None

    def fetch_videos(self, video: str, *, sample_indices_fn=None) -> tuple[object, object]:
        num_frames = sample_indices_fn.keywords["num_frames"]
        self.calls.append((video, num_frames))
        return [video, num_frames], {"video": video, "num_frames": num_frames}


def test_video_decode_cache_reuses_exact_path_and_frame_policy() -> None:
    video_processor = FakeVideoProcessor()
    processor = SimpleNamespace(video_processor=video_processor)
    install_video_decode_cache(processor, video_num_frames=8)
    sample_eight = partial(video_processor.sample_frames, num_frames=8)

    first = processor.video_processor.fetch_videos("scene-1.mp4", sample_indices_fn=sample_eight)
    second = processor.video_processor.fetch_videos(
        "scene-1.mp4",
        sample_indices_fn=partial(video_processor.sample_frames, num_frames=8),
    )

    assert first is second
    assert video_processor.calls == [("scene-1.mp4", 8)]


def test_video_decode_cache_separates_sampling_policy_and_evicts_lru() -> None:
    video_processor = FakeVideoProcessor()
    processor = SimpleNamespace(video_processor=video_processor)
    install_video_decode_cache(processor, video_num_frames=8, capacity=1)

    def fetch(video: str, num_frames: int) -> object:
        return processor.video_processor.fetch_videos(
            video,
            sample_indices_fn=partial(video_processor.sample_frames, num_frames=num_frames),
        )

    fetch("scene-1.mp4", 8)
    fetch("scene-1.mp4", 8)
    fetch("scene-2.mp4", 8)
    fetch("scene-1.mp4", 8)

    assert video_processor.calls == [
        ("scene-1.mp4", 8),
        ("scene-2.mp4", 8),
        ("scene-1.mp4", 8),
    ]


def test_video_decode_cache_requires_positive_capacity() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        install_video_decode_cache(
            SimpleNamespace(video_processor=FakeVideoProcessor()),
            video_num_frames=8,
            capacity=0,
        )
