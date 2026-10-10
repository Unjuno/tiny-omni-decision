from __future__ import annotations

from scripts.extract_frozen_videoprism_features import _apply_temporal_order


def test_reverse_temporal_order_preserves_frames_and_reverses_only_order() -> None:
    frames = ["frame-0", "frame-1", "frame-2", "frame-3"]

    reversed_frames = _apply_temporal_order(frames, "reverse")

    assert reversed_frames == ["frame-3", "frame-2", "frame-1", "frame-0"]
    assert sorted(reversed_frames) == sorted(frames)
    assert frames == ["frame-0", "frame-1", "frame-2", "frame-3"]
    assert _apply_temporal_order(reversed_frames, "reverse") == frames


def test_forward_temporal_order_keeps_the_same_sequence() -> None:
    frames = ["frame-0", "frame-1", "frame-2"]

    assert _apply_temporal_order(frames, "forward") is frames
