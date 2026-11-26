from __future__ import annotations

from PIL import Image

from scripts.audit_physicalai_normalized_frame_overlap import (
    add_frame_digest,
    canonical_frame_digest,
    summarize_frame_overlap,
)


def test_normalized_digest_matches_same_pixels_at_different_resolution() -> None:
    small = Image.new("RGB", (32, 32), (20, 90, 160))
    large = Image.new("RGB", (64, 64), (20, 90, 160))

    assert canonical_frame_digest(small) == canonical_frame_digest(large)


def test_normalized_digest_changes_when_rgb_content_changes() -> None:
    blue = Image.new("RGB", (32, 32), (20, 90, 160))
    green = Image.new("RGB", (32, 32), (20, 160, 90))

    assert canonical_frame_digest(blue) != canonical_frame_digest(green)


def test_cross_split_frame_reuse_is_reported_with_both_episode_locations() -> None:
    index: dict[str, dict[str, list[dict[str, int]]]] = {}
    frame_hash = "same-normalized-frame"
    add_frame_digest(index, frame_hash, "train", episode=10, frame_index=5)
    add_frame_digest(index, frame_hash, "validation", episode=20, frame_index=7)

    result = summarize_frame_overlap(index)

    assert result["cross_split_unique_frame_hashes"] == 1
    assert result["cross_split_frame_occurrences"] == 1
    assert result["examples"] == [
        {
            "frame_sha256": frame_hash,
            "train": [{"episode": 10, "frame_index": 5}],
            "validation": [{"episode": 20, "frame_index": 7}],
        }
    ]


def test_same_split_repeated_frame_is_not_reported_as_cross_split_overlap() -> None:
    index: dict[str, dict[str, list[dict[str, int]]]] = {}
    add_frame_digest(index, "train-only", "train", episode=10, frame_index=2)
    add_frame_digest(index, "train-only", "train", episode=11, frame_index=4)

    result = summarize_frame_overlap(index)

    assert result["cross_split_unique_frame_hashes"] == 0
    assert result["cross_split_frame_occurrences"] == 0
    assert result["examples"] == []
