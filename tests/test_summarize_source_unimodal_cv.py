import pytest

from scripts.summarize_source_unimodal_cv import paired_interval


def test_resamples_video_groups_and_is_deterministic():
    rows = [
        {"video_id": "one", "delta": 1},
        {"video_id": "one", "delta": 1},
        {"video_id": "two", "delta": -1},
    ]
    first = paired_interval(rows, repetitions=100)
    assert first == paired_interval(rows, repetitions=100)
    assert first["video_groups"] == 2
    assert first["delta"] == pytest.approx(1 / 3)


def test_empty_comparison_fails():
    with pytest.raises(ValueError):
        paired_interval([])
