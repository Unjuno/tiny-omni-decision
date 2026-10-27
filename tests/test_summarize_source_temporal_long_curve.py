import pytest

pytest.importorskip("torch")

from scripts.summarize_source_temporal_long_curve import paired_bootstrap  # noqa: E402


def test_bootstrap_groups_records_and_is_deterministic():
    pairs = [
        ({"video_id": "v1", "correct": False}, {"correct": True}),
        ({"video_id": "v1", "correct": True}, {"correct": True}),
        ({"video_id": "v2", "correct": True}, {"correct": False}),
    ]
    result = paired_bootstrap(pairs, "correct", repetitions=100)
    assert result["video_groups"] == 2
    assert result["delta"] == pytest.approx(0.0)
    assert result == paired_bootstrap(pairs, "correct", repetitions=100)


def test_empty_pairing_fails():
    with pytest.raises(ValueError):
        paired_bootstrap([], "correct")
