from __future__ import annotations

import pytest

from scripts.compare_intphys_train_identity_manifests import compare_identity_records


def _record(shard: int, source: str, status: str, rgb: str) -> dict[str, str]:
    return {
        "sample_id": f"train.{shard}/{source}",
        "source_id": source,
        "status_identity_sha256": status * 64,
        "rgb_sequence_sha256": rgb * 64,
    }


def test_comparison_unions_source_status_and_rgb_identity_transitively() -> None:
    result = compare_identity_records(
        {
            1: [_record(1, "a", "a", "1"), _record(1, "b", "b", "2")],
            2: [
                _record(2, "a", "c", "3"),  # source-ID overlap with shard 1
                _record(2, "c", "a", "4"),  # status overlap with shard 1
                _record(2, "d", "d", "2"),  # RGB overlap connects to b
                _record(2, "e", "e", "5"),  # independent identity group
            ],
        }
    )

    assert result["unqualified_source_id_overlap_by_shard_pair"] == {"1-2": 1}
    assert result["exact_status_identity_overlap_by_shard_pair"] == {"1-2": 1}
    assert result["exact_full_rgb_sequence_overlap_by_shard_pair"] == {"1-2": 1}
    assert result["global_unique_groups_after_identity_union"] == 3
    assert result["records_removed_by_global_deduplication"] == 3
    assert result["normalized_sampled_input_identity_compared"] is False


def test_comparison_rejects_duplicate_sample_ids() -> None:
    duplicate = _record(1, "a", "a", "1")
    with pytest.raises(ValueError, match="duplicate"):
        compare_identity_records(
            {
                1: [duplicate, duplicate.copy()],
                2: [_record(2, "b", "b", "2")],
            }
        )


def test_comparison_rejects_malformed_digest() -> None:
    invalid = _record(2, "b", "b", "2")
    invalid["rgb_sequence_sha256"] = "not-a-sha256"
    with pytest.raises(ValueError, match="malformed rgb_sequence_sha256"):
        compare_identity_records({1: [_record(1, "a", "a", "1")], 2: [invalid]})


def test_comparison_rejects_non_hex_identity_digest() -> None:
    invalid = _record(2, "b", "b", "2")
    invalid["rgb_sequence_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="malformed rgb_sequence_sha256"):
        compare_identity_records({1: [_record(1, "a", "a", "1")], 2: [invalid]})


def test_comparison_rejects_partial_sampled_identity() -> None:
    invalid = _record(2, "b", "b", "2")
    invalid["sampled_rgb_payload_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="incomplete sampled RGB identities"):
        compare_identity_records({1: [_record(1, "a", "a", "1")], 2: [invalid]})


def test_comparison_includes_sampled_decoded_pixels_when_present_in_every_shard() -> None:
    shard1 = _record(1, "a", "a", "1")
    shard2 = _record(2, "b", "b", "2")
    shard1["sampled_rgb_payload_sha256"] = "a" * 64
    shard2["sampled_rgb_payload_sha256"] = "a" * 64
    shard1["sampled_decoded_rgb_pixels_sha256"] = "b" * 64
    shard2["sampled_decoded_rgb_pixels_sha256"] = "c" * 64

    result = compare_identity_records({1: [shard1], 2: [shard2]})

    assert result["exact_sampled_rgb_payload_overlap_by_shard_pair"] == {"1-2": 1}
    assert result["exact_sampled_decoded_rgb_pixel_overlap_by_shard_pair"] == {"1-2": 0}
    assert result["sampled_decoded_pixel_identity_compared"] is True
