from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_intphys_normalized_input_identity import (
    hash_normalized_frames,
    merge_normalized_identities,
)


def test_hash_normalized_frames_uses_contiguous_tensor_bytes() -> None:
    frames = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, :, ::2]

    digest = hash_normalized_frames(frames, np)

    assert digest == hash_normalized_frames(np.ascontiguousarray(frames), np)
    assert len(digest) == 64


def test_merge_normalized_identities_preserves_base_rows_and_order() -> None:
    base = [
        {"sample_id": "train.1/a", "rgb_sequence_sha256": "a" * 64},
        {"sample_id": "train.1/b", "rgb_sequence_sha256": "b" * 64},
    ]

    merged = merge_normalized_identities(
        base,
        {"train.1/a": "1" * 64, "train.1/b": "2" * 64},
    )

    assert [row["sample_id"] for row in merged] == ["train.1/a", "train.1/b"]
    assert merged[0]["rgb_sequence_sha256"] == "a" * 64
    assert [row["normalized_input_sha256"] for row in merged] == ["1" * 64, "2" * 64]


@pytest.mark.parametrize(
    ("base", "normalized", "message"),
    [
        (
            [{"sample_id": "train.1/a"}],
            {"train.1/b": "a" * 64},
            "exactly match",
        ),
        (
            [{"sample_id": "train.1/a"}, {"sample_id": "train.1/a"}],
            {"train.1/a": "a" * 64},
            "duplicate",
        ),
        ([{"sample_id": "train.1/a"}], {"train.1/a": "z" * 64}, "SHA-256"),
    ],
)
def test_merge_normalized_identities_rejects_bad_correspondence(
    base: list[dict[str, str]], normalized: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        merge_normalized_identities(base, normalized)
