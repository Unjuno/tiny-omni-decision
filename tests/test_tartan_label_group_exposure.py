from __future__ import annotations

import pytest

from scripts.hash_tartan_label_group_exposure import (
    DERIVATIVE_REVISION,
    DERIVATIVE_SOURCE,
    OFFICIAL_REVISION,
    OFFICIAL_SOURCE,
    build_exposure_roster,
    canonical_clip_group,
    source_group_digest,
)


def test_canonical_clip_group_ignores_utterance_index() -> None:
    assert canonical_clip_group("KAGC_01-02-22_0007_001") == "kagc/01-02-22/0007"
    assert canonical_clip_group("KAGC_01-02-22_0007_042") == "kagc/01-02-22/0007"


def test_canonical_clip_group_rejects_unexpected_identity_shape() -> None:
    with pytest.raises(ValueError, match="six nonempty identity tokens"):
        canonical_clip_group("KAGC_01-02-22_0007")


def test_group_digest_is_source_and_revision_scoped() -> None:
    group = "kagc/01-02-22/0007"
    official = source_group_digest(OFFICIAL_SOURCE, OFFICIAL_REVISION, group)
    derivative = source_group_digest(DERIVATIVE_SOURCE, DERIVATIVE_REVISION, group)
    assert official != derivative
    assert official == source_group_digest(OFFICIAL_SOURCE, OFFICIAL_REVISION, group)


def test_roster_records_only_digests_and_separate_source_revisions() -> None:
    roster = build_exposure_roster(
        ["KAGC_01-02-22_0007_001", "KAGC_01-02-22_0007_002"],
        summary_sha256="a" * 64,
        shard_sha256s={"shard-00000.parquet": "b" * 64},
    )
    assert roster["utterance_id_count"] == 2
    assert roster["unique_parent_clip_count"] == 1
    assert [item["source_id"] for item in roster["sources"]] == [
        OFFICIAL_SOURCE,
        DERIVATIVE_SOURCE,
    ]
    assert "KAGC_01-02-22_0007_001" not in str(roster)
