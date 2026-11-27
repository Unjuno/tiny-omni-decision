from __future__ import annotations

import pytest

from scripts.smoke_physical_concepts_videoprism import (
    uniform_frame_indices,
    validate_source_audit,
)


def test_physical_concepts_15_frames_are_sampled_uniformly_for_videoprism() -> None:
    assert uniform_frame_indices(15, 8) == [0, 2, 4, 6, 8, 10, 12, 14]


def test_uniform_frame_sampling_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="requested frame count"):
        uniform_frame_indices(7, 8)


def test_feature_extraction_accepts_only_matching_train_or_validation_audit() -> None:
    manifest = {
        "train": {"object_name": "train", "generation": "1", "sha256": "a", "records": 2},
        "validation": {
            "object_name": "validation",
            "generation": "2",
            "sha256": "b",
            "records": 3,
        },
    }
    audit = {
        "status": "complete",
        "object_name": "validation",
        "generation": "2",
        "source_sha256": "b",
        "records": 3,
    }
    assert validate_source_audit(manifest, "validation", audit) == manifest["validation"]
    with pytest.raises(ValueError, match="does not match"):
        validate_source_audit(manifest, "train", audit)


def test_feature_extraction_rejects_non_training_split() -> None:
    with pytest.raises(ValueError, match="only train and validation"):
        validate_source_audit({}, "probe", {})
