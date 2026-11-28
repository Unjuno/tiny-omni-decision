from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_xcapture_frozen_retrieval import rank_cross_modal, rank_cross_points


def test_rank_cross_points_matches_same_object_in_both_directions():
    features = np.eye(3, dtype=np.float32).repeat(2, axis=0)
    records = [
        {"partition": "validation", "object_id": object_id}
        for object_id in ("a", "a", "b", "b", "c", "c")
    ]

    result = rank_cross_points(features, records, "validation")

    assert result["query_count"] == 6
    assert result["accuracy_top1"] == 1.0
    assert result["accuracy_top5"] == 1.0
    assert result["chance_top1"] == pytest.approx(1 / 3)
    assert all(
        row["query_object_id"] == row["predicted_gallery_object_id"]
        for row in result["predictions"]
    )


def test_rank_cross_points_requires_two_points_per_object():
    features = np.ones((3, 2), dtype=np.float32)
    records = [
        {"partition": "train", "object_id": object_id}
        for object_id in ("a", "a", "b")
    ]

    with pytest.raises(ValueError, match="exactly two points"):
        rank_cross_points(features, records, "train")


def test_rank_cross_modal_uses_distinct_point_gallery():
    query = np.eye(3, dtype=np.float32).repeat(2, axis=0)
    gallery = np.eye(3, dtype=np.float32).repeat(2, axis=0)
    records = [
        {"partition": "validation", "object_id": object_id}
        for object_id in ("a", "a", "b", "b", "c", "c")
    ]

    result = rank_cross_modal(
        query,
        gallery,
        records,
        "validation",
        query_modality="image",
        gallery_modality="tactile",
    )

    assert result["accuracy_top1"] == 1.0
    assert result["query_modality"] == "image"
    assert result["gallery_modality"] == "tactile"


def test_rank_cross_modal_rejects_incompatible_dimensions():
    query = np.ones((4, 2), dtype=np.float32)
    gallery = np.ones((4, 3), dtype=np.float32)
    records = [
        {"partition": "train", "object_id": object_id}
        for object_id in ("a", "a", "b", "b")
    ]

    with pytest.raises(ValueError, match="equal feature dimensions"):
        rank_cross_modal(
            query,
            gallery,
            records,
            "train",
            query_modality="image",
            gallery_modality="tactile",
        )


def test_rank_cross_points_rejects_zero_features():
    features = np.zeros((4, 2), dtype=np.float32)
    records = [
        {"partition": "train", "object_id": object_id}
        for object_id in ("a", "a", "b", "b")
    ]

    with pytest.raises(ValueError, match="zero feature"):
        rank_cross_points(features, records, "train")
