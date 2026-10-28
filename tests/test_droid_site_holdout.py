from __future__ import annotations

import pytest

from tiny_omni_decision.droid_site_holdout import (
    canonical_droid_path,
    droid_site_from_path,
    select_site_holdout,
    stable_droid_id,
    uniform_step_indices,
    validate_site_split,
)


def test_canonical_droid_ids_normalize_path_case_and_separators() -> None:
    assert canonical_droid_path(r"GS://Source\R2D2-Data-Full\TRI\Episode.H5") == (
        "gs:/source/r2d2-data-full/tri/episode.h5"
    )
    assert stable_droid_id(r"GS://Source\R2D2-Data-Full\TRI\Episode.H5") == (
        stable_droid_id("gs:/source/r2d2-data-full/tri/episode.h5")
    )


def test_site_and_uniform_indices_match_the_frozen_task_rule() -> None:
    assert droid_site_from_path(
        "gs://xembodiment_data/r2d2/r2d2-data-full/TRI/success/trajectory.h5"
    ) == "tri"
    assert uniform_step_indices(37, 8) == [0, 5, 10, 15, 21, 26, 31, 36]
    assert uniform_step_indices(8, 8) == list(range(8))


def test_site_holdout_is_deterministic_and_uses_counts_not_labels() -> None:
    counts = {
        "autolab": 126,
        "bvl": 31,
        "clvr": 47,
        "guptalab": 8,
        "iliad": 67,
        "iprl": 73,
        "iris": 93,
        "pennpal": 32,
        "rad": 20,
        "rail": 90,
        "real": 51,
        "rpl": 32,
        "tri": 204,
        "weird": 32,
    }
    first = select_site_holdout(counts)
    second = select_site_holdout(dict(reversed(list(counts.items()))))
    assert first == second
    assert first["validation_sites"] == ["bvl", "iliad", "pennpal", "real"]
    assert first["validation_rows"] == 181
    assert first["validation_fraction"] == pytest.approx(0.1997792494)
    assert len(first["training_sites"]) == 10


def test_site_holdout_fails_when_no_three_site_partition_meets_bounds() -> None:
    with pytest.raises(ValueError, match="no label-independent"):
        select_site_holdout({"a": 90, "b": 5, "c": 5}, minimum_fraction=0.15)


def test_split_validation_rejects_media_identity_overlap() -> None:
    rows = [
        {
            "id": "train-id",
            "group_id": "train-group",
            "site": "site-a",
            "split": "train",
            "input_frame_sha256": ["frame-a"],
        },
        {
            "id": "validation-id",
            "group_id": "validation-group",
            "site": "site-b",
            "split": "validation",
            "input_frame_sha256": ["frame-a"],
        },
    ]
    with pytest.raises(ValueError, match="frame overlap"):
        validate_site_split(rows)


def test_split_validation_rejects_shared_site_even_with_distinct_recordings() -> None:
    rows = [
        {
            "id": "train-id",
            "group_id": "train-group",
            "site": "site-a",
            "split": "train",
            "input_frame_sha256": ["frame-a"],
        },
        {
            "id": "validation-id",
            "group_id": "validation-group",
            "site": "site-a",
            "split": "validation",
            "input_frame_sha256": ["frame-b"],
        },
    ]
    with pytest.raises(ValueError, match="site overlap"):
        validate_site_split(rows)
