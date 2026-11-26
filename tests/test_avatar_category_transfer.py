from __future__ import annotations

import numpy as np
import pytest

from scripts.run_avatar_category_transfer import (
    _macro_category_metrics,
    _rank,
    build_choices,
    build_transfer_partition,
    validate_choice_splits,
)


def _fixture():
    categories = [f"category-{index}" for index in range(8)]
    held = set(
        sorted(categories, key=lambda category: _rank("avatar-category-transfer-v1", category))[:4]
    )
    seen = set(categories) - held
    queries = []
    videos = []
    for partition, allowed in (("held", held), ("seen", seen)):
        group_count = 25 if partition == "seen" else 10
        for group_index in range(group_count):
            group = f"{partition}-group-{group_index}"
            for category in sorted(allowed):
                key = f"{group}-{category}"
                queries.append(
                    {
                        "query_id": key,
                        "asset_key": key,
                        "source_group": group,
                        "category": category,
                        "timestamp_seconds": float(group_index + 1),
                    }
                )
                videos.append(
                    {
                        "asset_key": f"video-{key}",
                        "source_group": group,
                        "category": category,
                        "timestamp_seconds": float(group_index + 1) + 0.5,
                    }
                )
    return queries, videos, held, seen


def test_category_transfer_partition_is_deterministic_and_group_disjoint():
    queries, videos, held, seen = _fixture()
    first = build_transfer_partition(queries, videos, held_category_count=4)
    second = build_transfer_partition(queries, videos, held_category_count=4)

    assert first["held_categories"] == second["held_categories"] == sorted(held)
    assert first["seen_categories"] == second["seen_categories"] == sorted(seen)
    assert first["fit_groups"] == second["fit_groups"]
    assert first["validation_groups"] == second["validation_groups"]
    assert first["held_test_groups"] == second["held_test_groups"]
    assert first["fit_groups"].isdisjoint(first["validation_groups"])
    assert first["fit_groups"].isdisjoint(first["held_test_groups"])
    assert first["validation_groups"].isdisjoint(first["held_test_groups"])


def test_rebuilt_choice_pools_keep_held_categories_and_sources_out_of_fit():
    queries, videos, held, seen = _fixture()
    partition = build_transfer_partition(queries, videos, held_category_count=4)
    choices = {
        "fit": build_choices(
            queries,
            videos,
            split="fit",
            allowed_groups=partition["fit_groups"],
            allowed_categories=seen,
        ),
        "validation": build_choices(
            queries,
            videos,
            split="validation",
            allowed_groups=partition["validation_groups"],
            allowed_categories=seen,
        ),
        "test": build_choices(
            queries,
            videos,
            split="test",
            allowed_groups=partition["held_test_groups"],
            allowed_categories=held,
        ),
    }
    integrity = validate_choice_splits(choices)

    fit_categories = {category for row in choices["fit"] for category in row["option_categories"]}
    validation_categories = {
        category for row in choices["validation"] for category in row["option_categories"]
    }
    test_categories = {category for row in choices["test"] for category in row["option_categories"]}
    assert fit_categories <= seen
    assert validation_categories <= seen
    assert test_categories <= held
    assert integrity["cross_split_source_group_overlap"] == 0
    assert integrity["cross_split_asset_overlap"] == 0
    assert all(len(row["video_source_groups"]) == 4 for rows in choices.values() for row in rows)
    assert all(
        row["video_source_groups"][row["target_index"]] != row["audio_source_group"]
        for rows in choices.values()
        for row in rows
    )


def test_choice_split_rejects_group_overlap():
    queries, videos, held, seen = _fixture()
    partition = build_transfer_partition(queries, videos, held_category_count=4)
    fit = build_choices(
        queries,
        videos,
        split="fit",
        allowed_groups=partition["fit_groups"],
        allowed_categories=seen,
    )
    held_test = build_choices(
        queries,
        videos,
        split="test",
        allowed_groups=partition["held_test_groups"],
        allowed_categories=held,
    )
    contaminated = dict(held_test[0])
    contaminated["audio_source_group"] = fit[0]["audio_source_group"]

    with pytest.raises(ValueError, match="source-group overlap"):
        validate_choice_splits({"fit": fit, "test": [contaminated]})


def test_partition_rejects_invalid_validation_fraction():
    queries, videos, _, _ = _fixture()

    with pytest.raises(ValueError, match="validation_fraction"):
        build_transfer_partition(
            queries,
            videos,
            held_category_count=4,
            validation_fraction=1.0,
        )


def test_macro_metrics_weight_categories_equally():
    probabilities = np.asarray(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    result = _macro_category_metrics(
        probabilities,
        np.asarray([0, 0, 0]),
        ["frequent", "frequent", "rare"],
    )

    assert result["category_count"] == 2
    assert result["macro_over_categories"]["accuracy"] == pytest.approx(0.25)
