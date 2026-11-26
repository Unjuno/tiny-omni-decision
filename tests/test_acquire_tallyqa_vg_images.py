from __future__ import annotations

import pytest

from scripts.acquire_tallyqa_vg_images import (
    ARCHIVE_SHA256,
    SOURCE_COMMIT,
    _image_jobs,
)


def _screen(
    train: dict[str, object], validation: dict[str, object]
) -> dict[str, object]:
    return {
        "source": {"commit": SOURCE_COMMIT},
        "archive_sha256": ARCHIVE_SHA256,
        "split": {
            "image_overlap_count": 0,
            "normalized_question_overlap_count": 0,
            "train": train,
            "validation": validation,
        },
    }


def _split(image_id: str, path: str) -> dict[str, object]:
    return {"image_ids": [image_id], "image_paths": {image_id: path}}


def test_only_pinned_visual_genome_train_and_validation_assets_become_jobs() -> None:
    screen = _screen(
        _split("101", "VG_100K/123.jpg"),
        _split("202", "VG_100K_2/456.jpg"),
    )

    jobs = _image_jobs(screen)

    assert [(job.split, job.image_id, job.source_path) for job in jobs] == [
        ("train", "101", "VG_100K/123.jpg"),
        ("validation", "202", "VG_100K_2/456.jpg"),
    ]


@pytest.mark.parametrize(
    "source_path",
    [
        "train2014/COCO_123.jpg",
        "VG_100K/../outside.jpg",
        "/VG_100K/123.jpg",
        "VG_100K/123.png",
        "VG_100K/nested/123.jpg",
    ],
)
def test_rejects_non_vg_or_unsafe_source_paths(source_path: str) -> None:
    screen = _screen(
        _split("101", source_path),
        _split("202", "VG_100K_2/456.jpg"),
    )

    with pytest.raises(ValueError, match="disallowed Visual Genome asset path"):
        _image_jobs(screen)


def test_rejects_cross_split_identity_or_path_overlap() -> None:
    duplicate_id = _screen(
        _split("101", "VG_100K/123.jpg"),
        _split("101", "VG_100K_2/456.jpg"),
    )
    duplicate_path = _screen(
        _split("101", "VG_100K/123.jpg"),
        _split("202", "VG_100K/123.jpg"),
    )

    with pytest.raises(ValueError, match="duplicate image identity/path"):
        _image_jobs(duplicate_id)
    with pytest.raises(ValueError, match="duplicate image identity/path"):
        _image_jobs(duplicate_path)


def test_rejects_unpinned_archive_or_nonzero_split_overlap() -> None:
    wrong_commit = _screen(
        _split("101", "VG_100K/123.jpg"),
        _split("202", "VG_100K_2/456.jpg"),
    )
    wrong_commit["source"]["commit"] = "not-pinned"  # type: ignore[index]
    overlap = _screen(
        _split("101", "VG_100K/123.jpg"),
        _split("202", "VG_100K_2/456.jpg"),
    )
    overlap["split"]["image_overlap_count"] = 1  # type: ignore[index]

    with pytest.raises(ValueError, match="pinned TallyQA commit"):
        _image_jobs(wrong_commit)
    with pytest.raises(ValueError, match="image-disjoint split"):
        _image_jobs(overlap)
