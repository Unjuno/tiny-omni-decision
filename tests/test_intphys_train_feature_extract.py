import pytest

from scripts.extract_intphys_train_features import (
    canonical_status_hash,
    deterministic_group_split,
    parse_train_member,
)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("train", ("", None, None)),
        ("train/00017/", ("00017", None, None)),
        ("train/00017/status.json", ("00017", "status", None)),
        ("train/00017/scene/scene_015.png", ("00017", "scene", 15)),
        ("train/00017/depth/depth_100.png", ("00017", "depth", 100)),
    ],
)
def test_parse_train_member(path, expected):
    assert parse_train_member(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "../train/00017/status.json",
        "/train/00017/status.json",
        "train/00017/scene/depth_001.png",
        "train/00017/other/other_001.png",
    ],
)
def test_parse_train_member_rejects_unsafe_or_unexpected_paths(path):
    with pytest.raises(ValueError):
        parse_train_member(path)


def test_status_identity_requires_possible_label_and_one_hundred_frames():
    status = {"header": {"is_possible": True}, "frames": [{} for _ in range(100)]}
    assert len(canonical_status_hash(status)) == 64

    impossible = {"header": {"is_possible": False}, "frames": [{} for _ in range(100)]}
    incomplete = {"header": {"is_possible": True}, "frames": [{} for _ in range(99)]}
    with pytest.raises(ValueError, match="must mark every sample possible"):
        canonical_status_hash(impossible)
    with pytest.raises(ValueError, match="exactly 100 frames"):
        canonical_status_hash(incomplete)


def test_deterministic_split_groups_duplicate_status_or_normalized_media():
    sample_ids = [f"{index:05d}" for index in range(30)]
    status_hashes = [f"status-{index}" for index in range(30)]
    media_hashes = [f"media-{index}" for index in range(30)]
    status_hashes[1] = status_hashes[0]
    media_hashes[3] = media_hashes[2]

    train, validation, group_ids = deterministic_group_split(
        sample_ids, status_hashes, media_hashes, seed=17
    )
    train_groups = {group_ids[index] for index in train}
    validation_groups = {group_ids[index] for index in validation}

    assert train and validation
    assert group_ids[0] == group_ids[1]
    assert group_ids[2] == group_ids[3]
    assert not train_groups & validation_groups
    assert deterministic_group_split(
        sample_ids, status_hashes, media_hashes, seed=17
    ) == (train, validation, group_ids)
