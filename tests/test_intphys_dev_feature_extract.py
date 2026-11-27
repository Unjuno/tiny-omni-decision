import pytest

from scripts.extract_intphys_dev_features import (
    get_or_create_dev_video_record,
    parse_dev_member,
    sampled_media_label_collision_audit,
    validate_audit_summary,
    validate_dev_video_record,
    validate_reference_dev_rows,
)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("dev/", ("", "", "", None)),
        ("dev/O1/", ("O1", "", "", None)),
        ("dev/O1/01/", ("O1", "01", "", None)),
        ("dev/O1/01/1/", ("O1", "01", "", None)),
        ("dev/O1/01/1/status.json", ("O1", "01", "status", None)),
        ("dev/O2/17/3/scene/scene_15.png", ("O2", "17", "scene", 15)),
        ("dev/O3/30/4/masks/masks_100.png", ("O3", "30", "masks", 100)),
    ],
)
def test_parse_dev_member(path, expected):
    assert parse_dev_member(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "../dev/O1/01/1/status.json",
        "/dev/O1/01/1/status.json",
        "dev/O4/01/1/status.json",
        "dev/O1/01/1/scene/depth_001.png",
        "dev/O1/01/1/other/other_001.png",
    ],
)
def test_parse_dev_member_rejects_unsafe_or_unexpected_paths(path):
    with pytest.raises(ValueError):
        parse_dev_member(path)


def test_dev_feature_extraction_requires_the_previously_audited_archive_summary():
    valid = {
        "archive_bytes": 3_203_229_667,
        "archive_md5": "6a72715007f2f3b0e9546bfa8d8fc39b",
        "archive_sha256": "441eebe09f6583a18dbcd9d2b6c6b5667c66bac99e3875622e0768dc22e16a0a",
        "video_count": 360,
        "scene_group_count": 90,
        "matched_groups_have_two_labels_each": True,
    }
    validate_audit_summary(valid)

    invalid = {**valid, "scene_group_count": 89}
    with pytest.raises(ValueError, match="admitted audit"):
        validate_audit_summary(invalid)


def test_sampled_media_collision_audit_reports_opposite_label_duplicates():
    report = sampled_media_label_collision_audit(
        ["g1", "g1", "g2", "g2"],
        [0, 1, 0, 1],
        ["same", "same", "possible-only", "impossible-only"],
    )
    assert report["groups_with_opposite_label_identical_sampled_media"] == 1
    assert report["opposite_label_identical_sampled_media_signatures_global"] == 1
    assert report["within_group_collision_group_ids"] == ["g1"]


def test_sampled_media_collision_audit_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match="equal nonzero length"):
        sampled_media_label_collision_audit(["g1"], [0, 1], ["m1"])


def test_dev_records_remain_grouped_when_archive_members_interleave_videos():
    records = {}
    order = []
    first = ("O1", "01", "1")
    second = ("O2", "02", "2")

    first_record = get_or_create_dev_video_record(records, order, first)
    second_record = get_or_create_dev_video_record(records, order, second)
    first_again = get_or_create_dev_video_record(records, order, first)

    first_again["label"] = 1
    assert first_record is first_again
    assert order == [first, second]
    assert second_record["label"] is None


def test_dev_record_requires_complete_frames_for_every_modality():
    key = ("O1", "01", "1")
    row = {
        "frames": {
            channel: set(range(1, 101)) for channel in ("scene", "depth", "masks")
        },
        "selected_png": {frame: b"png" for frame in (1, 15, 29, 43, 57, 71, 85, 100)},
        "label": 1,
    }
    validate_dev_video_record(key, row)

    row["frames"]["masks"].remove(100)
    with pytest.raises(ValueError, match="masks frames"):
        validate_dev_video_record(key, row)


def test_dev_record_accepts_requested_sixteen_frame_policy():
    from scripts.audit_intphys_temporal_sampling_coverage import uniform_frame_numbers

    key = ("O1", "01", "1")
    selected = uniform_frame_numbers(100, 16)
    row = {
        "frames": {
            channel: set(range(1, 101)) for channel in ("scene", "depth", "masks")
        },
        "selected_png": {frame: b"png" for frame in selected},
        "label": 1,
    }
    validate_dev_video_record(key, row, selected)


def test_reference_dev_validation_requires_exact_order_and_labels(tmp_path):
    import numpy as np

    path = tmp_path / "reference.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        group_ids=np.asarray(["g"] * 4),
        labels=np.asarray([1, 1, 0, 0], dtype=np.int8),
        blocks=np.asarray(["O1"] * 4),
    )
    digest = validate_reference_dev_rows(
        path, ["a", "b", "c", "d"], ["g"] * 4, ["O1"] * 4, [1, 1, 0, 0]
    )
    assert len(digest) == 64
    with pytest.raises(ValueError, match="IDs/order/groups/labels"):
        validate_reference_dev_rows(
            path, ["b", "a", "c", "d"], ["g"] * 4, ["O1"] * 4, [1, 1, 0, 0]
        )
