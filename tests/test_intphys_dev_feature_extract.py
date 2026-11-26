import pytest

from scripts.extract_intphys_dev_features import parse_dev_member, validate_audit_summary


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
