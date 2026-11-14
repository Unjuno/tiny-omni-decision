import binascii
import hashlib

import pytest

from scripts.fetch_cater_temporal_media import safe_member_name, validate_existing_media


@pytest.mark.parametrize(
    "member,expected",
    [
        ("videos/CATER_new_000001.avi", "CATER_new_000001.avi"),
    ],
)
def test_safe_member_name_accepts_only_cater_video_members(member, expected) -> None:
    assert safe_member_name(member) == expected


@pytest.mark.parametrize(
    "member",
    [
        "../CATER_new_000001.avi",
        "videos/../CATER_new_000001.avi",
        "scenes/CATER_new_000001.avi",
        "videos/CLEVR_new_000001.avi",
        "videos/CATER_new_000001.json",
    ],
)
def test_safe_member_name_rejects_unexpected_paths(member) -> None:
    with pytest.raises(ValueError):
        safe_member_name(member)


def test_resume_validation_accepts_only_files_matching_pinned_size_and_crc(tmp_path) -> None:
    payload = b"pinned media bytes"
    media = tmp_path / "train" / "CATER_new_000001.avi"
    media.parent.mkdir()
    media.write_bytes(payload)
    key = ("train", media.name)
    crc = binascii.crc32(payload) & 0xFFFFFFFF

    records = validate_existing_media(
        tmp_path,
        [key, ("validation", "CATER_new_000002.avi")],
        {key: (len(payload), 10, crc)},
    )

    assert records[key]["media_sha256"] == hashlib.sha256(payload).hexdigest()
    assert records[key]["source_crc32"] == f"{crc:08x}"


def test_resume_validation_rejects_corrupt_or_unexpected_files(tmp_path) -> None:
    media = tmp_path / "train" / "CATER_new_000001.avi"
    media.parent.mkdir()
    media.write_bytes(b"corrupt")
    key = ("train", media.name)

    with pytest.raises(ValueError, match="size/CRC validation"):
        validate_existing_media(tmp_path, [key], {key: (8, 5, 0)})

    media.unlink()
    unexpected = tmp_path / "train" / "other.bin"
    unexpected.write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected file"):
        validate_existing_media(tmp_path, [key], {key: (8, 5, 0)})
