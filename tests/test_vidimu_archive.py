import hashlib

import pytest

from scripts.audit_vidimu_analysis import verify_archive


def test_archive_integrity_rejects_partial_and_same_length_corruption(tmp_path):
    path = tmp_path / "archive.zip"
    data = b"fixed archive contents"
    checksum = "md5:" + hashlib.md5(data).hexdigest()
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError, match="size"):
        verify_archive(path, size=len(data), checksum=checksum)
    path.write_bytes(b"X" + data[1:])
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(path, size=len(data), checksum=checksum)
    path.write_bytes(data)
    verify_archive(path, size=len(data), checksum=checksum)
