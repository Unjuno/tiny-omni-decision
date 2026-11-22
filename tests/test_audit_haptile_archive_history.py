import io
import zipfile

import pytest

from scripts.pretrained_reuse.audit_haptile_archive_history import parse_zip_tail


def make_test_archive() -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("0417_163342/trajectory.h5", b"placeholder")
        archive.writestr("0417_163342/freq.txt", b"15")
    return payload.getvalue()


def test_parse_zip_tail_reads_only_central_directory_names():
    archive = make_test_archive()
    tail_size = min(65_536, len(archive))
    tail = archive[-tail_size:]
    names = parse_zip_tail(
        tail, absolute_offset=len(archive) - tail_size, archive_size=len(archive)
    )
    assert names == ["0417_163342/trajectory.h5", "0417_163342/freq.txt"]


def test_parse_zip_tail_rejects_missing_eocd():
    with pytest.raises(ValueError, match="end-of-central-directory"):
        parse_zip_tail(b"not a zip", absolute_offset=0, archive_size=8)


def test_parse_zip_tail_rejects_wrong_absolute_offset():
    archive = make_test_archive()
    tail = archive[-64:]
    with pytest.raises(ValueError, match="central directory"):
        parse_zip_tail(tail, absolute_offset=len(archive) - 64, archive_size=len(archive))
