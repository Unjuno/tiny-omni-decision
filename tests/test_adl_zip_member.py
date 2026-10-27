import io
import zipfile

import pytest

from scripts.audit_adl_zip_member import central_records, decode_local_member
from scripts.audit_robomnist_archive_directories import parse_eocd


def fixture_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('activity/multisensor-1.hdf5', b'hello' * 500)
        z.writestr('activity/multisensor-2.hdf5', b'other' * 50)
    return buf.getvalue()


def test_central_records_and_local_member_round_trip():
    data = fixture_zip()
    _, entries, size, offset = parse_eocd(data, 0, len(data))
    records = central_records(data[offset:offset + size], entries)
    assert len(records) == 2
    first, second = records
    actual = decode_local_member(data[first['offset']:second['offset']], first)
    assert actual == b'hello' * 500


def test_detects_wrong_crc_and_local_name():
    data = fixture_zip()
    _, entries, size, offset = parse_eocd(data, 0, len(data))
    records = central_records(data[offset:offset + size], entries)
    first = records[0]
    block = data[first['offset']:records[1]['offset']]
    with pytest.raises(ValueError, match='CRC'):
        decode_local_member(block, {**first, 'crc': first['crc'] ^ 1})
    with pytest.raises(ValueError, match='name'):
        decode_local_member(block, {**first, 'name_bytes': b'wrong'})


def test_rejects_truncated_and_wrong_entry_counts():
    data = fixture_zip()
    _, entries, size, offset = parse_eocd(data, 0, len(data))
    with pytest.raises(ValueError):
        central_records(data[offset:offset + size - 1], entries)
    with pytest.raises(ValueError):
        central_records(data[offset:offset + size], entries + 1)
