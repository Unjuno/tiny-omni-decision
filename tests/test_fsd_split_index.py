import io
import struct
import zipfile

from scripts.fetch_fsd50k_training_smoke import central_entries


def test_directory_only_preserves_disk_and_member_offset():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("FSD50K.dev_audio/123.wav", b"dummy")
    raw = buffer.getvalue()
    directory = bytearray(raw[raw.index(b"PK\x01\x02"):-22])
    struct.pack_into("<H", directory, 34, 3)
    struct.pack_into("<I", directory, 42, 12345)
    entry = central_entries(bytes(directory), 1)["FSD50K.dev_audio/123.wav"]
    assert entry.volume == 3
    assert entry.header_offset == 12345
    assert entry.file_size == 5
