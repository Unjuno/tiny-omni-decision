from __future__ import annotations

import io
import zipfile

from scripts.prepare_grid_selected_video_ranges import (
    parse_central_directory,
    ranged_member,
)


class RangeResponse:
    def __init__(self, payload: bytes, start: int, end: int):
        self.status_code = 206
        self.content = payload[start : end + 1]
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}


class RangeSession:
    def __init__(self, archive: bytes):
        self.archive = archive

    def get(self, url: str, *, headers: dict[str, str], timeout: int):
        assert url == "https://example.invalid/zip"
        assert timeout == 60
        first, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return RangeResponse(self.archive, first, last)


def zip_fixture() -> tuple[bytes, str, bytes]:
    name = "s13/abc123.mpg"
    member = b"fixed native video fixture" * 20
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, member)
    return stream.getvalue(), name, member


def test_central_directory_indexes_selected_video_member(tmp_path):
    archive, name, _ = zip_fixture()
    path = tmp_path / "tail.bin"
    path.write_bytes(archive)

    index = parse_central_directory(path, range_start=0, total_size=len(archive))

    assert list(index) == [name]
    assert index[name]["method"] == zipfile.ZIP_DEFLATED
    assert index[name]["size"] > index[name]["compressed_size"]
    assert index[name]["local_header_offset"] == 0


def test_member_range_checks_headers_size_and_crc():
    archive, name, expected = zip_fixture()
    path = io.BytesIO(archive)
    with zipfile.ZipFile(path) as zipped:
        info = zipped.getinfo(name)
        index_item = {
            "method": info.compress_type,
            "crc32": info.CRC,
            "compressed_size": info.compress_size,
            "size": info.file_size,
            "local_header_offset": info.header_offset,
        }

    result = ranged_member(
        "https://example.invalid/zip",
        name,
        index_item,
        len(archive),
        RangeSession(archive),
    )

    assert result == expected
