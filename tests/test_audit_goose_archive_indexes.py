import io
import zipfile

import pytest


def make_zip(names):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in names:
            archive.writestr(name, b"metadata-test-payload")
    return buffer.getvalue()


def make_zip64():
    name = b"images/train/seq-a/frame-1_windshield_vis.png"
    zip64_extra = b"\x01\x00\x10\x00" + (0).to_bytes(8, "little") * 2
    local = (
        b"PK\x03\x04"
        + (45).to_bytes(2, "little")
        + (0).to_bytes(2, "little") * 4
        + (0).to_bytes(4, "little") * 3
        + len(name).to_bytes(2, "little")
        + len(zip64_extra).to_bytes(2, "little")
        + name
        + zip64_extra
    )
    central = (
        b"PK\x01\x02"
        + (45).to_bytes(2, "little") * 2
        + (0).to_bytes(2, "little") * 2
        + (0).to_bytes(2, "little") * 2
        + (0).to_bytes(4, "little")
        + (0xFFFFFFFF).to_bytes(4, "little") * 2
        + len(name).to_bytes(2, "little")
        + len(zip64_extra).to_bytes(2, "little")
        + (0).to_bytes(2, "little")
        + (0).to_bytes(2, "little")
        + (0).to_bytes(2, "little")
        + (0).to_bytes(4, "little") * 2
        + name
        + zip64_extra
    )
    zip64_offset = len(local) + len(central)
    zip64_end = (
        b"PK\x06\x06"
        + (44).to_bytes(8, "little")
        + (45).to_bytes(2, "little") * 2
        + (0).to_bytes(4, "little") * 2
        + (1).to_bytes(8, "little") * 2
        + len(central).to_bytes(8, "little")
        + len(local).to_bytes(8, "little")
    )
    locator = (
        b"PK\x06\x07"
        + (0).to_bytes(4, "little")
        + zip64_offset.to_bytes(8, "little")
        + (1).to_bytes(4, "little")
    )
    eocd = (
        b"PK\x05\x06"
        + (0).to_bytes(2, "little") * 2
        + (0xFFFF).to_bytes(2, "little") * 2
        + (0xFFFFFFFF).to_bytes(4, "little") * 2
        + (0).to_bytes(2, "little")
    )
    return local + central + zip64_end + locator + eocd


class Response:
    def __init__(self, status, headers, body, url):
        self.status_code = status
        self.headers = headers
        self.body = body
        self.url = url
        self.raw = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()


class RangeSession:
    def __init__(self, payload, *, status=206, etag='"v1"'):
        self.payload = payload
        self.status = status
        self.etag = etag
        self.transferred = 0

    def get(self, url, headers, stream, timeout):
        assert stream is True
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        body = self.payload[start : end + 1]
        self.transferred += len(body)
        return Response(
            self.status,
            {
                "Content-Range": f"bytes {start}-{end}/{len(self.payload)}",
                "Content-Length": str(len(body)),
                "ETag": self.etag,
                "Last-Modified": "Tue, 18 Mar 2025 19:34:46 GMT",
            },
            body,
            url,
        )


def spec_for(payload, *, modality="image", split="train"):
    return {
        "name": f"goose_{modality}_{split}.zip",
        "url": "https://goose-dataset.de/test.zip",
        "size": len(payload),
        "etag": '"v1"',
        "last_modified": "Tue, 18 Mar 2025 19:34:46 GMT",
        "modality": modality,
        "split": split,
    }


def test_audits_only_bounded_zip_index_and_summarizes_shared_assets(monkeypatch):
    from scripts import audit_goose_archive_indexes as audit

    examples = {
        ("image", "train"): ["seq-a/frame-1_windshield_vis.png"],
        ("image", "validation"): ["seq-b/frame-2_windshield_vis.png"],
        ("pointcloud", "train"): ["seq-a/frame-1_vls128.bin"],
        ("pointcloud", "validation"): ["seq-b/frame-2_vls128.bin"],
    }
    specs = []
    indexes = []
    sessions = []
    for (modality, split), names in examples.items():
        prefix = "images" if modality == "image" else "velodyne"
        contents = [f"{prefix}/{split}/{name}" for name in names]
        payload = make_zip(contents)
        spec = spec_for(payload, modality=modality, split=split)
        session = RangeSession(payload)
        index = audit.read_archive_index(
            session,
            spec,
            {"remaining": audit.MAX_TOTAL_INDEX_BYTES, "transferred": 0, "timings": []},
        )
        specs.append(spec)
        indexes.append(index)
        sessions.append(session)
        assert session.transferred < len(payload)
        assert index["central_directory_sha256"]
    monkeypatch.setattr(audit, "ARCHIVES", tuple(specs))

    summary = audit.summarize(indexes)
    assert summary["cross_modal"]["train"]["exact_shared_asset_keys"] == 1
    assert summary["cross_modal"]["validation"]["exact_shared_asset_keys"] == 1
    assert summary["split_group_overlap"]["image"]["overlap_groups"] == 0
    assert summary["split_group_overlap"]["pointcloud"]["overlap_groups"] == 0
    filtered = summary["scenario_group_filtered_counts"]
    assert filtered["train_groups_retained"] == 1
    assert filtered["validation_groups"] == 1
    assert filtered["filtered_train_exact_image_pointcloud_pairs"] == 1
    assert filtered["validation_exact_image_pointcloud_pairs"] == 1


def test_reads_zip64_directory_and_extended_member_offsets():
    from scripts.audit_goose_archive_indexes import read_archive_index

    payload = make_zip64()
    session = RangeSession(payload)
    index = read_archive_index(
        session,
        spec_for(payload),
        {"remaining": 1024 * 1024, "transferred": 0, "timings": []},
    )
    assert index["entry_count"] == 1
    assert index["member_inventory"][0]["local_header_offset"] == 0
    assert session.transferred < len(payload)


@pytest.mark.parametrize("status,etag", [(200, '"v1"'), (206, '"changed"')])
def test_refuses_unpinned_or_non_range_response(status, etag):
    from scripts.audit_goose_archive_indexes import fetch_exact_range

    payload = make_zip(["images/train/seq-a/frame-1_windshield_vis.png"])
    spec = spec_for(payload)
    session = RangeSession(payload, status=status, etag=etag)
    budget = {"remaining": 1024, "transferred": 0, "timings": []}
    with pytest.raises(ValueError):
        fetch_exact_range(session, spec, 0, 0, budget)
    assert budget["transferred"] == 0


def test_rejects_unsafe_zip_member_paths():
    from scripts.audit_goose_archive_indexes import read_archive_index

    payload = make_zip(["images/train/../secret.png"])
    session = RangeSession(payload)
    with pytest.raises(ValueError, match="unsafe"):
        read_archive_index(
            session,
            spec_for(payload),
            {"remaining": 1024 * 1024, "transferred": 0, "timings": []},
        )


def test_three_sample_reader_verifies_member_without_saving_media():
    from scripts.probe_goose_train_transfer import read_member

    buffer = io.BytesIO()
    contents = b"sample-image-payload" * 32_000
    name = "images/train/seq-a/frame-1_windshield_vis.png"
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(name, contents)
        info = archive.getinfo(name)
    payload = buffer.getvalue()
    record = {
        "name": info.filename,
        "flags": info.flag_bits,
        "method": info.compress_type,
        "crc32": f"{info.CRC:08x}",
        "compressed_bytes": info.compress_size,
        "uncompressed_bytes": info.file_size,
        "local_header_offset": info.header_offset,
    }
    session = RangeSession(payload)
    result = read_member(
        session,
        spec_for(payload),
        record,
        {"remaining": 1024 * 1024, "transferred": 0, "timings": []},
    )
    assert result["crc32_verified"] is True
    assert result["uncompressed_bytes"] == len(contents)
    assert session.transferred < len(payload)
