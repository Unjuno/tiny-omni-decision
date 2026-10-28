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


def test_instance_correspondence_probe_excludes_validation_scenarios():
    from scripts.probe_goose_instance_correspondence import select_frames

    def members(split, scenario, modality):
        stem = f"{scenario}__0000_123456"
        if modality == "2d":
            names = [
                f"images/{split}/{scenario}/{stem}_windshield_vis.png",
                f"labels/{split}/{scenario}/{stem}_labelids.png",
                f"labels/{split}/{scenario}/{stem}_instanceids.png",
            ]
        else:
            names = [
                f"lidar/{split}/{scenario}/{stem}_vls128.bin",
                f"labels/{split}/{scenario}/{stem}_goose.label",
            ]
        return [{"name": name} for name in names]

    indexes = [
        {
            "name": "goose_2d_train.zip",
            "member_inventory": members("train", "scene-a", "2d")
            + members("train", "scene-b", "2d")
            + members("train", "scene-c", "2d")
            + members("train", "scene-d", "2d"),
        },
        {
            "name": "goose_3d_train.zip",
            "member_inventory": members("train", "scene-a", "3d")
            + members("train", "scene-b", "3d")
            + members("train", "scene-c", "3d")
            + members("train", "scene-d", "3d"),
        },
        {"name": "goose_2d_val.zip", "member_inventory": members("val", "scene-b", "2d")},
        {"name": "goose_3d_val.zip", "member_inventory": members("val", "scene-b", "3d")},
    ]

    selected, validation_group_count, complete_count = select_frames(indexes)
    assert validation_group_count == 1
    assert complete_count == 3
    assert len(selected) == 3
    assert {item[1]["group"] for item in selected} == {"scene-a", "scene-c", "scene-d"}
    followup, _, _ = select_frames(indexes, start_group_index=1, sample_count=2)
    assert {item[1]["group"] for item in followup} == {"scene-c", "scene-d"}


def test_summarizes_cross_modal_instance_ids_without_retaining_payloads():
    import numpy as np
    from PIL import Image

    from scripts.probe_goose_instance_correspondence import summarize_pair

    def png(array):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return buffer.getvalue()

    instance_2d = np.array([[1, 1], [0, 2]], dtype=np.uint16)
    semantic_2d = np.array([[3, 3], [0, 5]], dtype=np.uint8)
    image_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    labels_3d = np.array(
        [(1 << 16) | 3, (1 << 16) | 3, 0, (2 << 16) | 5], dtype="<u4"
    )
    points = np.array(
        [[0, 0, 0, 1], [1, 0, 0, 2], [2, 0, 0, 3], [3, 0, 0, 4]], dtype="<f4"
    )
    payloads = {
        "image": png(image_rgb),
        "instance2d": png(instance_2d),
        "semantic2d": png(semantic_2d),
        "pointcloud": points.tobytes(),
        "labels3d": labels_3d.tobytes(),
    }
    receipts = {key: {"zip_crc32_verified": True} for key in payloads}
    result = summarize_pair(payloads, receipts, {3, 5})
    assert result["shared_instance_object_keys"] == 2
    assert result["shared_instance_class_key_count"] == 2
    assert result["shared_instance_class_key_jaccard"] == 1.0
    assert result["shared_id_jaccard"] == 1.0
    assert result["instance_id_zero_included"] is True
    assert result["point_count"] == 4
    assert result["point_finite"] is True


def test_summarizes_frames_with_no_2d_instance_targets():
    import numpy as np
    from PIL import Image

    from scripts.probe_goose_instance_correspondence import summarize_pair

    def png(array):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return buffer.getvalue()

    payloads = {
        "image": png(np.zeros((1, 1, 3), dtype=np.uint8)),
        "instance2d": png(np.zeros((1, 1), dtype=np.uint8)),
        "semantic2d": png(np.zeros((1, 1), dtype=np.uint8)),
        "pointcloud": np.zeros((1, 4), dtype="<f4").tobytes(),
        "labels3d": np.zeros(1, dtype="<u4").tobytes(),
    }
    result = summarize_pair(payloads, {}, {1})
    assert result["image_instance_class_key_count"] == 0
    assert result["shared_instance_class_key_count"] == 0
    assert result["shared_instance_class_key_jaccard"] == 0.0
