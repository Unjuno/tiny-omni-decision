"""Audit GOOSE train/validation ZIP indexes using bounded HTTP byte ranges only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

PAGE_URLS = {
    "project": "https://goose-dataset.de/",
    "setup": "https://goose-dataset.de/docs/setup/",
    "structure": "https://goose-dataset.de/docs/dataset-structure/",
    "platforms": "https://goose-dataset.de/docs/platform-descriptions/",
}

# Values were observed using a single-byte HTTP range request on 2026-10-28.
# ETags pin the exact server objects for all subsequent metadata ranges.
ARCHIVES = (
    {
        "name": "goose_2d_train.zip",
        "url": "https://goose-dataset.de/storage/goose_2d_train.zip",
        "size": 22_466_451_989,
        "etag": '"67d9cad6-53b1ada15"',
        "last_modified": "Tue, 18 Mar 2025 19:34:46 GMT",
        "modality": "image",
        "split": "train",
    },
    {
        "name": "goose_2d_val.zip",
        "url": "https://goose-dataset.de/storage/goose_2d_val.zip",
        "size": 2_857_251_081,
        "etag": '"67d9a658-aa4e3109"',
        "last_modified": "Tue, 18 Mar 2025 16:59:04 GMT",
        "modality": "image",
        "split": "validation",
    },
    {
        "name": "goose_3d_train.zip",
        "url": "https://goose-dataset.de/storage/goose_3d_train.zip",
        "size": 28_846_555_347,
        "etag": '"67daa5b4-6b76380d3"',
        "last_modified": "Wed, 19 Mar 2025 11:08:36 GMT",
        "modality": "pointcloud",
        "split": "train",
    },
    {
        "name": "goose_3d_val.zip",
        "url": "https://goose-dataset.de/storage/goose_3d_val.zip",
        "size": 3_498_402_435,
        "etag": '"67da9ea9-d0856283"',
        "last_modified": "Wed, 19 Mar 2025 10:38:33 GMT",
        "modality": "pointcloud",
        "split": "validation",
    },
)

EOCD_FIXED_BYTES = 22
MAX_ARCHIVE_INDEX_BYTES = 128 * 1024 * 1024
MAX_TOTAL_INDEX_BYTES = 256 * 1024 * 1024
MAX_FILE_COUNT = 1_000_000
SPLIT_NAMES = {"train", "val", "validation", "test"}
ASSET_SUFFIXES = (
    "_windshield_vis",
    "_vls128",
    "_goose",
    "_labelids",
    "_instanceids",
    "_color",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_exact_range(session, spec, start: int, end: int, budget: dict) -> tuple[bytes, float]:
    expected = end - start + 1
    if expected <= 0 or expected > MAX_ARCHIVE_INDEX_BYTES:
        raise ValueError("invalid or over-cap range request")
    if expected > budget["remaining"]:
        raise ValueError("range exceeds total transfer budget")
    if urlparse(spec["url"]).scheme != "https":
        raise ValueError("archive URL must use HTTPS")
    started = time.perf_counter()
    with session.get(
        spec["url"],
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(15, 90),
    ) as response:
        if response.status_code != 206:
            raise ValueError(f"expected HTTP 206, got {response.status_code}")
        expected_range = f"bytes {start}-{end}/{spec['size']}"
        if response.headers.get("Content-Range") != expected_range:
            raise ValueError("unexpected Content-Range")
        if response.headers.get("Content-Length") != str(expected):
            raise ValueError("unexpected Content-Length")
        if response.headers.get("ETag") != spec["etag"]:
            raise ValueError("archive ETag changed")
        if response.headers.get("Last-Modified") != spec["last_modified"]:
            raise ValueError("archive Last-Modified changed")
        if urlparse(response.url).hostname != "goose-dataset.de":
            raise ValueError("archive redirected outside the official publisher host")
        data = response.raw.read(expected + 1)
    elapsed = time.perf_counter() - started
    if len(data) != expected:
        raise ValueError("range response length mismatch")
    budget["remaining"] -= len(data)
    budget["transferred"] += len(data)
    budget["timings"].append({"bytes": len(data), "seconds": elapsed})
    return data, elapsed


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def parse_central_directory(data: bytes, expected_entries: int) -> list[dict]:
    records = []
    offset = 0
    while offset < len(data):
        if offset + 46 > len(data) or data[offset : offset + 4] != b"PK\x01\x02":
            raise ValueError("invalid central-directory record")
        flags = int.from_bytes(data[offset + 8 : offset + 10], "little")
        crc = int.from_bytes(data[offset + 16 : offset + 20], "little")
        compressed = int.from_bytes(data[offset + 20 : offset + 24], "little")
        uncompressed = int.from_bytes(data[offset + 24 : offset + 28], "little")
        name_size = int.from_bytes(data[offset + 28 : offset + 30], "little")
        extra_size = int.from_bytes(data[offset + 30 : offset + 32], "little")
        comment_size = int.from_bytes(data[offset + 32 : offset + 34], "little")
        disk_start = int.from_bytes(data[offset + 34 : offset + 36], "little")
        local_offset = int.from_bytes(data[offset + 42 : offset + 46], "little")
        end = offset + 46 + name_size + extra_size + comment_size
        if end > len(data):
            raise ValueError("truncated central-directory record")
        name_bytes = data[offset + 46 : offset + 46 + name_size]
        extra_start = offset + 46 + name_size
        extra = data[extra_start : extra_start + extra_size]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)

        zip64_values = None
        extra_offset = 0
        while extra_offset < len(extra):
            if extra_offset + 4 > len(extra):
                raise ValueError("truncated ZIP extra-field header")
            field_id = int.from_bytes(extra[extra_offset : extra_offset + 2], "little")
            field_size = int.from_bytes(extra[extra_offset + 2 : extra_offset + 4], "little")
            field_start = extra_offset + 4
            field_end = field_start + field_size
            if field_end > len(extra):
                raise ValueError("truncated ZIP extra field")
            if field_id == 0x0001:
                zip64_values = extra[field_start:field_end]
            extra_offset = field_end

        if (
            any(value == 0xFFFFFFFF for value in (compressed, uncompressed, local_offset))
            or disk_start == 0xFFFF
        ):
            if zip64_values is None:
                raise ValueError("ZIP64 sentinel lacks ZIP64 extended information")
            zoff = 0
            if uncompressed == 0xFFFFFFFF:
                uncompressed = int.from_bytes(zip64_values[zoff : zoff + 8], "little")
                zoff += 8
            if compressed == 0xFFFFFFFF:
                compressed = int.from_bytes(zip64_values[zoff : zoff + 8], "little")
                zoff += 8
            if local_offset == 0xFFFFFFFF:
                local_offset = int.from_bytes(zip64_values[zoff : zoff + 8], "little")
                zoff += 8
            if disk_start == 0xFFFF:
                disk_start = int.from_bytes(zip64_values[zoff : zoff + 4], "little")
        if disk_start != 0:
            raise ValueError("multi-disk ZIP64 member is unsupported")
        records.append(
            {
                "name": name,
                "flags": flags,
                "method": int.from_bytes(data[offset + 10 : offset + 12], "little"),
                "crc32": f"{crc:08x}",
                "compressed_bytes": compressed,
                "uncompressed_bytes": uncompressed,
                "local_header_offset": local_offset,
            }
        )
        offset = end
    if len(records) != expected_entries or len({row["name"] for row in records}) != len(records):
        raise ValueError("central-directory member count or name uniqueness mismatch")
    return records


def locate_central_directory(session, spec, budget):
    size = spec["size"]
    if size < EOCD_FIXED_BYTES:
        raise ValueError("archive is too small to contain a ZIP EOCD")
    eocd_offset = size - EOCD_FIXED_BYTES
    eocd, _ = fetch_exact_range(session, spec, eocd_offset, size - 1, budget)
    if eocd[:4] != b"PK\x05\x06":
        raise ValueError("non-empty ZIP comment or trailing data is unsupported")
    disk = int.from_bytes(eocd[4:6], "little")
    cd_disk = int.from_bytes(eocd[6:8], "little")
    disk_entries = int.from_bytes(eocd[8:10], "little")
    entries = int.from_bytes(eocd[10:12], "little")
    cd_size = int.from_bytes(eocd[12:16], "little")
    cd_offset = int.from_bytes(eocd[16:20], "little")
    comment_size = int.from_bytes(eocd[20:22], "little")
    if comment_size or disk or cd_disk or disk_entries != entries:
        raise ValueError("non-empty ZIP comment or multi-disk archive is unsupported")
    if any(value in (0xFFFF, 0xFFFFFFFF) for value in (entries, cd_size, cd_offset)):
        locator_offset = eocd_offset - 20
        locator, _ = fetch_exact_range(session, spec, locator_offset, eocd_offset - 1, budget)
        if locator[:4] != b"PK\x06\x07":
            raise ValueError("ZIP64 locator is absent")
        locator_disk = int.from_bytes(locator[4:8], "little")
        zip64_offset = int.from_bytes(locator[8:16], "little")
        total_disks = int.from_bytes(locator[16:20], "little")
        if locator_disk != 0 or total_disks != 1:
            raise ValueError("multi-disk ZIP64 archives are unsupported")
        zip64_end, _ = fetch_exact_range(session, spec, zip64_offset, zip64_offset + 55, budget)
        if zip64_end[:4] != b"PK\x06\x06":
            raise ValueError("invalid ZIP64 end-of-central-directory record")
        record_size = int.from_bytes(zip64_end[4:12], "little")
        if record_size < 44 or zip64_offset + 12 + record_size > locator_offset:
            raise ValueError("invalid or overlapping ZIP64 end record")
        disk64 = int.from_bytes(zip64_end[16:20], "little")
        cd_disk64 = int.from_bytes(zip64_end[20:24], "little")
        disk_entries = int.from_bytes(zip64_end[24:32], "little")
        entries = int.from_bytes(zip64_end[32:40], "little")
        cd_size = int.from_bytes(zip64_end[40:48], "little")
        cd_offset = int.from_bytes(zip64_end[48:56], "little")
        if disk64 or cd_disk64 or disk_entries != entries:
            raise ValueError("multi-disk ZIP64 archive is unsupported")
    if entries <= 0 or entries > MAX_FILE_COUNT:
        raise ValueError("archive member count exceeds audit limits")
    if cd_size <= 0 or cd_size > MAX_ARCHIVE_INDEX_BYTES:
        raise ValueError("central-directory size exceeds audit limits")
    if cd_offset + cd_size > size:
        raise ValueError("central-directory bounds exceed archive size")
    return eocd_offset, entries, cd_size, cd_offset


def member_split_and_group(name: str) -> tuple[str | None, str | None]:
    parts = PurePosixPath(name).parts
    split_positions = [i for i, part in enumerate(parts) if part.lower() in SPLIT_NAMES]
    if len(split_positions) != 1:
        return None, None
    index = split_positions[0]
    following = parts[index + 1 :]
    if len(following) < 2:
        return parts[index].lower(), None
    return parts[index].lower(), following[0]


def asset_key(name: str) -> str:
    path = PurePosixPath(name)
    stem = path.stem
    for suffix in ASSET_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    split_positions = [i for i, part in enumerate(path.parts) if part.lower() in SPLIT_NAMES]
    if len(split_positions) == 1:
        relative_parent = path.parts[split_positions[0] + 1 : -1]
    else:
        relative_parent = path.parts[:-1]
    return "/".join((*relative_parent, stem))


def read_archive_index(session, spec, budget):
    size = spec["size"]
    eocd_offset, entries, central_size, central_offset = locate_central_directory(
        session, spec, budget
    )
    central, _ = fetch_exact_range(
        session, spec, central_offset, central_offset + central_size - 1, budget
    )
    if len(central) != central_size:
        raise ValueError("central directory length mismatch")
    records = parse_central_directory(central, entries)
    if any(not safe_member_name(record["name"]) for record in records):
        raise ValueError("archive contains unsafe member path")
    return {
        "name": spec["name"],
        "size": size,
        "etag": spec["etag"],
        "last_modified": spec["last_modified"],
        "eocd_offset": eocd_offset,
        "entry_count": entries,
        "central_directory_sha256": sha256(central),
        "central_directory_bytes": len(central),
        "member_inventory": records,
    }


def summarize(indexes):
    by_name = {row["name"]: row for row in indexes}
    out = {
        "archives": [],
        "cross_modal": {},
        "split_group_overlap": {},
        "split_asset_overlap": {},
    }
    group_sets = {}
    asset_sets = {}
    group_asset_sets = {}
    for spec in ARCHIVES:
        index = by_name[spec["name"]]
        records = index["member_inventory"]
        split_counts = Counter()
        groups = set()
        suffix_counts = Counter()
        assets_by_type = {"image": set(), "pointcloud": set()}
        assets_by_group = {}
        for record in records:
            name = record["name"]
            if name.endswith("/"):
                continue
            split, group = member_split_and_group(name)
            if split:
                split_counts[split] += 1
            if group:
                groups.add(group)
            suffix = PurePosixPath(name).suffix.lower() or "<none>"
            suffix_counts[suffix] += 1
            lower = name.lower()
            if spec["modality"] == "image" and lower.endswith("_windshield_vis.png"):
                key = asset_key(name)
                assets_by_type["image"].add(key)
                assets_by_group.setdefault(group, {}).setdefault("image", set()).add(key)
            if spec["modality"] == "pointcloud" and lower.endswith("_vls128.bin"):
                key = asset_key(name)
                assets_by_type["pointcloud"].add(key)
                assets_by_group.setdefault(group, {}).setdefault("pointcloud", set()).add(key)
        group_sets[(spec["modality"], spec["split"])] = groups
        asset_sets[(spec["modality"], spec["split"])] = assets_by_type[spec["modality"]]
        group_asset_sets[(spec["modality"], spec["split"])] = assets_by_group
        out["archives"].append(
            {
                "name": spec["name"],
                "entry_count": index["entry_count"],
                "asset_count": len(assets_by_type[spec["modality"]]),
                "group_count": len(groups),
                "split_member_counts": dict(sorted(split_counts.items())),
                "file_extension_counts": dict(sorted(suffix_counts.items())),
                "metadata_sha256": index["central_directory_sha256"],
            }
        )
    for split in ("train", "validation"):
        image = asset_sets[("image", split)]
        pointcloud = asset_sets[("pointcloud", split)]
        out["cross_modal"][split] = {
            "image_assets": len(image),
            "pointcloud_assets": len(pointcloud),
            "exact_shared_asset_keys": len(image & pointcloud),
            "image_without_pointcloud": len(image - pointcloud),
            "pointcloud_without_image": len(pointcloud - image),
        }
    for modality in ("image", "pointcloud"):
        train = group_sets[(modality, "train")]
        validation = group_sets[(modality, "validation")]
        out["split_group_overlap"][modality] = {
            "train_groups": len(train),
            "validation_groups": len(validation),
            "overlap_groups": len(train & validation),
        }
        train_assets = asset_sets[(modality, "train")]
        validation_assets = asset_sets[(modality, "validation")]
        out["split_asset_overlap"][modality] = {
            "train_assets": len(train_assets),
            "validation_assets": len(validation_assets),
            "overlap_assets": len(train_assets & validation_assets),
        }
    train_groups = group_sets[("image", "train")]
    validation_groups = group_sets[("image", "validation")]
    train_only_groups = train_groups - validation_groups

    def assets_from_groups(modality, split, selected_groups):
        grouped = group_asset_sets[(modality, split)]
        values = (grouped.get(group, {}).get(modality, set()) for group in selected_groups)
        return set().union(*values)

    filtered_train_images = assets_from_groups("image", "train", train_only_groups)
    filtered_train_points = assets_from_groups("pointcloud", "train", train_only_groups)
    validation_images = assets_from_groups("image", "validation", validation_groups)
    validation_points = assets_from_groups("pointcloud", "validation", validation_groups)
    out["scenario_group_filtered_counts"] = {
        "train_groups_excluded_as_present_in_validation": len(train_groups & validation_groups),
        "train_groups_retained": len(train_only_groups),
        "validation_groups": len(validation_groups),
        "filtered_train_image_assets": len(filtered_train_images),
        "filtered_train_pointcloud_assets": len(filtered_train_points),
        "filtered_train_exact_image_pointcloud_pairs": len(
            filtered_train_images & filtered_train_points
        ),
        "validation_image_assets": len(validation_images),
        "validation_pointcloud_assets": len(validation_points),
        "validation_exact_image_pointcloud_pairs": len(validation_images & validation_points),
        "train_only_group_id_hashes": sorted(
            sha256(group.encode())[:16] for group in train_only_groups
        ),
        "validation_group_id_hashes": sorted(
            sha256(group.encode())[:16] for group in validation_groups
        ),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import requests

    args.output_dir.mkdir(parents=True, exist_ok=False)
    budget = {"remaining": MAX_TOTAL_INDEX_BYTES, "transferred": 0, "timings": []}
    indexes = []
    with requests.Session() as session:
        for spec in ARCHIVES:
            indexes.append(read_archive_index(session, spec, budget))
    aggregate = summarize(indexes)
    protocol = {
        "date": "2026-10-28",
        "scope": "official GOOSE train/validation ZIP indexes only; no media payload read",
        "source_pages": PAGE_URLS,
        "archives": [
            {key: value for key, value in spec.items() if key != "url"}
            | {"download_url": spec["url"]}
            for spec in ARCHIVES
        ],
        "source_archive_full_sha256": None,
        "test_split_accessed": False,
        "max_bytes_per_archive": MAX_ARCHIVE_INDEX_BYTES,
        "max_total_bytes": MAX_TOTAL_INDEX_BYTES,
        "script_sha256": sha256(Path(__file__).read_bytes()),
    }
    inventory_path = args.output_dir / "archive-indexes.json"
    inventory_path.write_text(json.dumps(indexes, indent=2) + "\n", encoding="utf-8")
    result = {
        "protocol": protocol,
        "protocol_sha256": sha256(json.dumps(protocol, indent=2).encode()),
        "index_inventory_sha256": sha256(inventory_path.read_bytes()),
        "summary": aggregate,
        "range_transfer": {
            "bytes": budget["transferred"],
            "range_count": len(budget["timings"]),
            "ranges": budget["timings"],
            "effective_mib_per_second": round(
                budget["transferred"] / max(sum(row["seconds"] for row in budget["timings"]), 1e-9)
                / 2**20,
                3,
            ),
        },
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
