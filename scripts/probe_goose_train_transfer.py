"""Measure three CRC-checked GOOSE train-image range reads without saving media."""

from __future__ import annotations

import argparse
import json
import statistics
import struct
from pathlib import Path

from scripts.audit_adl_zip_member import decode_local_member
from scripts.audit_goose_archive_indexes import (
    ARCHIVES,
    MAX_ARCHIVE_INDEX_BYTES,
    fetch_exact_range,
    member_split_and_group,
    sha256,
)

INDEX_SHA256 = "38fa35593065b5768e823dc67275d7088c359b29e1690724bfc3f54178d76cb4"
SAMPLE_COUNT = 3
MAX_SAMPLE_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_TOTAL_SAMPLE_BYTES = 128 * 1024 * 1024
TARGET_SAMPLE_COMPRESSED_BYTES = 2_500_000


def select_samples(indexes):
    train_archive = next(row for row in indexes if row["name"] == "goose_2d_train.zip")
    candidates = []
    for record in train_archive["member_inventory"]:
        split, group = member_split_and_group(record["name"])
        size = record["compressed_bytes"]
        if (
            split == "train"
            and group
            and record["name"].lower().endswith("_windshield_vis.png")
            and 512 * 1024 <= size <= MAX_SAMPLE_COMPRESSED_BYTES
        ):
            candidates.append((abs(size - TARGET_SAMPLE_COMPRESSED_BYTES), group, record))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]["name"]))
    selected, groups = [], set()
    for _, group, record in candidates:
        if group in groups:
            continue
        selected.append(record)
        groups.add(group)
        if len(selected) == SAMPLE_COUNT:
            break
    if len(selected) != SAMPLE_COUNT:
        raise ValueError("fewer than three train-image samples from distinct scenario groups")
    return selected


def read_member(session, archive_spec, record, budget):
    offset = record["local_header_offset"]
    fixed, fixed_seconds = fetch_exact_range(session, archive_spec, offset, offset + 29, budget)
    if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
        raise ValueError("invalid local ZIP member header")
    flags, method = struct.unpack_from("<HH", fixed, 6)
    crc, compressed, uncompressed = struct.unpack_from("<III", fixed, 14)
    name_size, extra_size = struct.unpack_from("<HH", fixed, 26)
    name_extra_size = name_size + extra_size
    if name_size > 65535 or name_extra_size + compressed > MAX_ARCHIVE_INDEX_BYTES:
        raise ValueError("local member range exceeds configured bound")
    remainder, payload_seconds = fetch_exact_range(
        session,
        archive_spec,
        offset + 30,
        offset + 30 + name_extra_size + compressed - 1,
        budget,
    )
    name_bytes = remainder[:name_size]
    expected_encoding = "utf-8" if record["flags"] & 0x800 else "cp437"
    if (
        name_bytes.decode(expected_encoding) != record["name"]
        or flags != record["flags"]
        or method != record["method"]
        or (not flags & 8 and crc != int(record["crc32"], 16))
    ):
        raise ValueError("local member metadata does not match the central directory")
    if not flags & 8 and (
        compressed != record["compressed_bytes"]
        or uncompressed != record["uncompressed_bytes"]
    ):
        raise ValueError("local member sizes do not match the central directory")
    decoder_record = {
        "name": record["name"],
        "name_bytes": name_bytes,
        "flags": flags,
        "method": method,
        "crc": crc,
        "compressed": record["compressed_bytes"],
        "uncompressed": record["uncompressed_bytes"],
        "offset": offset,
    }
    member = fixed + remainder
    decoded = decode_local_member(member, decoder_record)
    return {
        "member_id_sha256": sha256(record["name"].encode()),
        "scenario_group_sha256": sha256(member_split_and_group(record["name"])[1].encode())[:16],
        "compressed_bytes": len(member),
        "uncompressed_bytes": len(decoded),
        "compressed_sha256": sha256(member[30 + name_extra_size :]),
        "decoded_member_sha256": sha256(decoded),
        "crc32_verified": True,
        "range_requests": 2,
        "range_seconds": fixed_seconds + payload_seconds,
        "mib_per_second": round(
            len(member) / max(fixed_seconds + payload_seconds, 1e-9) / 2**20, 3
        ),
    }


def audit(index_path: Path, output_dir: Path, session):
    raw = index_path.read_bytes()
    if sha256(raw) != INDEX_SHA256:
        raise ValueError("source archive-index hash mismatch")
    indexes = json.loads(raw)
    records = select_samples(indexes)
    archive_spec = next(row for row in ARCHIVES if row["name"] == "goose_2d_train.zip")
    plan = {
        "source_index_sha256": INDEX_SHA256,
        "archive": archive_spec["name"],
        "archive_size": archive_spec["size"],
        "archive_etag": archive_spec["etag"],
        "selection": (
            "three distinct train scenario groups, selected by smallest absolute "
            "distance from 2,500,000 compressed bytes"
        ),
        "sample_ids_sha256": [sha256(row["name"].encode()) for row in records],
        "compressed_sizes": [row["compressed_bytes"] for row in records],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "sample-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    budget = {
        "remaining": MAX_TOTAL_SAMPLE_BYTES,
        "transferred": 0,
        "timings": [],
    }
    samples = [read_member(session, archive_spec, row, budget) for row in records]
    rates = [row["mib_per_second"] for row in samples]
    total_seconds = sum(row["range_seconds"] for row in samples)
    result = {
        "plan_sha256": sha256(plan_path.read_bytes()),
        "index_sha256": INDEX_SHA256,
        "samples": samples,
        "transfer_summary": {
            "bytes": budget["transferred"],
            "range_requests": len(budget["timings"]),
            "three_sample_mean_mib_per_second": round(statistics.mean(rates), 3),
            "pooled_mib_per_second": round(
                budget["transferred"] / max(total_seconds, 1e-9) / 2**20, 3
            ),
            "estimated_validation_archive_transfer_hours": round(
                (ARCHIVES[1]["size"] + ARCHIVES[3]["size"])
                / (budget["transferred"] / max(total_seconds, 1e-9))
                / 3600,
                2,
            ),
            "estimate_limit": (
                "extrapolation from three sequential train-image member reads; "
                "network throughput is not guaranteed"
            ),
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import requests

    with requests.Session() as session:
        result = audit(args.index_path, args.output_dir, session)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
