"""Verify paired 2D/3D GOOSE instance IDs on three train-only scenarios.

The script fetches only five explicitly selected ZIP members per frame through
bounded HTTP byte ranges. Decoded media and labels remain in memory and are
discarded after counts and hashes are written.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_adl_zip_member import decode_local_member
from scripts.audit_goose_archive_indexes import (
    ARCHIVES,
    MAX_ARCHIVE_INDEX_BYTES,
    asset_key,
    fetch_exact_range,
    member_split_and_group,
    sha256,
)

INDEX_SHA256 = "38fa35593065b5768e823dc67275d7088c359b29e1690724bfc3f54178d76cb4"
SAMPLE_COUNT = 3
MAX_TOTAL_SAMPLE_BYTES = 128 * 1024 * 1024
SUFFIX_TO_KIND = {
    "_windshield_vis.png": "image",
    "_labelids.png": "semantic2d",
    "_instanceids.png": "instance2d",
    "_vls128.bin": "pointcloud",
    "_goose.label": "labels3d",
}


def _archive(indexes, name):
    return next(row for row in indexes if row["name"] == name)


def select_frames(indexes, *, start_group_index=0, sample_count=SAMPLE_COUNT):
    validation_groups = set()
    for archive_name in ("goose_2d_val.zip", "goose_3d_val.zip"):
        for row in _archive(indexes, archive_name)["member_inventory"]:
            split, group = member_split_and_group(row["name"])
            if split in {"val", "validation"} and group:
                validation_groups.add(group)

    train_records = {}
    for archive_name in ("goose_2d_train.zip", "goose_3d_train.zip"):
        for row in _archive(indexes, archive_name)["member_inventory"]:
            split, group = member_split_and_group(row["name"])
            if split != "train" or not group or group in validation_groups:
                continue
            kind = next(
                (value for suffix, value in SUFFIX_TO_KIND.items() if row["name"].endswith(suffix)),
                None,
            )
            if kind:
                train_records.setdefault(asset_key(row["name"]), {"group": group})[kind] = row

    required = {"image", "semantic2d", "instance2d", "pointcloud", "labels3d"}
    complete = [
        (key, value)
        for key, value in sorted(train_records.items())
        if required <= value.keys()
    ]
    by_group = {}
    for key, value in complete:
        by_group.setdefault(value["group"], (key, value))
    ordered_groups = sorted(by_group)
    selected_groups = ordered_groups[start_group_index : start_group_index + sample_count]
    selected = [by_group[group] for group in selected_groups]
    if len(selected) != sample_count:
        raise ValueError("requested sample range exceeds complete train-only scenarios")
    if len({value["group"] for _, value in selected}) != sample_count:
        raise ValueError("selected frames are not from distinct scenarios")
    return selected, len(validation_groups), len(complete)


def read_member_payload(session, spec, record, budget):
    offset = record["local_header_offset"]
    fixed, t_header = fetch_exact_range(session, spec, offset, offset + 29, budget)
    if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
        raise ValueError("invalid ZIP local header")
    flags, method = struct.unpack_from("<HH", fixed, 6)
    crc, compressed, uncompressed = struct.unpack_from("<III", fixed, 14)
    name_size, extra_size = struct.unpack_from("<HH", fixed, 26)
    total = 30 + name_size + extra_size + record["compressed_bytes"]
    if total > MAX_ARCHIVE_INDEX_BYTES:
        raise ValueError("selected member exceeds bounded transfer size")
    remainder, t_payload = fetch_exact_range(
        session, spec, offset + 30, offset + total - 1, budget
    )
    encoding = "utf-8" if record["flags"] & 0x800 else "cp437"
    name_bytes = remainder[:name_size]
    if (
        name_bytes.decode(encoding) != record["name"]
        or flags != record["flags"]
        or method != record["method"]
        or (not flags & 8 and crc != int(record["crc32"], 16))
    ):
        raise ValueError("local member header does not match the pinned index")
    if not flags & 8 and (
        compressed != record["compressed_bytes"]
        or uncompressed != record["uncompressed_bytes"]
    ):
        raise ValueError("local member sizes do not match the pinned index")
    decoder_record = {
        "name": record["name"],
        "name_bytes": name_bytes,
        "flags": flags,
        "method": method,
        "crc": int(record["crc32"], 16),
        "compressed": record["compressed_bytes"],
        "uncompressed": record["uncompressed_bytes"],
        "offset": offset,
    }
    raw_member = fixed + remainder
    payload = decode_local_member(raw_member, decoder_record)
    return payload, {
        "member_name_sha256": sha256(record["name"].encode()),
        "payload_sha256": sha256(payload),
        "zip_crc32_verified": True,
        "transferred_bytes": len(raw_member),
        "range_seconds": round(t_header + t_payload, 4),
    }


def summarize_pair(payloads, receipts):
    input_image = np.asarray(Image.open(io.BytesIO(payloads["image"])).convert("RGB"))
    if input_image.ndim != 3 or input_image.shape[2] != 3:
        raise ValueError("RGB input image failed to decode")
    image = Image.open(io.BytesIO(payloads["instance2d"]))
    instance_image = np.asarray(image)
    image_mode = image.mode
    semantic = np.asarray(Image.open(io.BytesIO(payloads["semantic2d"])))
    if instance_image.ndim != 2 or semantic.ndim != 2 or instance_image.shape != semantic.shape:
        raise ValueError("2D semantic and instance labels must be same-size scalar images")

    points = np.frombuffer(payloads["pointcloud"], dtype="<f4")
    if points.size % 4:
        raise ValueError("point cloud is not an XYZI float32 array")
    points = points.reshape(-1, 4)
    labels = np.frombuffer(payloads["labels3d"], dtype="<u4")
    if len(labels) != len(points):
        raise ValueError("point count and 3D label count differ")
    semantic3d = labels & 0xFFFF
    instance3d = labels >> 16

    keys2d = {
        (int(class_id), int(instance_id))
        for class_id, instance_id in zip(
            semantic.ravel(), instance_image.ravel(), strict=True
        )
        if int(class_id) > 0 and int(instance_id) > 0
    }
    keys3d = {
        (int(class_id), int(instance_id))
        for class_id, instance_id in zip(semantic3d, instance3d, strict=True)
        if int(class_id) > 0 and int(instance_id) > 0
    }
    matched_instance_class_keys = keys2d & keys3d

    ids2d, counts2d = np.unique(instance_image.astype(np.int64), return_counts=True)
    ids3d, counts3d = np.unique(instance3d.astype(np.int64), return_counts=True)
    count2d = {int(i): int(c) for i, c in zip(ids2d, counts2d, strict=True) if i > 0}
    count3d = {int(i): int(c) for i, c in zip(ids3d, counts3d, strict=True) if i > 0}
    common = sorted(count2d.keys() & count3d.keys())
    semantic_match = []
    class_pairs = Counter()
    for instance_id in common:
        classes2d = semantic[instance_image == instance_id]
        classes3d = semantic3d[instance3d == instance_id]
        if len(classes2d) and len(classes3d):
            mode2d = int(Counter(map(int, classes2d.ravel())).most_common(1)[0][0])
            mode3d = int(Counter(map(int, classes3d)).most_common(1)[0][0])
            semantic_match.append(mode2d == mode3d)
            class_pairs[(mode2d, mode3d)] += 1

    return {
        "rgb_input_shape": list(input_image.shape),
        "image_instance_mode": image_mode,
        "image_shape": list(instance_image.shape),
        "image_nonzero_instance_count": len(count2d),
        "point_count": len(points),
        "point_finite": bool(np.isfinite(points).all()),
        "point_nonzero_instance_count": len(count3d),
        "shared_nonzero_instance_ids": len(common),
        "shared_instance_semantic_class_matches": int(sum(semantic_match)),
        "shared_instance_semantic_class_mismatches": int(len(semantic_match) - sum(semantic_match)),
        "image_instance_class_key_count": len(keys2d),
        "pointcloud_instance_class_key_count": len(keys3d),
        "shared_instance_class_key_count": len(matched_instance_class_keys),
        "shared_instance_class_key_jaccard": round(
            len(matched_instance_class_keys) / max(len(keys2d | keys3d), 1), 6
        ),
        "shared_instance_semantic_class_pair_counts": [
            {"image_class_id": image_class, "pointcloud_class_id": point_class, "instances": count}
            for (image_class, point_class), count in sorted(class_pairs.items())
        ],
        "shared_id_jaccard": round(
            len(common) / max(len(count2d.keys() | count3d.keys()), 1), 6
        ),
        "shared_ids_semantic_mode_agreement": round(
            float(np.mean(semantic_match)), 6
        ) if semantic_match else None,
        "instance_id_values_sha256": sha256(
            json.dumps(common, separators=(",", ":")).encode()
        ),
        "member_receipts": receipts,
    }


def audit(
    index_path: Path,
    output_dir: Path,
    session,
    *,
    start_group_index=0,
    sample_count=SAMPLE_COUNT,
):
    index_bytes = index_path.read_bytes()
    if sha256(index_bytes) != INDEX_SHA256:
        raise ValueError("source archive-index SHA-256 mismatch")
    indexes = json.loads(index_bytes)
    selected, validation_group_count, complete_train_count = select_frames(
        indexes, start_group_index=start_group_index, sample_count=sample_count
    )
    archive_specs = {row["name"]: row for row in ARCHIVES}
    archive_names = {
        "image": "goose_2d_train.zip",
        "semantic2d": "goose_2d_train.zip",
        "instance2d": "goose_2d_train.zip",
        "pointcloud": "goose_3d_train.zip",
        "labels3d": "goose_3d_train.zip",
    }
    plan = {
        "source_index_sha256": INDEX_SHA256,
        "selection": "first complete paired frame per lexically ordered train-only scenario",
        "start_group_index": start_group_index,
        "sample_count": sample_count,
        "validation_scenario_count_excluded": validation_group_count,
        "complete_train_only_paired_frame_count": complete_train_count,
        "frames": [
            {
                "frame_key_sha256": sha256(key.encode()),
                "scenario_sha256": sha256(value["group"].encode())[:16],
                "members": {
                    kind: sha256(value[kind]["name"].encode())
                    for kind in sorted(archive_names)
                },
            }
            for key, value in selected
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    budget = {"remaining": MAX_TOTAL_SAMPLE_BYTES, "transferred": 0, "timings": []}
    frame_results = []
    for key, value in selected:
        payloads, receipts = {}, {}
        for kind in sorted(archive_names):
            spec = archive_specs[archive_names[kind]]
            payloads[kind], receipts[kind] = read_member_payload(
                session, spec, value[kind], budget
            )
        frame_results.append(
            {
                "frame_key_sha256": sha256(key.encode()),
                "scenario_sha256": sha256(value["group"].encode())[:16],
                **summarize_pair(payloads, receipts),
            }
        )
        del payloads
    rates = [
        row["transferred_bytes"] / max(row["range_seconds"], 1e-9) / 2**20
        for frame in frame_results
        for row in frame["member_receipts"].values()
    ]
    elapsed = sum(
        row["range_seconds"]
        for frame in frame_results
        for row in frame["member_receipts"].values()
    )
    result = {
        "plan_sha256": sha256(plan_path.read_bytes()),
        "source_index_sha256": INDEX_SHA256,
        "frames": frame_results,
        "transfer": {
            "members": sum(len(row["member_receipts"]) for row in frame_results),
            "bytes": budget["transferred"],
            "sum_range_seconds": round(elapsed, 4),
            "mean_member_mib_per_second": round(float(np.mean(rates)), 3),
            "pooled_mib_per_second": round(budget["transferred"] / max(elapsed, 1e-9) / 2**20, 3),
        },
        "raw_media_or_labels_retained": False,
        "train_only_scenarios": True,
        "point_model_loaded": False,
        "quality_evaluation": False,
        "sealed_audit_accessed": False,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-group-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    args = parser.parse_args()
    if args.start_group_index < 0 or args.sample_count < 1:
        parser.error("start-group-index must be >= 0 and sample-count must be >= 1")
    import requests

    with requests.Session() as session:
        result = audit(
            args.index_path,
            args.output_dir,
            session,
            start_group_index=args.start_group_index,
            sample_count=args.sample_count,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
