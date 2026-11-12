"""Audit DROID 1.0.1 episode metadata coverage without reading media or labels."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

USER_AGENT = "tiny-omni-decision-metadata-audit/1.0"
SAMPLE_DOMAIN = b"tiny-omni-decision:droid-1.0.1:metadata-sample:v1\0"
GROUP_DOMAIN = b"tiny-omni-decision:droid-1.0.1:scene-group:v1\0"
BUILDING_DOMAIN = b"tiny-omni-decision:droid-1.0.1:building:v1\0"
COLLECTOR_DOMAIN = b"tiny-omni-decision:droid-1.0.1:collector:v1\0"
ASSET_DOMAIN = b"tiny-omni-decision:droid-1.0.1:asset-ref:v1\0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_identity(domain: bytes, value: str) -> str:
    return hashlib.sha256(domain + value.strip().casefold().encode("utf-8")).hexdigest()


def metadata_objects(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("name", "")).startswith(prefix)
        and Path(str(row.get("name", ""))).name.startswith("metadata_")
        and str(row.get("name", "")).endswith(".json")
    ]
    names = [str(row["name"]) for row in selected]
    if len(names) != len(set(names)):
        raise ValueError("Inventory contains duplicate metadata object names")
    return selected


def select_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count < 1 or count > len(rows):
        raise ValueError(f"sample size must be between 1 and {len(rows)}")
    ranked = []
    seed_bytes = str(seed).encode("ascii") + b"\0"
    for row in rows:
        name = str(row["name"])
        rank = hashlib.sha256(SAMPLE_DOMAIN + seed_bytes + name.encode("utf-8")).digest()
        ranked.append((rank, name, row))
    return [row for _, _, row in sorted(ranked)[:count]]


def normalize_object_ref(value: Any, lab: Any = None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    ref = value.strip()
    parsed = urlparse(ref)
    if parsed.scheme == "gs":
        ref = f"{parsed.netloc}/{parsed.path.lstrip('/')}"
        if ref.startswith("gresearch/"):
            ref = ref[len("gresearch/") :]
    elif parsed.scheme in {"http", "https"}:
        ref = unquote(parsed.path).lstrip("/")
        if ref.startswith("gresearch/"):
            ref = ref[len("gresearch/") :]
    else:
        ref = unquote(ref).lstrip("/")
    if ref.startswith("robotics/"):
        return ref
    if ref.startswith("droid_raw/"):
        return f"robotics/{ref}"
    if ref.startswith("1.0.1/"):
        return f"robotics/droid_raw/{ref}"
    if isinstance(lab, str) and lab.strip() and not ref.startswith(f"{lab.strip()}/"):
        return f"robotics/droid_raw/1.0.1/{lab.strip()}/{ref}"
    return f"robotics/droid_raw/1.0.1/{ref}"


def verify_payload(row: dict[str, Any], payload: bytes) -> str:
    expected_size = int(row["size"])
    if len(payload) != expected_size:
        raise ValueError(f"size mismatch: expected {expected_size}, got {len(payload)}")
    actual_md5 = base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")
    if actual_md5 != row["md5Hash"]:
        raise ValueError("GCS MD5 mismatch")
    return hashlib.sha256(payload).hexdigest()


def parse_group_metadata(payload: bytes, inventory_names: set[str]) -> dict[str, Any]:
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError("episode metadata root must be an object")
    building = record.get("building")
    scene_id = record.get("scene_id")
    collector = record.get("user_id") or record.get("user")
    if not isinstance(building, str) or not building.strip():
        raise ValueError("missing stable building identifier")
    if (
        isinstance(scene_id, bool)
        or not isinstance(scene_id, (str, int))
        or not str(scene_id).strip()
    ):
        raise ValueError("missing stable scene identifier")
    if not isinstance(collector, str) or not collector.strip():
        raise ValueError("missing collector identity")
    lab = record.get("lab")
    group_key = f"{building.strip().casefold()}\0{str(scene_id).strip()}"
    scene_hash = digest_identity(GROUP_DOMAIN, group_key)
    building_hash = digest_identity(BUILDING_DOMAIN, building)
    collector_hash = digest_identity(COLLECTOR_DOMAIN, collector)

    video_refs = [
        normalize_object_ref(value, lab)
        for key, value in record.items()
        if key.endswith("_mp4_path") and value
    ]
    video_refs = [ref for ref in video_refs if ref]
    trajectory_ref = normalize_object_ref(record.get("hdf5_path"), lab)
    video_hashes = [
        hashlib.sha256(ASSET_DOMAIN + ref.encode("utf-8")).hexdigest() for ref in video_refs
    ]
    return {
        "building_scene_id_tuple_sha256": scene_hash,
        "building_sha256": building_hash,
        "collector_sha256": collector_hash,
        "video_asset_sha256": video_hashes,
        "video_refs": len(video_refs),
        "video_refs_present": sum(ref in inventory_names for ref in video_refs),
        "trajectory_ref_present": bool(trajectory_ref and trajectory_ref in inventory_names),
    }


def video_reference_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    references = [record["video_asset_sha256"] for record in records]
    per_record_unique = [set(row) for row in references]
    all_unique = {asset for row in per_record_unique for asset in row}
    repeated_within_record = sum(
        len(row) - len(unique) for row, unique in zip(references, per_record_unique, strict=True)
    )
    distinct_record_references = sum(len(unique) for unique in per_record_unique)
    return {
        "reference_occurrences": sum(len(row) for row in references),
        "unique_asset_references": len(all_unique),
        "within_record_alias_occurrences": repeated_within_record,
        "cross_record_duplicate_occurrences": distinct_record_references - len(all_unique),
    }


def object_url(row: dict[str, Any]) -> str:
    name = quote(str(row["name"]), safe="")
    generation = quote(str(row["generation"]), safe="")
    return f"https://storage.googleapis.com/storage/v1/b/gresearch/o/{name}?alt=media&generation={generation}"


def fetch_metadata(
    row: dict[str, Any], inventory_names: set[str], timeout: float
) -> dict[str, Any]:
    request = Request(
        object_url(row), headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"unexpected HTTP status {response.status}")
                payload = response.read(1_000_001)
                if len(payload) > 1_000_000:
                    raise ValueError("metadata object exceeded 1 MB safety limit")
            content_sha256 = verify_payload(row, payload)
            group = parse_group_metadata(payload, inventory_names)
            return {
                "name": row["name"],
                "generation": str(row["generation"]),
                "size": int(row["size"]),
                "md5Hash": row["md5Hash"],
                "content_sha256": content_sha256,
                **group,
            }
        except (
            Exception
        ) as error:  # retry transport errors; integrity/schema errors still fail after retries
            last_error = error
            if attempt < 3:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(
        f"metadata fetch failed for a selected object ({type(last_error).__name__})"
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")

    source_inventory_sha256 = sha256_file(args.inventory)
    with args.inventory.open("r", encoding="utf-8") as stream:
        inventory = json.load(stream)
    if not isinstance(inventory, list):
        raise ValueError("raw object inventory must be a JSON array")
    names = {str(row["name"]) for row in inventory}
    prefix = "robotics/droid_raw/1.0.1/"
    candidates = metadata_objects(inventory, prefix)
    selected = select_sample(candidates, args.sample_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_descriptor = [
        {
            "name": row["name"],
            "generation": str(row["generation"]),
            "size": int(row["size"]),
            "md5Hash": row["md5Hash"],
        }
        for row in selected
    ]
    descriptor_bytes = (
        json.dumps(sample_descriptor, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    sample_order_sha256 = hashlib.sha256(descriptor_bytes).hexdigest()
    selected_path = args.output_dir / "selected-metadata-objects.json"
    selected_path.write_bytes(descriptor_bytes)

    receipts_path = args.output_dir / "metadata-receipts.jsonl"
    cached: dict[str, dict[str, Any]] = {}
    if receipts_path.exists():
        for line in receipts_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                receipt = json.loads(line)
                if "building_scene_id_tuple_sha256" not in receipt:
                    receipt["building_scene_id_tuple_sha256"] = receipt.pop("scene_group_sha256")
                cached[receipt["name"]] = receipt

    selected_by_name = {str(row["name"]): row for row in selected}
    valid_cached: dict[str, dict[str, Any]] = {}
    for name, receipt in cached.items():
        row = selected_by_name.get(name)
        if row and all(
            receipt.get(field) == expected
            for field, expected in (
                ("generation", str(row["generation"])),
                ("size", int(row["size"])),
                ("md5Hash", row["md5Hash"]),
            )
        ):
            valid_cached[name] = receipt

    pending = [row for row in selected if str(row["name"]) not in valid_cached]
    print(f"inventory_sha256={source_inventory_sha256}", flush=True)
    print(
        f"metadata_objects={len(candidates)} sample={len(selected)} cached={len(valid_cached)}",
        flush=True,
    )
    print(f"sample_order_sha256={sample_order_sha256}", flush=True)

    done = 0
    with (
        ThreadPoolExecutor(max_workers=args.workers) as pool,
        receipts_path.open("a", encoding="utf-8", newline="\n") as receipt_stream,
    ):
        futures = {pool.submit(fetch_metadata, row, names, args.timeout): row for row in pending}
        for future in as_completed(futures):
            receipt = future.result()
            receipt_stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
            done += 1
            if done % 100 == 0:
                receipt_stream.flush()
            if done % 500 == 0 or done == len(pending):
                print(f"verified={len(valid_cached) + done}/{len(selected)}", flush=True)
    receipts_by_name = {**valid_cached}
    for line in receipts_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            receipt = json.loads(line)
            if receipt["name"] in selected_by_name:
                receipts_by_name[receipt["name"]] = receipt
    ordered = [receipts_by_name[str(row["name"])] for row in selected]
    metadata_scene_keys = {row["building_scene_id_tuple_sha256"] for row in ordered}
    buildings = {row["building_sha256"] for row in ordered}
    collectors = {row["collector_sha256"] for row in ordered}
    group_collectors: dict[str, set[str]] = {}
    collector_groups: dict[str, set[str]] = {}
    video_assets: list[str] = []
    for row in ordered:
        group_collectors.setdefault(row["building_scene_id_tuple_sha256"], set()).add(
            row["collector_sha256"]
        )
        collector_groups.setdefault(row["collector_sha256"], set()).add(
            row["building_scene_id_tuple_sha256"]
        )
        video_assets.extend(row["video_asset_sha256"])
    video_counts = video_reference_counts(ordered)
    summary = {
        "schema_version": 1,
        "source": "Google Research DROID raw 1.0.1 metadata JSON only",
        "seed": args.seed,
        "sample_size": len(ordered),
        "sample_order_sha256": sample_order_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "metadata_object_count": len(candidates),
        "unique_building_scene_id_tuples_in_sample": len(metadata_scene_keys),
        "physical_scene_count_verified": None,
        "official_scene_deduplication_required": True,
        "unique_buildings_in_sample": len(buildings),
        "unique_collectors_in_sample": len(collectors),
        "building_scene_id_tuples_with_multiple_collectors": sum(
            len(value) > 1 for value in group_collectors.values()
        ),
        "collectors_spanning_multiple_building_scene_id_tuples": sum(
            len(value) > 1 for value in collector_groups.values()
        ),
        "video_reference_counts": video_counts,
        "video_references_present_in_object_inventory": sum(
            row["video_refs_present"] for row in ordered
        ),
        "trajectory_references_present_in_object_inventory": sum(
            row["trajectory_ref_present"] for row in ordered
        ),
        "payloads_downloaded": {
            "metadata_json": len(ordered),
            "video": 0,
            "hdf5": 0,
            "svo": 0,
            "npy": 0,
        },
        "task_text_fields_used_or_logged": False,
        "success_label_used_or_logged": False,
        "sealed_audit_opened": False,
        "model_loaded_or_run": False,
        "group_split_created": False,
        "selected_object_manifest_sha256": sha256_file(selected_path),
        "metadata_receipts_sha256": sha256_file(receipts_path),
    }
    summary_path = args.output_dir / "cohort-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
