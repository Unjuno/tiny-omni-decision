"""Download a small, generation-pinned fresh DROID train-shard cohort."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .download_droid100_pinned import REPOSITORY_ROOT, download_object
except ImportError:
    from download_droid100_pinned import REPOSITORY_ROOT, download_object

EXPECTED_EXCLUDED = [110, 451, 715, 1019, 1033, 1139, 1186, 1242, 1497, 1573, 1696, 1718]
EXPECTED_FRESH = [580, 1635, 1018, 717, 577, 1634, 487, 748, 1768]


def validate_manifest(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("dataset_id") != "droid/droid"
        or manifest.get("dataset_version") != "1.4.0"
        or manifest.get("bucket") != "gresearch"
        or manifest.get("prefix") != "robotics/droid/1.0.0/"
        or manifest.get("split") != "train"
        or manifest.get("selection_seed") != 17
    ):
        raise ValueError("manifest does not pin the expected DROID train release")
    if manifest.get("excluded_previous_shards") != EXPECTED_EXCLUDED:
        raise ValueError("manifest must exclude every previously used shard")
    ranked = sorted(
        (index for index in range(2048) if index not in set(EXPECTED_EXCLUDED)),
        key=lambda index: hashlib.sha256(
            f"droid-r2-fresh-seed17:{index}".encode()
        ).hexdigest(),
    )
    # The first ranked shard (654) was excluded before acquisition because it
    # was the largest candidate and keeping nine shards preserves disk reserve.
    if ranked[:10] != [654, *EXPECTED_FRESH]:
        raise ValueError("seed-17 fresh-shard selection order changed")
    selected = manifest.get("selected_shards")
    if not isinstance(selected, list) or [int(row["index"]) for row in selected] != EXPECTED_FRESH:
        raise ValueError("manifest does not match the frozen nine-shard selection")
    if set(EXPECTED_EXCLUDED) & {int(row["index"]) for row in selected}:
        raise ValueError("fresh cohort reuses a previously selected shard")
    if sum(int(row["episodes"]) for row in selected) != int(
        manifest.get("expected_episode_count", -1)
    ):
        raise ValueError("selected episode counts do not match the frozen total")
    metadata = manifest.get("metadata_objects")
    if not isinstance(metadata, list) or len(metadata) != 3:
        raise ValueError("manifest must pin license and both schema objects")
    if {row["name"] for row in metadata} != {
        f"{manifest['prefix']}CC-BY-4.0",
        f"{manifest['prefix']}dataset_info.json",
        f"{manifest['prefix']}features.json",
    }:
        raise ValueError("unexpected DROID metadata object set")
    objects = [*metadata, *selected]
    if sum(int(row["size"]) for row in objects) != int(
        manifest.get("expected_total_bytes", -1)
    ):
        raise ValueError("pinned object sizes do not match the total")
    names = [row["name"] if "name" in row else row["object_name"] for row in objects]
    if len(names) != len(set(names)):
        raise ValueError("pinned object names must be unique")
    for row in objects:
        if not str(row["generation"]).isdigit() or len(
            __import__("base64").b64decode(row["md5Hash"], validate=True)
        ) != 16:
            raise ValueError("object generation or MD5 metadata is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "manifests/pretrained_reuse/droid-fresh-next-action-seed17-20261026.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("workers must be between 1 and 3")
    output = args.output_dir.resolve()
    if output.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("dataset output must stay outside the repository")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    objects = [
        {**row, "name": row.get("name", row.get("object_name"))}
        for row in [*manifest["metadata_objects"], *manifest["selected_shards"]]
    ]
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_object, output, manifest["bucket"], row)
            for row in objects
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"verified {result['name']} ({result['size']} bytes)", flush=True)
    data_root = output / manifest["prefix"]
    for name, expected_key in (
        ("dataset_info.json", "dataset_info_sha256"),
        ("features.json", "features_sha256"),
    ):
        actual = hashlib.sha256((data_root / name).read_bytes()).hexdigest()
        if actual != manifest[expected_key]:
            raise ValueError(f"downloaded {name} SHA-256 differs from pinned metadata")
    license_md5 = hashlib.md5(
        (data_root / "CC-BY-4.0").read_bytes(), usedforsecurity=False
    ).digest()
    if license_md5 != base64.b64decode(manifest["license_record"]["md5Hash"]):
        raise ValueError("downloaded CC-BY license text MD5 differs from pinned metadata")
    info = json.loads((data_root / "dataset_info.json").read_text(encoding="utf-8"))
    train = next(row for row in info["splits"] if row["name"] == "train")
    for shard in manifest["selected_shards"]:
        if int(train["shardLengths"][int(shard["index"])]) != int(shard["episodes"]):
            raise ValueError(f"dataset_info count mismatch for shard {shard['index']}")
    results.sort(key=lambda row: row["name"])
    receipt = {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selected_shards": manifest["selected_shards"],
        "episode_count": manifest["expected_episode_count"],
        "object_count": len(results),
        "total_bytes": sum(row["size"] for row in results),
        "duration_seconds": time.perf_counter() - started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "objects": results,
    }
    (output / "download-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: receipt[key] for key in (
                "dataset_id", "dataset_version", "manifest_sha256", "episode_count",
                "object_count", "total_bytes", "duration_seconds"
            )},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
