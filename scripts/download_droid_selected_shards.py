"""Download a seed-17, generation-pinned sample of full DROID RLDS shards."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .download_droid100_pinned import REPOSITORY_ROOT, download_object
except ImportError:
    from download_droid100_pinned import REPOSITORY_ROOT, download_object


def validate_manifest(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("dataset_id") != "droid/droid"
        or manifest.get("dataset_version") != "1.4.0"
        or manifest.get("bucket") != "gresearch"
        or manifest.get("prefix") != "robotics/droid/1.0.0/"
        or manifest.get("split") != "train"
    ):
        raise ValueError("manifest does not pin the expected full DROID release and split")
    if manifest.get("selection_seed") != 17:
        raise ValueError("selection seed must remain 17")

    selected = manifest.get("selected_shards")
    if not isinstance(selected, list) or len(selected) != 12:
        raise ValueError("manifest must select exactly 12 predetermined shards")
    indices = [int(row["index"]) for row in selected]
    expected_indices = sorted(random.Random(17).sample(range(2048), 12))
    if indices != expected_indices:
        raise ValueError("selected shards must match the fixed seed-17 sample")
    for row in selected:
        expected_name = (
            f"{manifest['prefix']}r2d2_faceblur-train.tfrecord-"
            f"{int(row['index']):05d}-of-02048"
        )
        if row.get("object_name") != expected_name or int(row["episodes"]) <= 0:
            raise ValueError("selected shard identity or episode count is invalid")
    expected_episodes = sum(int(row["episodes"]) for row in selected)
    if expected_episodes != int(manifest.get("expected_episode_count", -1)):
        raise ValueError("selected shard episode counts do not match the manifest total")

    objects = manifest.get("objects")
    if not isinstance(objects, list) or len(objects) != 15:
        raise ValueError("manifest must contain three metadata objects and 12 shards")
    names = [str(item.get("name", "")) for item in objects]
    expected_names = {
        f"{manifest['prefix']}CC-BY-4.0",
        f"{manifest['prefix']}dataset_info.json",
        f"{manifest['prefix']}features.json",
        *(str(row["object_name"]) for row in selected),
    }
    if len(names) != len(set(names)) or set(names) != expected_names:
        raise ValueError("manifest contains an unexpected or duplicate GCS object")

    total_bytes = 0
    for item in objects:
        if int(item["size"]) < 0 or not str(item["generation"]).isdigit():
            raise ValueError("object size or generation is invalid")
        if len(base64.b64decode(item["md5Hash"], validate=True)) != 16:
            raise ValueError("GCS MD5 digest is invalid")
        total_bytes += int(item["size"])
    if total_bytes != int(manifest.get("expected_total_bytes", -1)):
        raise ValueError("manifest total byte count does not match its objects")


def verify_metadata(output: Path, manifest: dict[str, Any]) -> None:
    prefix = manifest["prefix"]
    dataset_info_path = output / f"{prefix}dataset_info.json"
    features_path = output / f"{prefix}features.json"
    dataset_info_sha256 = hashlib.sha256(dataset_info_path.read_bytes()).hexdigest()
    if dataset_info_sha256 != manifest["dataset_info_sha256"]:
        raise ValueError("downloaded dataset_info.json SHA-256 does not match the manifest")
    if hashlib.sha256(features_path.read_bytes()).hexdigest() != manifest["features_sha256"]:
        raise ValueError("downloaded features.json SHA-256 does not match the manifest")
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    if dataset_info.get("version") != manifest["dataset_version"]:
        raise ValueError("downloaded dataset_info.json has an unexpected version")
    train = next(
        (row for row in dataset_info.get("splits", []) if row.get("name") == manifest["split"]),
        None,
    )
    if train is None or len(train.get("shardLengths", [])) != 2048:
        raise ValueError("downloaded dataset_info.json does not describe the pinned train split")
    for row in manifest["selected_shards"]:
        if int(train["shardLengths"][int(row["index"])]) != int(row["episodes"]):
            raise ValueError("selected shard episode count differs from dataset_info.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "manifests/pretrained_reuse/droid-selected-shards-seed17-20261026.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    output = args.output_dir.resolve()
    if output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("dataset output must stay outside the repository")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_object, output, manifest["bucket"], item)
            for item in manifest["objects"]
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"verified {result['name']} ({result['size']} bytes)", flush=True)
    verify_metadata(output, manifest)
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
    print(json.dumps({key: receipt[key] for key in (
        "dataset_id", "dataset_version", "manifest_sha256", "episode_count",
        "object_count", "total_bytes", "duration_seconds",
    )}, indent=2))


if __name__ == "__main__":
    main()
