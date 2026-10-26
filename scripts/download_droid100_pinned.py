"""Download the generation-pinned, face-blurred DROID-100 debug subset."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def validate_manifest(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("dataset_id") != "droid/droid_100"
        or manifest.get("bucket") != "gresearch"
        or manifest.get("prefix") != "robotics/droid_100/1.0.0/"
        or manifest.get("expected_episode_count") != 100
    ):
        raise ValueError("manifest does not pin the expected DROID-100 release")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or len(objects) != 33:
        raise ValueError("manifest must contain two metadata objects and 31 TFRecord shards")
    names = [str(item.get("name", "")) for item in objects]
    if len(names) != len(set(names)):
        raise ValueError("GCS object names must be unique")
    if any(
        not name.startswith(manifest["prefix"])
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        for name in names
    ):
        raise ValueError("GCS object path escapes the pinned prefix")
    shard_names = sorted(name for name in names if "tfrecord-" in name)
    expected_shards = [
        f"{manifest['prefix']}r2d2_faceblur-train.tfrecord-{i:05d}-of-00031"
        for i in range(31)
    ]
    if shard_names != expected_shards:
        raise ValueError("manifest must pin all 31 face-blurred train shards in order")
    expected_names = {
        f"{manifest['prefix']}dataset_info.json",
        f"{manifest['prefix']}features.json",
        *expected_shards,
    }
    if set(names) != expected_names:
        raise ValueError("manifest contains unexpected or missing DROID-100 objects")
    total_bytes = 0
    for item in objects:
        if int(item["size"]) < 0 or not re.fullmatch(r"[0-9]+", str(item["generation"])):
            raise ValueError("object size or generation is invalid")
        try:
            digest = base64.b64decode(item["md5Hash"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("GCS MD5 must be valid base64") from exc
        if len(digest) != 16:
            raise ValueError("GCS MD5 digest has unexpected length")
        total_bytes += int(item["size"])
    if total_bytes != int(manifest.get("expected_total_bytes", -1)):
        raise ValueError("manifest total byte count does not match its objects")


def safe_object_path(output_root: Path, object_name: str) -> Path:
    root = output_root.resolve()
    relative = Path(object_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("GCS object name must be a safe relative path")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("resolved GCS object path escapes output directory")
    return target


def file_hashes(path: Path) -> dict[str, Any]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return {"size": size, "md5Hash": base64.b64encode(md5.digest()).decode("ascii"),
            "sha256": sha256.hexdigest()}


def verify_file(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    hashes = file_hashes(path)
    if hashes["size"] != int(item["size"]) or hashes["md5Hash"] != item["md5Hash"]:
        raise ValueError(f"size/MD5 mismatch for {item['name']}; preserved at {path}")
    return hashes


def download_object(
    output_root: Path,
    bucket: str,
    item: dict[str, Any],
    *,
    retries: int = 3,
) -> dict[str, Any]:
    target = safe_object_path(output_root, str(item["name"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {"name": item["name"], **verify_file(target, item), "reused_existing": True}
    part = target.with_name(target.name + ".part")
    escaped = urllib.parse.quote(str(item["name"]), safe="")
    query = urllib.parse.urlencode({"generation": item["generation"], "alt": "media"})
    url = f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{escaped}?{query}"
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        if offset > int(item["size"]):
            raise ValueError(f"partial file exceeds pinned size: {part}")
        headers = {"User-Agent": "tiny-omni-decision/metadata-pinned-droid100"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status = response.status
                if offset and status != 206:
                    raise ValueError("server did not honor byte-range resume")
                if not offset and status != 200:
                    raise ValueError(f"unexpected HTTP status {status}")
                content_range = response.headers.get("Content-Range", "")
                if offset and not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError("resumed response starts at an unexpected byte")
                mode = "ab" if part.exists() else "xb"
                with part.open(mode) as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            hashes = verify_file(part, item)
            os.replace(part, target)
            return {"name": item["name"], **hashes, "reused_existing": False}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"failed downloading {item['name']}: {exc}") from exc
            time.sleep(2**attempt)
    raise RuntimeError(f"download retries exhausted for {item['name']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "manifests/pretrained_reuse/droid100-pinned-objects-20261026.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    output = args.output_dir.resolve()
    if output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("dataset download output must stay outside the repository")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
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
    results.sort(key=lambda row: row["name"])
    receipt = {
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "object_count": len(results),
        "total_bytes": sum(row["size"] for row in results),
        "duration_seconds": time.perf_counter() - started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "objects": results,
    }
    receipt_path = output / "download-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in (
        "dataset_id", "manifest_sha256", "object_count", "total_bytes", "duration_seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
