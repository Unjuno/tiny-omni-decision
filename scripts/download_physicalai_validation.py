"""Download one non-reserved split and verify pinned LFS hashes."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests


def download_split(
    manifest_path: Path,
    output_dir: Path,
    *,
    split: str = "validation",
    workers: int = 8,
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError("only train and validation payloads may be downloaded")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    source = manifest["source"]
    revision = source["revision"]
    dataset_id = source["dataset_id"]
    subset = source["subset"]
    api = f"https://huggingface.co/api/datasets/{dataset_id}/revision/{revision}?blobs=true"
    response = requests.get(api, timeout=90)
    response.raise_for_status()
    inventory = response.json()
    if inventory.get("sha") != revision:
        raise ValueError("publisher API returned a different revision")
    files = {item["rfilename"]: item for item in inventory["siblings"]}
    selected = [row for row in manifest["episodes"] if row["split"] == split]
    if not selected:
        raise ValueError(f"the project {split} split is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for row in selected:
        episode = row["episode_index"]
        chunk = episode // 1000
        paths = {
            "video": row["video_path"],
            "parquet": f"{subset}/data/chunk-{chunk:03d}/episode_{episode:06d}.parquet",
        }
        for kind, path in paths.items():
            remote = files[path]
            lfs = remote.get("lfs") or {}
            digest = lfs.get("sha256")
            size = remote.get("size")
            if not digest or not isinstance(size, int):
                raise ValueError(f"missing pinned LFS metadata for {path}")
            items.append(
                {
                    "episode": episode,
                    "kind": kind,
                    "path": path,
                    "size": size,
                    "sha256": digest,
                }
            )

    started = time.perf_counter()

    def fetch(item: dict[str, Any]) -> dict[str, Any]:
        suffix = ".mp4" if item["kind"] == "video" else ".parquet"
        target = output_dir / f"{item['kind']}-episode-{item['episode']:06d}{suffix}"
        url = (
            f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/"
            f"{item['path']}?download=true"
        )
        start = time.perf_counter()
        result = requests.get(url, timeout=(30, 180))
        result.raise_for_status()
        payload = result.content
        actual_hash = hashlib.sha256(payload).hexdigest()
        if len(payload) != item["size"] or actual_hash != item["sha256"]:
            raise ValueError(f"publisher hash/size mismatch for {item['path']}")
        target.write_bytes(payload)
        return {
            **item,
            "local_path": str(target),
            "seconds": time.perf_counter() - start,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(fetch, items))
    result = {
        "dataset_id": dataset_id,
        "revision": revision,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        f"{split}_episode_indices": [row["episode_index"] for row in selected],
        "split": split,
        "reserved_holdout_payload_downloaded": False,
        "files": receipts,
        "total_bytes": sum(item["size"] for item in receipts),
        "wall_seconds": time.perf_counter() - started,
    }
    receipt_path = output_dir / "download-receipt.json"
    receipt_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {
        "receipt_path": str(receipt_path),
        "files": len(receipts),
        "bytes": result["total_bytes"],
        "wall_seconds": result["wall_seconds"],
    }


def download_validation(
    manifest_path: Path, output_dir: Path, *, workers: int = 8
) -> dict[str, Any]:
    """Backward-compatible validation-only entry point."""
    return download_split(manifest_path, output_dir, split="validation", workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    args = parser.parse_args()
    print(
        json.dumps(
            download_split(args.manifest, args.output_dir, split=args.split, workers=args.workers)
        )
    )


if __name__ == "__main__":
    main()
