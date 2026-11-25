"""Fetch only the frozen train/development files in a RoboPro probe sample."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests

DATASET = "Hoshipu/RoboPro"
UPSTREAM_ROOT = "lerobot/roboreal_all_80tasks"
CHUNK_SIZE = 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected(sample: dict) -> list[tuple[str, int]]:
    selected = []
    for split in ("train", "development"):
        groups = sample[split]
        prompts = [row["prompt_sha256"] for row in groups]
        if len(prompts) != len(set(prompts)):
            raise ValueError(f"duplicate prompt group in {split}")
        for group in groups:
            for episode in group["episodes"]:
                selected.append((split, int(episode["episode_id"])))
    if len({episode_id for _, episode_id in selected}) != len(selected):
        raise ValueError("episode IDs overlap across partitions")
    return selected


def _download_one(
    revision: str,
    split: str,
    episode_id: int,
    kind: str,
    root: Path,
    timeout: float,
) -> dict:
    chunk = episode_id // CHUNK_SIZE
    stem = f"episode_{episode_id:06d}"
    if kind == "parquet":
        filename = f"{UPSTREAM_ROOT}/data/chunk-{chunk:03d}/{stem}.parquet"
        local = root / split / "data" / f"{stem}.parquet"
    else:
        filename = (
            f"{UPSTREAM_ROOT}/videos/chunk-{chunk:03d}/observation.images.cam_high/{stem}.mp4"
        )
        local = root / split / "video" / f"{stem}_cam_high.mp4"
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_suffix(local.suffix + ".partial")
    url = (
        f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/"
        f"{quote(filename, safe='/')}?download=true"
    )
    last_error = None
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            byte_count = 0
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        stream.write(block)
                        digest.update(block)
                        byte_count += len(block)
                stream.flush()
                os.fsync(stream.fileno())
            if byte_count == 0:
                raise ValueError(f"empty response for {filename}")
            os.replace(temporary, local)
            return {
                "split": split,
                "episode_id": episode_id,
                "kind": kind,
                "upstream_path": filename,
                "relative_path": str(local.relative_to(root)),
                "bytes": byte_count,
                "sha256": digest.hexdigest(),
                "etag": response.headers.get("ETag"),
                "http_status": response.status_code,
            }
        except Exception as exc:  # noqa: BLE001 - bounded retries, then fail closed
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(1 + attempt)
    raise RuntimeError(f"failed to fetch pinned file {filename}: {last_error}")


def run(sample_path: Path, output_root: Path, workers: int, timeout: float) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite acquisition: {output_root}")
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    sample_bytes = sample_path.read_bytes()
    sample = json.loads(sample_bytes)
    if sample.get("dataset") != DATASET or len(sample.get("revision", "")) != 40:
        raise ValueError("sample manifest has an unexpected source or revision")
    selected = _selected(sample)
    output_root.mkdir(parents=True)
    (output_root / "sample-manifest.json").write_bytes(sample_bytes)
    items = [
        (split, episode_id, kind) for split, episode_id in selected for kind in ("parquet", "mp4")
    ]
    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_one,
                sample["revision"],
                split,
                episode_id,
                kind,
                output_root,
                timeout,
            ): (split, episode_id, kind)
            for split, episode_id, kind in items
        }
        for future in concurrent.futures.as_completed(futures):
            receipts.append(future.result())
    receipts.sort(key=lambda row: (row["split"], row["episode_id"], row["kind"]))
    manifest = {
        "schema_version": "robopro-action-probe-download-receipt-v1",
        "dataset": DATASET,
        "revision": sample["revision"],
        "sample_manifest_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "files": receipts,
        "episode_count": len(selected),
        "total_bytes": sum(row["bytes"] for row in receipts),
        "file_hashes_verified_locally": all(
            sha256_file(output_root / row["relative_path"]) == row["sha256"] for row in receipts
        ),
    }
    if not manifest["file_hashes_verified_locally"]:
        raise ValueError("local receipt hash verification failed")
    receipt_path = output_root / "download-receipt.json"
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    receipt_path.write_bytes(payload)
    return {
        "status": "download_complete",
        "episode_count": len(selected),
        "file_count": len(receipts),
        "total_bytes": manifest["total_bytes"],
        "sample_manifest_sha256": manifest["sample_manifest_sha256"],
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "output": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.sample_manifest.resolve(strict=True),
                args.output_root.resolve(),
                args.workers,
                args.timeout_seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
