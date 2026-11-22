"""Download an immutable, publisher-hash-checked HABIT train/validation pilot."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

DATASET = "configinc/HABIT"
REVISION = "bae97d251907743f8fb6db7e70c61d361df4dc1e"
RETRYABLE = {429, 500, 502, 503, 504}


def validate_inventory(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Return approved train/validation files, rejecting audit or path drift."""
    if inventory.get("dataset") != DATASET or inventory.get("revision") != REVISION:
        raise ValueError("inventory dataset identity is missing or unexpected")
    if inventory.get("payload_downloaded") is not False:
        raise ValueError("inventory must be a metadata-only preflight")
    if inventory.get("sealed_audit_accessed") is not False:
        raise ValueError("sealed audit must remain unopened")
    items = inventory.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("inventory items must be a nonempty list")
    seen: set[tuple[str, int, str]] = set()
    pairs: dict[tuple[str, int], set[str]] = {}
    split_by_episode: dict[int, str] = {}
    validated = []
    for item in items:
        if item.get("split") not in {"train", "validation"}:
            raise ValueError("only train and validation payloads may be downloaded")
        episode = item.get("episode_index")
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise ValueError("episode_index must be a nonnegative integer")
        kind = item.get("kind")
        chunk = episode // 1000
        if kind == "parquet":
            expected_path = f"full/data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"
        elif kind == "video":
            expected_path = (
                f"full/videos/chunk-{chunk:03d}/observation.images.front_view/"
                f"episode_{episode:06d}.mp4"
            )
        else:
            raise ValueError("unsupported HABIT payload kind")
        if item.get("path") != expected_path:
            raise ValueError("payload path does not match episode and approved camera")
        size = item.get("size")
        digest = item.get("sha256")
        if (isinstance(size, bool) or not isinstance(size, int) or size < 1
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("payload size or SHA-256 is invalid")
        key = (item["split"], episode, kind)
        if key in seen:
            raise ValueError("duplicate payload inventory entry")
        seen.add(key)
        pair = (item["split"], episode)
        pairs.setdefault(pair, set()).add(kind)
        previous_split = split_by_episode.setdefault(episode, item["split"])
        if previous_split != item["split"]:
            raise ValueError("an episode appears in more than one split")
        validated.append(item)
    if any(kinds != {"video", "parquet"} for kinds in pairs.values()):
        raise ValueError("each episode must have exactly one video and Parquet")
    if len(validated) != inventory.get("file_count"):
        raise ValueError("inventory file_count does not match its items")
    if len(pairs) != inventory.get("episode_count"):
        raise ValueError("inventory episode_count does not match its items")
    if sum(item["size"] for item in validated) != inventory.get("total_bytes"):
        raise ValueError("inventory total_bytes does not match its items")
    if inventory.get("camera_view") != "observation.images.front_view":
        raise ValueError("only the preflighted front_view is approved")
    if not any(item["kind"] == "video" for item in validated):
        raise ValueError("inventory contains no video payloads")
    if not any(item["kind"] == "parquet" for item in validated):
        raise ValueError("inventory contains no Parquet payloads")
    return validated


def download_habit_pilot(
    inventory_path: Path,
    output_dir: Path,
    *,
    workers: int = 4,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Fetch only inventory-pinned items into a new directory and verify hashes."""
    if workers < 1 or max_retries < 0:
        raise ValueError("workers must be positive and retries nonnegative")
    if output_dir.exists():
        raise ValueError("output directory must be new; preserve previous attempts")
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    items = validate_inventory(inventory)
    output_dir.mkdir(parents=True)
    started = time.perf_counter()

    def fetch(item: dict[str, Any]) -> dict[str, Any]:
        split = item["split"]
        folder = "video" if item["kind"] == "video" else "parquet"
        suffix = ".mp4" if item["kind"] == "video" else ".parquet"
        target = output_dir / split / folder / f"episode_{item['episode_index']:06d}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/datasets/{DATASET}/resolve/"
            f"{inventory['revision']}/{quote(item['path'], safe='/')}?download=true"
        )
        temporary = target.with_suffix(target.suffix + ".part")
        for attempt in range(max_retries + 1):
            digest = hashlib.sha256()
            size = 0
            try:
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    if response.status_code in RETRYABLE and attempt < max_retries:
                        time.sleep(min(2**attempt, 30))
                        continue
                    response.raise_for_status()
                    with temporary.open("xb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                                digest.update(chunk)
                                size += len(chunk)
                    break
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                if attempt == max_retries:
                    raise
                time.sleep(min(2**attempt, 30))
        if size != item["size"] or digest.hexdigest() != item["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"publisher size/SHA-256 mismatch: {item['path']}")
        temporary.replace(target)
        return {
            "split": split,
            "episode_index": item["episode_index"],
            "kind": item["kind"],
            "path": item["path"],
            "local_path": str(target),
            "bytes": size,
            "sha256": digest.hexdigest(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        downloaded = list(pool.map(fetch, items))
    video_splits_by_hash: dict[str, set[str]] = {}
    for item in downloaded:
        if item["kind"] == "video":
            video_splits_by_hash.setdefault(item["sha256"], set()).add(item["split"])
    cross_split_duplicates = sorted(
        digest for digest, splits in video_splits_by_hash.items() if len(splits) > 1
    )
    receipt = {
        "dataset": DATASET,
        "revision": inventory["revision"],
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "file_count": len(downloaded),
        "bytes": sum(item["bytes"] for item in downloaded),
        "files": downloaded,
        "cross_split_duplicate_video_sha256": cross_split_duplicates,
        "cross_split_media_overlap": bool(cross_split_duplicates),
        "sealed_audit_accessed": False,
        "wall_seconds": time.perf_counter() - started,
    }
    (output_dir / "download-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    if cross_split_duplicates:
        raise ValueError("duplicate video bytes cross train/validation; inspect receipt")
    return receipt


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()
    result = download_habit_pilot(
        args.inventory,
        args.output_dir,
        workers=args.workers,
        max_retries=args.max_retries,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
