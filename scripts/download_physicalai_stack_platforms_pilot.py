"""Download only frozen train/validation pilot episodes with LFS hash checks."""

from __future__ import annotations

import argparse
import concurrent.futures
import email.utils
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

RETRYABLE_HTTP_STATUSES = frozenset({429, 499, 500, 502, 503, 504})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hub_auth_headers() -> dict[str, str]:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return {}
    token = get_token()
    if token is None:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
    value = response.headers.get("Retry-After")
    if value is not None:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_time = email.utils.parsedate_to_datetime(value).timestamp()
                return max(0.0, retry_time - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    return min(2.0 * (2**attempt), 120.0)


def download_pilot(
    manifest_path: Path,
    output_dir: Path,
    *,
    workers: int = 8,
    max_retries: int = 5,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    source = manifest["source"]
    dataset_id = source["dataset_id"]
    revision = source["revision"]
    subset = source["subset"]
    selection = manifest["pilot_selection"]
    train_ids = selection["train_episode_indices"]
    validation_ids = selection["validation_episode_indices"]
    if set(train_ids) & set(validation_ids):
        raise ValueError("pilot train and validation episode IDs overlap")
    if len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(
        set(validation_ids)
    ):
        raise ValueError("pilot episode ID lists must not contain duplicates")
    if manifest["split_policy"].get("allocation", "").find("reserved_holdout") < 0:
        raise ValueError("manifest does not declare a reserved holdout partition")
    scene_status = manifest["grouping"].get("scene_or_layout_identity_status", "")
    if not scene_status.startswith("UNKNOWN"):
        raise ValueError("unexpectedly missing the explicit scene/layout limitation")

    by_episode = {row["episode_index"]: row for row in manifest["episode_assignments"]}
    requested = {"train": train_ids, "validation": validation_ids}
    items = []
    for split, episode_ids in requested.items():
        for episode in episode_ids:
            row = by_episode.get(episode)
            if row is None or row["split"] != split:
                raise ValueError(f"pilot {split} episode {episode} is not assigned to {split}")
            items.extend(
                [
                    {
                        "episode": episode,
                        "split": split,
                        "kind": "video",
                        "path": row["video_path"],
                        "size": row["video_bytes"],
                        "sha256": row["video_sha256"],
                    },
                    {
                        "episode": episode,
                        "split": split,
                        "kind": "parquet",
                        "path": row["parquet_path"],
                        "size": row["parquet_bytes"],
                        "sha256": row["parquet_sha256"],
                    },
                ]
            )
    if any(item["split"] == "reserved_holdout" for item in items):
        raise ValueError("reserved holdout payload may not be downloaded")
    if len(items) != 2 * (len(train_ids) + len(validation_ids)):
        raise ValueError("pilot payload inventory is incomplete")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    auth_headers = _hub_auth_headers()

    def fetch(item: dict[str, Any]) -> dict[str, Any]:
        suffix = ".mp4" if item["kind"] == "video" else ".parquet"
        target = output_dir / f"{item['kind']}-episode-{item['episode']:06d}{suffix}"
        if target.exists():
            payload = target.read_bytes()
        else:
            url = (
                f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/"
                f"{item['path']}?download=true"
            )
            for attempt in range(max_retries + 1):
                response = requests.get(url, headers=auth_headers, timeout=(30, 180))
                status_code = getattr(response, "status_code", 200)
                if status_code not in RETRYABLE_HTTP_STATUSES:
                    response.raise_for_status()
                    break
                if attempt == max_retries:
                    response.raise_for_status()
                delay = _retry_after_seconds(response, attempt)
                print(
                    json.dumps(
                        {
                            "event": "http_transient_retry",
                            "status_code": status_code,
                            "episode": item["episode"],
                            "kind": item["kind"],
                            "retry": attempt + 1,
                            "max_retries": max_retries,
                            "sleep_seconds": delay,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
            payload = response.content
        actual_sha = _sha256(payload)
        if len(payload) != item["size"] or actual_sha != item["sha256"]:
            raise ValueError(f"publisher size/hash mismatch for {item['path']}")
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(target)
        return {
            **item,
            "local_path": str(target),
            "verified_sha256": actual_sha,
            "bytes": len(payload),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        verified = list(pool.map(fetch, items))
    receipt = {
        "dataset_id": dataset_id,
        "revision": revision,
        "subset": subset,
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selection": {
            "train_episode_indices": train_ids,
            "validation_episode_indices": validation_ids,
            "train_episode_id_order_sha256": selection["train_episode_id_order_sha256"],
            "validation_episode_id_order_sha256": selection[
                "validation_episode_id_order_sha256"
            ],
        },
        "reserved_holdout_payload_downloaded": False,
        "files": verified,
        "file_count": len(verified),
        "total_bytes": sum(item["size"] for item in verified),
        "wall_seconds": time.perf_counter() - started,
    }
    receipt_path = output_dir / "download-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return {
        "receipt_path": str(receipt_path),
        "files": len(verified),
        "bytes": receipt["total_bytes"],
        "wall_seconds": receipt["wall_seconds"],
        "train_episode_count": len(train_ids),
        "validation_episode_count": len(validation_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            download_pilot(
                args.manifest,
                args.output_dir,
                workers=args.workers,
                max_retries=args.max_retries,
            )
        )
    )


if __name__ == "__main__":
    main()
