"""Download selected, unseen EgoViz train/development MCAPs with hash receipts.

The script rejects the sealed-audit split, checks disk reserve before writing,
and never overwrites an existing output directory or file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

ALLOWED_SPLITS = ("train", "development")
DEFAULT_RESERVE_BYTES = 1_000_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_media_rows(
    plan: dict[str, Any],
    *,
    splits: tuple[str, ...] = ALLOWED_SPLITS,
    limit_per_split: int | None = None,
    offset_per_split: int = 0,
) -> list[dict[str, Any]]:
    if plan.get("format") != "egoviz-unseen-session-temporal-correspondence-plan-v1":
        raise ValueError("unsupported EgoViz group plan format")
    if "sealed_audit" in splits:
        raise ValueError("sealed_audit cannot be downloaded before configuration freeze")
    if not splits or any(split not in ALLOWED_SPLITS for split in splits):
        raise ValueError("unsupported split; only train and development are allowed")
    if len(set(splits)) != len(splits):
        raise ValueError("split list contains duplicates")
    if limit_per_split is not None and limit_per_split <= 0:
        raise ValueError("limit_per_split must be positive")
    if offset_per_split < 0:
        raise ValueError("offset_per_split must be nonnegative")

    result: list[dict[str, Any]] = []
    groups: set[str] = set()
    for split in splits:
        rows = plan["splits"][split]
        remaining = rows[offset_per_split:]
        chosen = remaining if limit_per_split is None else remaining[:limit_per_split]
        for original in chosen:
            row = dict(original)
            row["split"] = split
            group_id = str(row["group_id"])
            if group_id in groups:
                raise ValueError("group overlap among selected media rows")
            groups.add(group_id)
            if int(row["expected_bytes"]) <= 0:
                raise ValueError("selected row has invalid expected_bytes")
            if not str(row.get("mcap_path", "")).startswith("data/"):
                raise ValueError("selected row has invalid bucket path")
            result.append(row)
    if not result:
        raise ValueError("selection contains no media rows")
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def download(
    plan_path: Path,
    output: Path,
    *,
    splits: tuple[str, ...] = ALLOWED_SPLITS,
    limit_per_split: int | None = None,
    offset_per_split: int = 0,
    max_total_bytes: int = 5_000_000_000,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
) -> dict[str, Any]:
    import requests

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if max_total_bytes <= 0 or reserve_bytes < 0:
        raise ValueError("byte limits must be positive and reserve nonnegative")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = select_media_rows(
        plan,
        splits=splits,
        limit_per_split=limit_per_split,
        offset_per_split=offset_per_split,
    )
    total_bytes = sum(int(row["expected_bytes"]) for row in rows)
    if total_bytes > max_total_bytes:
        raise ValueError(f"selection requires {total_bytes} bytes, over max_total_bytes")
    free_bytes = shutil.disk_usage(output.parent).free
    if total_bytes + reserve_bytes > free_bytes:
        raise OSError(
            f"insufficient disk reserve: need {total_bytes + reserve_bytes}, "
            f"have {free_bytes} bytes"
        )

    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "acquisition-manifest.json"
    manifest: dict[str, Any] = {
        "format": "egoviz-unseen-group-media-acquisition-v1",
        "status": "ACQUIRING_TRAIN_DEVELOPMENT_ONLY",
        "plan_sha256": sha256_file(plan_path),
        "splits": list(splits),
        "limit_per_split": limit_per_split,
        "offset_per_split": offset_per_split,
        "expected_files": len(rows),
        "expected_bytes": total_bytes,
        "sealed_audit_downloaded": False,
        "files": [],
    }
    _write_json(manifest_path, manifest)
    timing: list[float] = []

    with requests.Session() as session:
        for position, row in enumerate(rows, start=1):
            chunk_id = str(row["chunk_id"])
            expected_bytes = int(row["expected_bytes"])
            relative = str(row["mcap_path"])
            url = "https://huggingface.co/buckets/humaidtech/EgoViz-120/resolve/" + quote(
                relative, safe="/"
            )
            target = output / f"{chunk_id}.mcap"
            partial = output / f"{chunk_id}.mcap.part"
            if target.exists() or partial.exists():
                raise FileExistsError(f"refusing to overwrite file for {chunk_id}")

            head = session.head(url, allow_redirects=False, timeout=(20, 60))
            if head.status_code not in (301, 302, 303, 307, 308):
                raise RuntimeError(f"unexpected HEAD status for {chunk_id}: {head.status_code}")
            linked_size = int(head.headers.get("X-Linked-Size", "-1"))
            etag = head.headers.get("X-Linked-ETag", "").strip('"')
            if linked_size != expected_bytes or not etag:
                raise ValueError(f"source metadata mismatch for {chunk_id}")

            started = time.perf_counter()
            digest = hashlib.sha256()
            bytes_written = 0
            with session.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                content_length = int(response.headers.get("Content-Length", "-1"))
                if content_length not in (-1, expected_bytes):
                    raise ValueError(f"content length mismatch for {chunk_id}")
                with partial.open("xb") as stream:
                    for block in response.iter_content(4 * 1024 * 1024):
                        if block:
                            stream.write(block)
                            digest.update(block)
                            bytes_written += len(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            elapsed = time.perf_counter() - started
            if bytes_written != expected_bytes:
                raise ValueError(
                    f"download size mismatch for {chunk_id}: "
                    f"{bytes_written} != {expected_bytes}; partial retained"
                )
            os.replace(partial, target)
            record = {
                **row,
                "path": str(target),
                "bytes": bytes_written,
                "sha256": digest.hexdigest(),
                "head_linked_etag": etag,
                "download_seconds": elapsed,
                "MiB_per_second": bytes_written / (1024 * 1024) / elapsed,
            }
            manifest["files"].append(record)
            manifest["status"] = "ACQUIRED_TRAIN_DEVELOPMENT_ONLY"
            manifest["downloaded_bytes"] = sum(
                int(item["bytes"]) for item in manifest["files"]
            )
            _write_json(manifest_path, manifest)
            timing.append(elapsed)
            _write_json(
                output / "download-timing.json",
                {
                    "samples": timing,
                    "mean_seconds": sum(timing) / len(timing),
                    "completed_files": position,
                    "expected_files": len(rows),
                },
            )
            print(
                json.dumps(
                    {
                        "completed": position,
                        "expected": len(rows),
                        "chunk_id": chunk_id,
                        "bytes": bytes_written,
                        "download_seconds": elapsed,
                        "mean_seconds_per_file": sum(timing) / len(timing),
                    }
                ),
                flush=True,
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=ALLOWED_SPLITS, default=list(ALLOWED_SPLITS))
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--offset-per-split", type=int, default=0)
    parser.add_argument("--max-total-bytes", type=int, default=5_000_000_000)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    args = parser.parse_args()
    print(
        json.dumps(
            download(
                args.plan,
                args.output,
                splits=tuple(args.splits),
                limit_per_split=args.limit_per_split,
                offset_per_split=args.offset_per_split,
                max_total_bytes=args.max_total_bytes,
                reserve_bytes=args.reserve_bytes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
