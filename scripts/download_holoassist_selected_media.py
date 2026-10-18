"""Download selected HoloAssist train/validation MP4 members via checked HTTP ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests

from scripts.audit_holoassist_metadata import sha256_file

SESSION_PATTERN = re.compile(r"^[A-Za-z0-9 _-]+$")
ALLOWED_MEDIA_SUFFIXES = (
    "/Export_py/Video_compress.mp4",
    "/Export_py/Video_pitchshift.mp4",
)


def validate_member_suffix(member: str, expected_suffix: str) -> None:
    if expected_suffix not in ALLOWED_MEDIA_SUFFIXES:
        raise ValueError(f"unsupported selected-media suffix: {expected_suffix}")
    if not member.endswith(expected_suffix):
        raise ValueError(f"unexpected video member path: {member}")


def download_member(
    session: requests.Session,
    *,
    url: str,
    etag: str,
    archive_size: int,
    offset: int,
    size: int,
    destination: Path,
    retries: int = 4,
    resume_partial: bool = False,
) -> dict[str, int | str]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite selected media: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and not resume_partial:
        raise FileExistsError(f"stale partial media exists; inspect before retry: {partial}")
    if partial.exists() and partial.stat().st_size > size:
        raise ValueError(f"partial media exceeds expected member size: {partial}")
    last_error: Exception | None = None
    for attempt in range(retries):
        prefix_bytes = partial.stat().st_size if partial.exists() else 0
        if prefix_bytes == size:
            partial.replace(destination)
            return {
                "path": str(destination),
                "size_bytes": size,
                "sha256": sha256_file(destination),
                "archive_etag": etag,
                "archive_offset": offset,
            }
        request_start = offset + prefix_bytes
        end = offset + size - 1
        expected_range = f"bytes {request_start}-{end}/{archive_size}"
        try:
            with session.get(
                url,
                headers={
                    "Range": f"bytes={request_start}-{end}",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                timeout=(15, 120),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise ValueError(
                        f"server ignored media Range request: HTTP {response.status_code}"
                    )
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError(
                        f"unexpected media Content-Range: {response.headers.get('Content-Range')}"
                    )
                if response.headers.get("ETag") != etag:
                    raise ValueError(f"archive ETag changed: {response.headers.get('ETag')}")
                mode = "ab" if partial.exists() else "xb"
                with partial.open(mode) as target:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        target.write(chunk)
            if partial.stat().st_size != size:
                raise ValueError(
                    f"expected {size} media bytes, received {partial.stat().st_size}"
                )
            partial.replace(destination)
            return {
                "path": str(destination),
                "size_bytes": size,
                "sha256": sha256_file(destination),
                "archive_etag": etag,
                "archive_offset": offset,
            }
        except (requests.RequestException, OSError, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    message = f"failed to fetch selected archive member at {offset}+{size}"
    raise RuntimeError(message) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--archive-etag", required=True)
    parser.add_argument("--archive-size", required=True, type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=32_000_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--expected-member-suffix",
        default="/Export_py/Video_compress.mp4",
        choices=ALLOWED_MEDIA_SUFFIXES,
        help="Exact allow-listed archive member type expected by this selection.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite media directory: {args.output_dir}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected = selection["selection"]
    entries = [
        (split, row)
        for split in ("train", "validation")
        for row in selected[split]["videos"]
    ]
    total_bytes = sum(int(row["size_bytes"]) for _, row in entries)
    if total_bytes != int(selected["all_selected_video_bytes"]):
        raise ValueError("selection media byte total is inconsistent")
    if total_bytes > args.max_bytes:
        raise ValueError(f"selection requires {total_bytes} bytes, above limit {args.max_bytes}")
    progress_path = args.output_dir / "download-progress.jsonl"
    output_manifest = args.output_dir / "media-manifest.json"
    if args.output_dir.exists():
        if not args.resume or not progress_path.is_file():
            raise FileExistsError("resume requires an existing progress journal in the output dir")
        if output_manifest.exists():
            raise FileExistsError(f"completed media manifest already exists: {output_manifest}")
    else:
        args.output_dir.mkdir(parents=True)

    selection_hash = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    completed: dict[tuple[str, str], dict] = {}
    if args.resume:
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("selection_sha256") != selection_hash:
                raise ValueError("progress journal belongs to a different session selection")
            key = (str(item["split"]), str(item["session_id"]))
            if key in completed:
                raise ValueError(f"duplicate progress-journal entry: {key}")
            completed[key] = item

    pending = []
    files = []
    for split, row in entries:
        session_id = str(row["session_id"])
        if session_id in {".", ".."} or not SESSION_PATTERN.fullmatch(session_id):
            raise ValueError(f"unsafe session identifier: {session_id}")
        member = str(row["video_member_name"])
        validate_member_suffix(member, args.expected_member_suffix)
        key = (split, session_id)
        destination = args.output_dir / split / f"{session_id}.mp4"
        previous = completed.get(key)
        if previous:
            if (
                not destination.is_file()
                or destination.stat().st_size != int(row["size_bytes"])
                or sha256_file(destination) != previous["sha256"]
                or previous.get("archive_etag") != args.archive_etag
                or previous.get("size_bytes") != int(row["size_bytes"])
            ):
                raise ValueError(f"completed media failed resume verification: {destination}")
            files.append(previous)
            continue
        partial = destination.with_suffix(".mp4.part")
        if destination.exists():
            raise FileExistsError(f"untracked existing media must be inspected: {destination}")
        if partial.exists() and not args.resume:
            raise FileExistsError(f"partial media requires --resume: {partial}")
        pending.append((split, row, destination))

    pending_bytes = sum(int(row["size_bytes"]) for _, row, _ in pending)
    required_free = int(pending_bytes * 1.2)
    if shutil.disk_usage(args.output_dir.parent).free < required_free:
        raise OSError(f"need at least {required_free} free bytes beside output directory")

    def fetch(split: str, row: dict, destination: Path) -> dict:
        with requests.Session() as session:
            item = download_member(
                session,
                url=args.url,
                etag=args.archive_etag,
                archive_size=args.archive_size,
                offset=int(row["offset_data"]),
                size=int(row["size_bytes"]),
                destination=destination,
                resume_partial=args.resume,
            )
        return {"split": split, "session_id": str(row["session_id"]), **item}

    with progress_path.open("a", encoding="utf-8", newline="\n") as journal:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_expected = {
                executor.submit(fetch, split, row, destination): (split, row)
                for split, row, destination in pending
            }
            for index, future in enumerate(as_completed(future_to_expected), start=1):
                item = future.result()
                item["selection_sha256"] = selection_hash
                files.append(item)
                journal.write(json.dumps(item, sort_keys=True) + "\n")
                journal.flush()
                key = (item["split"], item["session_id"])
                completed[key] = item
                if index % 10 == 0 or index == len(pending):
                    received_bytes = sum(int(file["size_bytes"]) for file in files)
                    print(
                        f"downloaded={len(completed)}/{len(entries)} bytes={received_bytes}",
                        flush=True,
                    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_sha256": selection_hash,
        "archive_etag": args.archive_etag,
        "selected_file_count": len(files),
        "selected_bytes": total_bytes,
        "files": sorted(files, key=lambda file: (file["split"], file["session_id"])),
    }
    temporary = output_manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_manifest)


if __name__ == "__main__":
    main()
