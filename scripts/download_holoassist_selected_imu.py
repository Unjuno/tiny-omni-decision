"""Download allow-listed HoloAssist train/validation IMU members by checked HTTP ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests

from scripts.audit_holoassist_metadata import sha256_file
from scripts.download_holoassist_selected_media import download_member

IMU_SUFFIXES = (
    "Accelerometer_sync.txt",
    "Gyroscope_sync.txt",
    "Magnetometer_sync.txt",
)
IMU_ARCHIVE_URL = "https://hl2data.z5.web.core.windows.net/holoassist-data-release/imu.tar"
IMU_ARCHIVE_ETAG = '"0x8DC8C9DD94B2BA2"'
IMU_ARCHIVE_SIZE = 4_971_059_200
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9 _-]+$")


def select_imu_members(media_manifest: dict, archive_index: dict) -> list[dict]:
    """Join the frozen media session split to exactly three synchronized IMU files."""
    if archive_index.get("payload_media_downloaded") is not False:
        raise ValueError("archive index must attest that payload media was not downloaded")
    sessions: dict[str, str] = {}
    for row in media_manifest.get("files", []):
        session_id = str(row.get("session_id", ""))
        split = str(row.get("split", ""))
        if not SESSION_PATTERN.fullmatch(session_id) or split not in {"train", "validation"}:
            raise ValueError(f"invalid selected session or split: {session_id!r}, {split!r}")
        if session_id in sessions:
            raise ValueError(f"duplicate selected media session: {session_id}")
        sessions[session_id] = split
    if not sessions:
        raise ValueError("media manifest contains no selected sessions")
    if archive_index.get("allowed_session_count") != len(sessions):
        raise ValueError("archive index selected-session count differs from media manifest")

    members = archive_index.get("index", {}).get("members", [])
    found: dict[tuple[str, str], dict] = {}
    selected_ids = set(sessions)
    for member in members:
        if not member.get("is_file"):
            continue
        name = str(member.get("name", ""))
        session_id = str(member.get("session_id", ""))
        if "/Export_py/IMU/" not in name:
            continue
        if session_id not in selected_ids:
            raise ValueError(
                f"archive index contains a session outside the selection: {session_id}"
            )
        suffix = name.rsplit("/", 1)[-1]
        if suffix not in IMU_SUFFIXES or name != f"{session_id}/Export_py/IMU/{suffix}":
            raise ValueError(f"unexpected synchronized IMU member path: {name}")
        key = (session_id, suffix)
        if key in found:
            raise ValueError(f"duplicate archive member: {name}")
        size = int(member["size"])
        offset = int(member["offset_data"])
        if size <= 0 or offset < 0:
            raise ValueError(f"invalid archive member range: {name}")
        found[key] = {
            "split": sessions[session_id],
            "session_id": session_id,
            "sensor": suffix.removesuffix("_sync.txt"),
            "archive_member": name,
            "archive_offset": offset,
            "size_bytes": size,
        }

    expected = {(session_id, suffix) for session_id in sessions for suffix in IMU_SUFFIXES}
    if set(found) != expected:
        raise ValueError("each selected session must have exactly the three synchronized IMU files")
    return sorted(found.values(), key=lambda row: (row["split"], row["session_id"], row["sensor"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-bytes", type=int, default=4_000_000_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite IMU output directory: {args.output_dir}")

    media_bytes = args.media_manifest.read_bytes()
    index_bytes = args.index.read_bytes()
    audit_bytes = args.audit.read_bytes()
    media_manifest = json.loads(media_bytes)
    archive_index = json.loads(index_bytes)
    audit = json.loads(audit_bytes)
    selected = select_imu_members(media_manifest, archive_index)
    total_bytes = sum(row["size_bytes"] for row in selected)
    if total_bytes > args.max_bytes:
        raise ValueError(f"selection requires {total_bytes} bytes, above limit {args.max_bytes}")
    archive = archive_index.get("archive", {})
    url = str(archive.get("url", archive_index.get("archive_url", "")))
    etag = str(archive.get("etag", archive_index.get("archive_etag", "")))
    archive_size = int(archive.get("size_bytes", archive_index.get("archive_size_bytes", 0)))
    if (url, etag, archive_size) != (
        IMU_ARCHIVE_URL,
        IMU_ARCHIVE_ETAG,
        IMU_ARCHIVE_SIZE,
    ):
        raise ValueError("archive index differs from the pinned official HoloAssist IMU source")
    selection_hash = hashlib.sha256(media_bytes).hexdigest()
    index_hash = hashlib.sha256(index_bytes).hexdigest()
    manifest_selection_hashes = {
        str(row.get("selection_sha256", "")) for row in media_manifest["files"]
    }
    if (
        len(manifest_selection_hashes) != 1
        or not next(iter(manifest_selection_hashes))
        or audit.get("index_sha256") != index_hash
        or audit.get("media_manifest_sha256") != selection_hash
        or audit.get("selection_sha256") != next(iter(manifest_selection_hashes))
        or audit.get("allowed_session_count") != len(media_manifest["files"])
        or audit.get("payload_downloaded") is not False
        or audit.get("imu_archive_url") != archive_index.get("archive_url")
        or audit.get("imu_archive_etag") != archive_index.get("archive_etag")
        or audit.get("imu_archive_size_bytes") != archive_index.get("archive_size_bytes")
    ):
        raise ValueError("media manifest, range index, and audit are not hash-bound")
    progress = args.output_dir / "download-progress.jsonl"
    manifest_path = args.output_dir / "imu-manifest.json"
    if args.output_dir.exists():
        if not args.resume or not progress.is_file() or manifest_path.exists():
            raise FileExistsError(
                "resume requires an incomplete output dir with a progress journal"
            )
    else:
        args.output_dir.mkdir(parents=True)

    completed: dict[tuple[str, str, str], dict] = {}
    if args.resume:
        for line in progress.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if (
                item.get("selection_sha256") != selection_hash
                or item.get("index_sha256") != index_hash
            ):
                raise ValueError("progress journal belongs to different pinned inputs")
            key = (str(item["split"]), str(item["session_id"]), str(item["sensor"]))
            if key in completed:
                raise ValueError(f"duplicate progress-journal entry: {key}")
            completed[key] = item

    pending: list[tuple[dict, Path]] = []
    files: list[dict] = []
    for row in selected:
        key = (row["split"], row["session_id"], row["sensor"])
        destination = (
            args.output_dir
            / row["split"]
            / row["session_id"]
            / f"{row['sensor']}_sync.txt"
        )
        previous = completed.get(key)
        if previous:
            if (
                not destination.is_file()
                or destination.stat().st_size != row["size_bytes"]
                or sha256_file(destination) != previous.get("sha256")
                or previous.get("archive_etag") != etag
                or previous.get("archive_offset") != row["archive_offset"]
            ):
                raise ValueError(f"completed IMU file failed resume verification: {destination}")
            files.append(previous)
            continue
        if destination.exists():
            raise FileExistsError(f"untracked existing IMU file must be inspected: {destination}")
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.exists() and not args.resume:
            raise FileExistsError(f"partial IMU file requires --resume: {partial}")
        pending.append((row, destination))

    pending_bytes = sum(row["size_bytes"] for row, _ in pending)
    if shutil.disk_usage(args.output_dir.parent).free < int(pending_bytes * 1.2):
        raise OSError(f"need at least {int(pending_bytes * 1.2)} free bytes beside output dir")

    def fetch(row: dict, destination: Path) -> dict:
        with requests.Session() as session:
            result = download_member(
                session,
                url=url,
                etag=etag,
                archive_size=archive_size,
                offset=row["archive_offset"],
                size=row["size_bytes"],
                destination=destination,
                resume_partial=args.resume,
            )
        return {**row, **result, "archive_etag": etag}

    with progress.open("a", encoding="utf-8", newline="\n") as journal:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch, row, path): row for row, path in pending}
            for number, future in enumerate(as_completed(futures), start=1):
                item = future.result()
                item["selection_sha256"] = selection_hash
                item["index_sha256"] = index_hash
                files.append(item)
                journal.write(json.dumps(item, sort_keys=True) + "\n")
                journal.flush()
                if number % 30 == 0 or number == len(pending):
                    print(f"downloaded={len(files)}/{len(selected)}", flush=True)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_sha256": selection_hash,
        "index_sha256": index_hash,
        "archive_url": url,
        "archive_etag": etag,
        "archive_size_bytes": archive_size,
        "selected_file_count": len(files),
        "selected_bytes": total_bytes,
        "files": sorted(files, key=lambda row: (row["split"], row["session_id"], row["sensor"])),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


if __name__ == "__main__":
    main()
