"""Fetch only frozen STARSS23 parents needed by DCASE's room-held-out dev-test.

The script reads ZIP central-directory metadata first, then measures three
parent-pair transfers by default. Pass ``--all`` only after reviewing the
measured size/time estimate. It never opens DCASE annotation CSV contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.fetch_starss23_smoke import (  # noqa: E402
    ARCHIVES,
    BASE,
    HttpRangeReader,
    archive_index,
)
from scripts.preflight_dcase2025 import (  # noqa: E402
    DEVTEST_ROOMS,
    METADATA_SHA256,
    sha256_file,
)
from tiny_omni_decision.dcase2025 import parse_dcase2025_clip_member  # noqa: E402

SOURCE_RECORD = "doi:10.5281/zenodo.7880637"
MEMBER_LIMIT = 512 * 1024 * 1024
TOTAL_UNCOMPRESSED_LIMIT = 12 * 1024 * 1024 * 1024
FREE_SPACE_MARGIN = 2 * 1024 * 1024 * 1024
RANGE_BUDGET = 8 * 1024 * 1024 * 1024
CHUNK = 1024 * 1024


def candidate_identity(metadata: Path, freeze: Path) -> tuple[dict, list[dict]]:
    if sha256_file(metadata) != METADATA_SHA256:
        raise ValueError("DCASE metadata archive hash differs from the pin")
    receipt = json.loads(freeze.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "FROZEN_BEFORE_DEVTEST"
        or receipt.get("devtest_annotation_content_opened") is not False
    ):
        raise ValueError("all fixed readout candidates must be frozen before dev-test access")
    model_dir = Path(receipt["source_training_output"])
    for mode, expected in receipt["candidate_checkpoint_sha256"].items():
        if sha256_file(model_dir / f"{mode}-readout.npz") != expected:
            raise ValueError(f"frozen {mode} candidate checkpoint changed")
    parents: dict[str, str] = {}
    with zipfile.ZipFile(metadata) as archive:
        for member in archive.namelist():
            if not member.startswith("metadata_dev/dev-test-"):
                continue
            clip = parse_dcase2025_clip_member(member)
            if clip is not None:
                old_room = parents.setdefault(clip.parent_recording, clip.room_id)
                if old_room != clip.room_id:
                    raise ValueError("one dev-test parent maps to multiple rooms")
    if {room for room in parents.values()} != DEVTEST_ROOMS:
        raise ValueError("dev-test room inventory differs from the pinned split")
    return receipt, [{"clip": clip, "room": room} for clip, room in sorted(parents.items())]


def inventories(parents: list[dict]) -> tuple[dict, dict, dict]:
    indexes: dict[str, dict] = {}
    archive_transfer = {}
    for name, (size, upstream_md5) in ARCHIVES.items():
        reader = HttpRangeReader(f"{BASE}/{name}/content", size, budget=256 * 1024 * 1024)
        with zipfile.ZipFile(reader) as archive:
            suffix = ".wav" if name.startswith("foa") else ".mp4"
            indexes["audio" if suffix == ".wav" else "video"] = archive_index(archive, suffix)
        archive_transfer[name] = {
            "archive_size": size,
            "upstream_md5_unverified": upstream_md5,
            "central_directory_bytes": reader.transferred,
        }
    parent_rows = []
    total_compressed = 0
    total_uncompressed = 0
    for row in parents:
        item = {**row, "files": {}}
        for modality in ("audio", "video"):
            entry = indexes[modality].get(row["clip"])
            if entry is None:
                raise ValueError(f"official {modality} archive lacks {row['clip']}")
            if entry.file_size > MEMBER_LIMIT or entry.compress_size > MEMBER_LIMIT:
                raise ValueError(f"media member exceeds the per-file cap: {row['clip']}")
            if entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError(f"unsupported ZIP compression method: {entry.compress_type}")
            item["files"][modality] = {
                "member": entry.filename,
                "bytes": entry.file_size,
                "compressed_bytes": entry.compress_size,
                "crc32": entry.CRC,
                "compress_type": entry.compress_type,
                "header_offset": entry.header_offset,
            }
            total_compressed += entry.compress_size
            total_uncompressed += entry.file_size
        parent_rows.append(item)
    if total_uncompressed > TOTAL_UNCOMPRESSED_LIMIT:
        raise ValueError(
            f"selected media exceeds the uncompressed output cap: {total_uncompressed}"
        )
    return (
        {
            "parents": parent_rows,
            "total_compressed_bytes": total_compressed,
            "total_uncompressed_bytes": total_uncompressed,
        },
        indexes,
        archive_transfer,
    )


def request_range(
    url: str, start: int, length: int, *, total_size: int, timeout: int = 120
) -> bytes:
    end = start + length - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        expected = f"bytes {start}-{end}/"
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or content_range != f"{expected}{total_size}":
            raise ValueError("source server did not honor the exact byte-range request")
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError("source range length mismatch")
    return data


def stream_range_to_file(
    url: str,
    start: int,
    length: int,
    destination: Path,
    *,
    total_size: int,
    archive_name: str,
    counters: dict[str, int],
) -> None:
    end = start + length - 1
    for attempt in range(3):
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        retryable_error = None
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                expected = f"bytes {start}-{end}/{total_size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("server did not return the exact member byte range")
                remaining = length
                with destination.open("wb") as output:
                    while remaining:
                        chunk = response.read(min(CHUNK, remaining))
                        if not chunk:
                            raise ValueError("truncated compressed ZIP member")
                        output.write(chunk)
                        counters[archive_name] += len(chunk)
                        remaining -= len(chunk)
                    if response.read(1):
                        raise ValueError("server returned bytes beyond requested member")
            return
        except OSError as exc:
            retryable_error = exc
        except ValueError as exc:
            if str(exc) != "truncated compressed ZIP member":
                raise
            retryable_error = exc
        if attempt == 2:
            raise retryable_error
        destination.unlink(missing_ok=True)


def extract_member(
    archive_name: str,
    entry: zipfile.ZipInfo,
    destination: Path,
    *,
    counters: dict[str, int],
) -> dict[str, object]:
    archive_size, _md5 = ARCHIVES[archive_name]
    url = f"{BASE}/{archive_name}/content"
    local = request_range(url, entry.header_offset, 30, total_size=archive_size)
    counters[archive_name] += len(local)
    values = struct.unpack("<IHHHHHIIIHH", local)
    if values[0] != 0x04034B50:
        raise ValueError(f"bad local ZIP header: {entry.filename}")
    name_size, extra_size = values[-2:]
    header_tail = request_range(
        url,
        entry.header_offset + 30,
        name_size + extra_size,
        total_size=archive_size,
    )
    counters[archive_name] += len(header_tail)
    encoded_name = header_tail[:name_size]
    local_name = encoded_name.decode("utf-8" if values[1] & 0x800 else "cp437")
    if local_name != entry.filename:
        raise ValueError("ZIP central and local member names differ")
    data_offset = entry.header_offset + 30 + name_size + extra_size
    if data_offset + entry.compress_size > archive_size:
        raise ValueError("member byte range escapes the pinned archive")
    if counters[archive_name] + entry.compress_size > RANGE_BUDGET:
        raise ValueError("bounded member-transfer budget exceeded")
    if entry.flag_bits & 0x1:
        raise ValueError("encrypted ZIP members are not supported")
    temp_compressed = destination.with_suffix(destination.suffix + ".compressed.part")
    crc = 0
    uncompressed = 0
    digest = hashlib.sha256()
    decompressor = zlib.decompressobj(-15) if entry.compress_type == zipfile.ZIP_DEFLATED else None
    try:
        stream_range_to_file(
            url,
            data_offset,
            entry.compress_size,
            temp_compressed,
            total_size=archive_size,
            archive_name=archive_name,
            counters=counters,
        )
        with temp_compressed.open("rb") as compressed_file, destination.open("wb") as output:
            while chunk := compressed_file.read(CHUNK):
                decoded = decompressor.decompress(chunk) if decompressor else chunk
                output.write(decoded)
                crc = zlib.crc32(decoded, crc)
                uncompressed += len(decoded)
                digest.update(decoded)
            if decompressor:
                tail = decompressor.flush()
                output.write(tail)
                crc = zlib.crc32(tail, crc)
                uncompressed += len(tail)
                digest.update(tail)
                if not decompressor.eof or decompressor.unused_data:
                    raise ValueError("deflate stream is incomplete or contains trailing data")
        if uncompressed != entry.file_size or crc & 0xFFFFFFFF != entry.CRC:
            raise ValueError(f"ZIP member size/CRC mismatch: {entry.filename}")
        return {
            "path": str(destination.resolve()),
            "member": entry.filename,
            "bytes": uncompressed,
            "compressed_bytes": entry.compress_size,
            "crc32": entry.CRC,
            "sha256": digest.hexdigest(),
        }
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        temp_compressed.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--candidate-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-count", type=int)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.sample_count is not None and args.sample_count != 3:
        raise ValueError("the preregistered network timing sample is exactly three parent pairs")
    freeze, parents = candidate_identity(args.metadata_zip, args.candidate_freeze)
    inventory, indexes, central_directory = inventories(parents)
    disk_free = shutil.disk_usage(args.output_dir.anchor or ".").free
    needed = inventory["total_uncompressed_bytes"] + FREE_SPACE_MARGIN
    if disk_free < needed:
        raise OSError(f"insufficient disk space: free={disk_free}, required={needed}")
    output = args.output_dir.resolve()
    progress_path = output / "progress.json"
    if output.exists() and not progress_path.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is non-empty and has no acquisition receipt")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "record": SOURCE_RECORD,
        "license": "MIT",
        "metadata_sha256": sha256_file(args.metadata_zip),
        "candidate_freeze_sha256": sha256_file(args.candidate_freeze),
        "selected_devtest_parent_inventory_sha256": hashlib.sha256(
            json.dumps(inventory["parents"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "total_parent_recordings": len(parents),
        "total_compressed_bytes": inventory["total_compressed_bytes"],
        "total_uncompressed_bytes": inventory["total_uncompressed_bytes"],
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("existing acquisition progress has different pinned inputs")
        for parent, files in progress.get("completed", {}).items():
            for modality, record in files.items():
                path = Path(record["path"])
                if not path.is_file() or sha256_file(path) != record["sha256"]:
                    raise ValueError(f"saved acquisition output changed: {parent}/{modality}")
    else:
        progress = {"status": "RUNNING", "identity": identity, "completed": {}, "timings": []}
        atomic_json(progress_path, progress)

    if args.sample_count is not None:
        sample_set = sorted(
            parents,
            key=lambda row: hashlib.sha256(f"17\0{row['clip']}".encode()).digest(),
        )[: args.sample_count]
    else:
        sample_set = parents
    entries_by_mode = {}
    for modality in ("audio", "video"):
        entries_by_mode[modality] = indexes[modality]
    for row in sample_set:
        parent = row["clip"]
        if parent in progress["completed"]:
            continue
        pair_started = time.perf_counter()
        pair_files = {}
        for modality, archive_name, suffix in (
            ("audio", "foa_dev.zip", ".wav"),
            ("video", "video_dev.zip", ".mp4"),
        ):
            entry = entries_by_mode[modality][parent]
            path = output / f"{parent}{suffix}"
            pair_files[modality] = extract_member(
                archive_name,
                entry,
                path,
                counters=progress.setdefault("range_bytes", {"foa_dev.zip": 0, "video_dev.zip": 0}),
            )
        pair_seconds = time.perf_counter() - pair_started
        progress["completed"][parent] = pair_files
        progress["timings"].append(
            {
                "parent": parent,
                "seconds": pair_seconds,
                "compressed_bytes": sum(file["compressed_bytes"] for file in pair_files.values()),
            }
        )
        progress["status"] = "SAMPLED" if args.sample_count is not None else "RUNNING"
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "event": "parent_pair_complete",
                    "completed": len(progress["completed"]),
                    "total": len(parents),
                    "sample_seconds": round(pair_seconds, 3),
                    "parent": parent,
                }
            ),
            flush=True,
        )

    if args.sample_count is not None:
        samples = progress["timings"]
        bytes_total = sum(item["compressed_bytes"] for item in samples)
        seconds_total = sum(item["seconds"] for item in samples)
        missing_bytes = inventory["total_compressed_bytes"] - sum(
            item["compressed_bytes"] for item in samples
        )
        rate = bytes_total / seconds_total if seconds_total else 0.0
        estimate = missing_bytes / rate if rate else None
        print(
            json.dumps(
                {
                    "status": "SAMPLED",
                    "sample_count": len(samples),
                    "mean_pair_seconds": sum(item["seconds"] for item in samples) / len(samples),
                    "pooled_compressed_bytes_per_second": rate,
                    "estimated_remaining_seconds": estimate,
                    "total_compressed_bytes": inventory["total_compressed_bytes"],
                    "total_uncompressed_bytes": inventory["total_uncompressed_bytes"],
                    "disk_free_bytes": disk_free,
                    "central_directory_bytes": central_directory,
                }
            )
        )
        return

    missing = set(row["clip"] for row in parents) - set(progress["completed"])
    if missing:
        raise ValueError("full dev-test media acquisition left unprocessed parents")
    manifest = {
        "status": "PASS",
        "record": SOURCE_RECORD,
        "license": "MIT",
        "license_basis": "official STARSS23 v1.1.0 dataset release",
        "candidate_freeze_sha256": identity["candidate_freeze_sha256"],
        "metadata_sha256": identity["metadata_sha256"],
        "archives": {
            **central_directory,
            "range_bytes_downloaded": progress["range_bytes"],
            "whole_archive_md5_verified": False,
        },
        "parents": len(parents),
        "pairs": [
            {"clip": row["clip"], "room": row["room"], "files": progress["completed"][row["clip"]]}
            for row in parents
        ],
    }
    manifest_path = output / "media-manifest.json"
    atomic_json(manifest_path, manifest)
    progress["status"] = "PASS"
    progress["media_manifest_sha256"] = sha256_file(manifest_path)
    progress["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(progress_path, progress)
    print(json.dumps({"status": "PASS", "media_manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
