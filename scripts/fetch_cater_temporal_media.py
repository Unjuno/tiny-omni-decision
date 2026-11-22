"""Fetch only hash-frozen CATER video members through checked HTTP ranges."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import io
import json
import shutil
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import requests

ARCHIVE_URL = "https://cmu.app.box.com/public/static/97kztikj3zwcgv5zu2by6c1jhlvh1d7x.zip"
ARCHIVE_BYTES = 14_215_629_625
TAIL_START = 14_207_241_017
TAIL_END = 14_215_629_624
TAIL_SHA256 = "6e07fd7f329227e8f63c1638cecac301bc373bc1443693d48cbd47211721bc01"
SPLITS = ("train", "validation")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        raise ValueError(f"unsafe CATER ZIP member path: {name}")
    if path.parts[0] != "videos" or not path.name.startswith("CATER_new_"):
        raise ValueError(f"unexpected CATER video member: {name}")
    if not path.name.endswith(".avi"):
        raise ValueError(f"unexpected CATER video extension: {name}")
    return path.name


def _fetch_range(
    session: requests.Session,
    url: str,
    size: int,
    etag: str | None,
    start: int,
    end: int,
) -> bytes:
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        allow_redirects=True,
        timeout=180,
    )
    response.raise_for_status()
    expected = f"bytes {start}-{end}/{size}"
    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
        raise RuntimeError("CATER server did not honor the exact requested byte range")
    if etag and response.headers.get("ETag") != etag:
        raise RuntimeError("CATER archive ETag changed during range reads")
    data = response.content
    if len(data) != end - start + 1:
        raise RuntimeError("short CATER range response")
    return data


class HttpRangeFile(io.RawIOBase):
    """Seekable ZIP archive view backed by verified tail bytes and HTTP ranges."""

    def __init__(
        self,
        url: str,
        size: int,
        etag: str | None,
        tail: bytes,
        session: requests.Session | None = None,
    ):
        self.url = url
        self.size = size
        self.etag = etag
        self.tail = tail
        self.tail_start = size - len(tail)
        self.position = 0
        self.session = session or requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        target = (
            offset
            if whence == io.SEEK_SET
            else self.position + offset
            if whence == io.SEEK_CUR
            else self.size + offset
        )
        if not 0 <= target <= self.size:
            raise ValueError("seek outside CATER archive")
        self.position = target
        return target

    def read(self, size: int = -1) -> bytes:
        if size == 0 or self.position >= self.size:
            return b""
        count = self.size - self.position if size < 0 else min(size, self.size - self.position)
        start, end = self.position, self.position + count - 1
        if start >= self.tail_start:
            data = self.tail[start - self.tail_start : end - self.tail_start + 1]
        elif end < self.tail_start:
            data = _fetch_range(self.session, self.url, self.size, self.etag, start, end)
        else:
            prefix = _fetch_range(
                self.session, self.url, self.size, self.etag, start, self.tail_start - 1
            )
            data = prefix + self.tail[: end - self.tail_start + 1]
        if len(data) != count:
            raise RuntimeError("CATER virtual archive returned a short read")
        self.position += count
        return data

    def close(self) -> None:
        self.session.close()
        super().close()


def load_manifest_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(row.get("split") != split for row in rows):
        raise ValueError(f"empty or mismarked {split} task manifest")
    videos: dict[str, str] = {}
    for row in rows:
        name = str(row["video_id"])
        expected_group = f"cater-video-{name.removeprefix('CATER_new_').removesuffix('.avi')}"
        if row.get("group_id") != expected_group:
            raise ValueError(f"task row has mismatched video group: {row['id']}")
        previous = videos.setdefault(name, expected_group)
        if previous != row["group_id"]:
            raise ValueError(f"video maps to multiple groups: {name}")
    return rows


def validate_existing_media(
    output_dir: Path,
    selected: list[tuple[str, str]],
    planned_sizes: dict[tuple[str, str], tuple[int, int, int]],
) -> dict[tuple[str, str], dict[str, Any]]:
    selected_set = set(selected)
    existing_files = {}
    for path in output_dir.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(output_dir)
        if len(relative.parts) != 2 or (relative.parts[0], relative.name) not in selected_set:
            raise ValueError(f"unexpected file in incomplete CATER output: {relative}")
        key = (relative.parts[0], relative.name)
        expected_size, compressed_size, expected_crc = planned_sizes[key]
        crc = 0
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                crc = binascii.crc32(block, crc)
                digest.update(block)
        if path.stat().st_size != expected_size or (crc & 0xFFFFFFFF) != expected_crc:
            raise ValueError(f"existing partial media failed size/CRC validation: {relative}")
        existing_files[key] = {
            "split": key[0],
            "video_id": key[1],
            "member": f"videos/{key[1]}",
            "relative_path": f"{key[0]}/{key[1]}",
            "source_compressed_bytes": compressed_size,
            "media_bytes": expected_size,
            "source_crc32": f"{expected_crc:08x}",
            "media_sha256": digest.hexdigest(),
        }
    return existing_files


def fetch_dataset(
    manifest_dir: Path,
    output_dir: Path,
    *,
    workers: int = 6,
    url: str = ARCHIVE_URL,
    resume_existing_incomplete: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and not resume_existing_incomplete:
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    if resume_existing_incomplete and not output_dir.is_dir():
        raise FileNotFoundError(f"resume directory does not exist: {output_dir}")
    if resume_existing_incomplete and (output_dir / "fetch-report.json").exists():
        raise FileExistsError("refusing to resume a directory that already has a fetch report")
    task_rows = {
        split: load_manifest_rows(manifest_dir / f"{split}.jsonl", split) for split in SPLITS
    }
    split_ids = {
        split: {str(row["video_id"]) for row in rows} for split, rows in task_rows.items()
    }
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("train/validation video identity overlap")

    session = requests.Session()
    head = session.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    size = int(head.headers.get("Content-Length", -1))
    etag = head.headers.get("ETag")
    last_modified = head.headers.get("Last-Modified")
    if size != ARCHIVE_BYTES:
        raise ValueError(f"unexpected CATER archive size: {size}")
    tail = _fetch_range(session, url, size, etag, TAIL_START, TAIL_END)
    if sha256_bytes(tail) != TAIL_SHA256:
        raise ValueError("CATER archive central-directory tail hash mismatch")

    selected = [
        (split, name)
        for split in SPLITS
        for name in sorted(split_ids[split])
    ]
    index_reader = HttpRangeFile(url, size, etag, tail)
    with zipfile.ZipFile(index_reader) as index_archive:
        planned_sizes = {}
        for split, name in selected:
            info = index_archive.getinfo(f"videos/{name}")
            if safe_member_name(info.filename) != name:
                raise ValueError(f"archive member identity mismatch: {info.filename}")
            if info.file_size <= 0 or info.flag_bits & 1:
                raise ValueError(f"invalid or encrypted media member: {info.filename}")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError(f"unsupported ZIP compression for {info.filename}")
            planned_sizes[(split, name)] = (info.file_size, info.compress_size, info.CRC)
    index_reader.close()
    existing_files = {}
    if resume_existing_incomplete:
        existing_files = validate_existing_media(output_dir, selected, planned_sizes)
    planned_uncompressed_bytes = sum(row[0] for row in planned_sizes.values())
    planned_compressed_bytes = sum(row[1] for row in planned_sizes.values())
    available_bytes = shutil.disk_usage(output_dir.parent).free
    required_free_bytes = int(planned_uncompressed_bytes * 1.10) + 1024**3
    if available_bytes < required_free_bytes:
        raise OSError(
            f"insufficient free space: need {required_free_bytes} bytes, have {available_bytes}"
        )

    output_dir.mkdir(parents=True, exist_ok=resume_existing_incomplete)
    thread_local = threading.local()
    readers: list[HttpRangeFile] = []
    readers_lock = threading.Lock()

    def read_member(split: str, name: str) -> dict[str, Any]:
        if not hasattr(thread_local, "archive"):
            reader = HttpRangeFile(url, size, etag, tail)
            with readers_lock:
                readers.append(reader)
            thread_local.reader = reader
            thread_local.archive = zipfile.ZipFile(reader)
        member = f"videos/{name}"
        archive: zipfile.ZipFile = thread_local.archive
        info = archive.getinfo(member)
        expected_size, expected_compressed, expected_crc = planned_sizes[(split, name)]
        if (info.file_size, info.compress_size, info.CRC) != (
            expected_size,
            expected_compressed,
            expected_crc,
        ):
            raise ValueError(f"CATER central directory changed for {member}")
        if info.file_size <= 0 or info.flag_bits & 1:
            raise ValueError(f"invalid or encrypted media member: {member}")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ValueError(f"unsupported ZIP compression for {member}")
        if safe_member_name(info.filename) != name:
            raise ValueError(f"archive member identity mismatch: {member}")
        payload = archive.read(info)
        if len(payload) != info.file_size:
            raise ValueError(f"uncompressed size mismatch for {member}")
        target = output_dir / split / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
        return {
            "split": split,
            "video_id": name,
            "member": info.filename,
            "relative_path": f"{split}/{name}",
            "source_compressed_bytes": info.compress_size,
            "media_bytes": len(payload),
            "source_crc32": f"{info.CRC:08x}",
            "media_sha256": sha256_bytes(payload),
        }

    tasks = [key for key in selected if key not in existing_files]
    records = list(existing_files.values())
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(read_member, split, name) for split, name in tasks]
            for future in as_completed(futures):
                records.append(future.result())
    finally:
        for reader in readers:
            reader.close()
        session.close()

    records.sort(key=lambda row: (row["split"], row["video_id"]))
    if len(records) != len(selected):
        raise ValueError("not all requested CATER media members were fetched")
    if len({(row["split"], row["video_id"]) for row in records}) != len(records):
        raise ValueError("duplicate downloaded CATER media identity")
    (output_dir / "media-manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "status": "complete_selected_cater_media_fetch",
        "archive_url": url,
        "archive_bytes": size,
        "archive_etag": etag,
        "archive_last_modified": last_modified,
        "archive_central_directory_tail_range": [TAIL_START, TAIL_END],
        "archive_central_directory_tail_sha256": TAIL_SHA256,
        "http_range_requests_used": True,
        "full_archive_downloaded": False,
        "workers": workers,
        "input_task_manifest_sha256": {
            split: sha256_file(manifest_dir / f"{split}.jsonl") for split in SPLITS
        },
        "train_video_groups": len(split_ids["train"]),
        "validation_video_groups": len(split_ids["validation"]),
        "train_validation_video_overlap": 0,
        "fetched_media_count": len(records),
        "fetched_media_bytes": sum(row["media_bytes"] for row in records),
        "planned_uncompressed_bytes": planned_uncompressed_bytes,
        "planned_compressed_bytes": planned_compressed_bytes,
        "free_bytes_before_fetch": available_bytes,
        "required_free_bytes_before_fetch": required_free_bytes,
        "media_manifest_sha256": sha256_file(output_dir / "media-manifest.jsonl"),
    }
    (output_dir / "fetch-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--resume-existing-incomplete",
        action="store_true",
        help="verify existing media against the pinned ZIP CRC and fetch only missing files",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        raise ValueError("workers must be between 1 and 12")
    report = fetch_dataset(
        args.manifest_dir,
        args.output_dir,
        workers=args.workers,
        resume_existing_incomplete=args.resume_existing_incomplete,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
