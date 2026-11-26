"""Acquire only the newly selected train media with pinned HTTP byte ranges."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import shutil
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://storage.googleapis.com/dm-perception-test/zip_data/"
ARCHIVES = {
    "train_videos.zip": (28_404_087_974, '"-CJXz5tD4nP8CEAE="'),
    "train_audios.zip": (13_222_906_156, '"-CMb2h5uDnf8CEAE="'),
}
MAX_EXPANDED_BYTES = 5_000_000_000
MIN_FREE_AFTER_ACQUISITION = 1_500_000_000
MAX_FILE_BYTES = 120_000_000


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int, etag: str) -> None:
        self.url, self.size, self.etag = url, size, etag
        self.position = 0
        self.bytes_read = 0
        self.session = requests.Session()
        response = self.session.head(url, timeout=30)
        response.raise_for_status()
        if int(response.headers.get("Content-Length", -1)) != size:
            raise ValueError("archive size pin mismatch")
        if response.headers.get("ETag") != etag:
            raise ValueError("archive ETag pin mismatch")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        position = (
            offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        )
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size or size == 0:
            return b""
        size = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        end = self.position + size - 1
        response = self.session.get(
            self.url,
            headers={
                "Range": f"bytes={self.position}-{end}",
                "Accept-Encoding": "identity",
                "If-Match": self.etag,
            },
            timeout=120,
        )
        expected = f"bytes {self.position}-{end}/{self.size}"
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != expected
            or response.headers.get("ETag") != self.etag
            or len(response.content) != size
        ):
            raise RuntimeError("strict HTTP byte-range integrity check failed")
        self.position += size
        self.bytes_read += size
        return response.content

    def read_range(self, start: int, size: int) -> bytes:
        if start < 0 or size < 0 or start + size > self.size:
            raise ValueError("range is outside the pinned archive")
        if size == 0:
            return b""
        end = start + size - 1
        response = self.session.get(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "If-Match": self.etag,
            },
            timeout=120,
        )
        expected = f"bytes {start}-{end}/{self.size}"
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != expected
            or response.headers.get("ETag") != self.etag
            or len(response.content) != size
        ):
            raise RuntimeError("strict HTTP byte-range integrity check failed")
        self.bytes_read += size
        return response.content


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_member_bytes(remote: RangeFile, info: zipfile.ZipInfo) -> bytes:
    header = remote.read_range(info.header_offset, 30)
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50:
        raise ValueError(f"invalid local ZIP header: {info.filename}")
    name_length, extra_length = fields[-2:]
    compressed_start = info.header_offset + 30 + name_length + extra_length
    compressed = remote.read_range(compressed_start, info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        content = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -zlib.MAX_WBITS)
    elif info.compress_type == zipfile.ZIP_BZIP2:
        content = bz2.decompress(compressed)
    else:
        raise ValueError(f"unsupported ZIP compression method {info.compress_type}")
    if len(content) != info.file_size or (zlib.crc32(content) & 0xFFFFFFFF) != info.CRC:
        raise RuntimeError(f"expanded ZIP member failed size/CRC validation: {info.filename}")
    return content


def acquire(
    manifest_path: Path,
    output: Path,
    *,
    existing_media_root: Path | None,
    preflight_only: bool,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows = manifest["train"]
    unique_ids = sorted({row["video_id"] for row in train_rows})
    missing_ids = [
        video_id
        for video_id in unique_ids
        if not all(
            (output / "train" / f"{video_id}.{ext}").is_file()
            or (
                existing_media_root is not None
                and (existing_media_root / "train" / f"{video_id}.{ext}").is_file()
            )
            for ext in ("mp4", "wav")
        )
    ]
    archive_records: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    total_expanded = 0

    for archive_name, (size, etag) in ARCHIVES.items():
        media_kind = "videos" if "videos" in archive_name else "audios"
        extension = "mp4" if media_kind == "videos" else "wav"
        remote = RangeFile(BASE_URL + archive_name, size, etag)
        with zipfile.ZipFile(remote) as archive:
            for video_id in missing_ids:
                member = f"{media_kind}/{video_id}.{extension}"
                info = archive.getinfo(member)
                if info.file_size > MAX_FILE_BYTES:
                    raise ValueError(f"per-file expanded-size cap exceeded: {member}")
                total_expanded += info.file_size
                files.append(
                    {
                        "video_id": video_id,
                        "split": "train",
                        "archive": archive_name,
                        "member": member,
                        "expanded_bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                    }
                )
            archive_records[archive_name] = {
                "size": size,
                "etag": etag,
                "central_directory_bytes_read": remote.bytes_read,
            }

    if total_expanded > MAX_EXPANDED_BYTES:
        raise ValueError(
            f"selected train media require {total_expanded} bytes, over cap {MAX_EXPANDED_BYTES}"
        )
    free_before = shutil.disk_usage(output.parent if output.parent.exists() else output.anchor).free
    if free_before < total_expanded + MIN_FREE_AFTER_ACQUISITION:
        raise ValueError(
            f"insufficient disk headroom: free={free_before}, selected={total_expanded}, "
            f"required reserve={MIN_FREE_AFTER_ACQUISITION}"
        )
    report: dict[str, Any] = {
        "status": "PASS_MEDIA_SIZE_PREFLIGHT" if preflight_only else "RUNNING_MEDIA_ACQUISITION",
        "manifest_sha256": sha256_file(manifest_path),
        "missing_video_count": len(missing_ids),
        "missing_media_file_count": len(files),
        "expanded_bytes": total_expanded,
        "max_expanded_bytes": MAX_EXPANDED_BYTES,
        "free_bytes_before_acquisition": free_before,
        "minimum_free_bytes_after_acquisition": MIN_FREE_AFTER_ACQUISITION,
        "files": files,
        "archives": archive_records,
    }
    if preflight_only:
        return report

    if output.exists() and any(output.iterdir()):
        unexpected = [
            path
            for path in output.rglob("*")
            if path.is_file()
            and path.name != ".temporal-match-acquisition.json"
            and path not in {output / "train" / Path(row["member"]).name for row in files}
        ]
        if unexpected:
            raise FileExistsError(
                f"refusing to mix acquisition into unrelated output: {unexpected[0]}"
            )
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".temporal-match-acquisition.json"
    if marker.exists():
        marker_value = json.loads(marker.read_text(encoding="utf-8"))
        if marker_value.get("manifest_sha256") != report["manifest_sha256"]:
            raise ValueError("existing acquisition marker belongs to another manifest")
    else:
        marker.write_text(
            json.dumps({"manifest_sha256": report["manifest_sha256"]}, indent=2) + "\n",
            encoding="utf-8",
        )
    file_map = {(row["archive"], row["member"]): row for row in files}
    acquisition_bytes_read: dict[str, int] = {}
    for archive_name, (size, etag) in ARCHIVES.items():
        remote = RangeFile(BASE_URL + archive_name, size, etag)
        with zipfile.ZipFile(remote) as archive:
            for (wanted_archive, member), row in file_map.items():
                if wanted_archive != archive_name:
                    continue
                info = archive.getinfo(member)
                destination = output / "train" / Path(member).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if destination.stat().st_size != info.file_size:
                        raise FileExistsError(
                            f"partial acquisition file needs manual recovery: {destination}"
                        )
                    with destination.open("rb") as source:
                        crc = 0
                        digest = hashlib.sha256()
                        for block in iter(lambda: source.read(1 << 20), b""):
                            crc = zlib.crc32(block, crc)
                            digest.update(block)
                    if (crc & 0xFFFFFFFF) != info.CRC:
                        raise ValueError(f"existing output CRC mismatch: {destination}")
                    content_hash = digest.hexdigest()
                else:
                    content = extract_member_bytes(remote, info)
                    digest = hashlib.sha256(content).hexdigest()
                    temporary = destination.with_suffix(destination.suffix + ".tmp")
                    with temporary.open("xb") as target:
                        target.write(content)
                    temporary.replace(destination)
                    content_hash = digest
                row["sha256"] = content_hash
                row["output_path"] = str(destination)
                row["entry_verified_to_eof"] = True
                print(f"verified {member}", flush=True)
        acquisition_bytes_read[archive_name] = remote.bytes_read
    report["status"] = "PASS_TRAIN_MEDIA_ACQUISITION"
    report["archives"] = {
        name: {**record, "acquisition_bytes_read": acquisition_bytes_read[name]}
        for name, record in archive_records.items()
    }
    report["train_validation_source_identity_overlap"] = 0
    report["test_split_opened"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-media-root", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.report}")
    result = acquire(
        args.manifest,
        args.output,
        existing_media_root=args.existing_media_root,
        preflight_only=args.preflight_only,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
