"""Fetch only the pinned CLEVR images named in train/development manifests."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import struct
import time
import urllib.request
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

SOURCE_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
SOURCE_BYTES = 19_021_600_724
SOURCE_ETAG = '"3426f0a360618fcf2179966a8d1428c3-1210"'
SOURCE_VERSION = "B2HsawCh6GoabZXQDbLoboNXB4FrlnvC"
MIN_FREE_RESERVE_BYTES = 2 * 1024**3


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_member(split: str, filename: str) -> str:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    archive_split = "val" if split == "validation" else "train"
    return f"CLEVR_v1.0/images/{archive_split}/{filename}"


def fetch_range(start: int, end: int) -> bytes:
    if start < 0 or end < start or end >= SOURCE_BYTES:
        raise ValueError("requested byte range is outside the pinned CLEVR archive")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "Range": f"bytes={start}-{end}",
            "If-Match": SOURCE_ETAG,
            "User-Agent": "tiny-omni-clevr-vqa/1",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        expected_range = f"bytes {start}-{end}/{SOURCE_BYTES}"
        if response.status != 206:
            raise OSError(f"expected HTTP 206, got {response.status}")
        if response.headers.get("Content-Range") != expected_range:
            raise OSError("CLEVR archive Content-Range did not match requested bytes")
        if response.headers.get("ETag") != SOURCE_ETAG:
            raise OSError("CLEVR archive ETag did not match the pinned source")
        payload = response.read()
    if len(payload) != end - start + 1:
        raise OSError("short CLEVR archive byte-range response")
    return payload


class RemoteArchive:
    """Minimal seekable ZIP reader used to load only the remote central directory."""

    def __init__(self) -> None:
        self.position = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = SOURCE_BYTES + offset
        else:
            raise ValueError("invalid seek mode")
        if not 0 <= self.position <= SOURCE_BYTES:
            raise ValueError("seek outside pinned archive")
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = SOURCE_BYTES - self.position
        size = min(size, SOURCE_BYTES - self.position)
        if size <= 0:
            return b""
        start = self.position
        payload = fetch_range(start, start + size - 1)
        self.position += len(payload)
        return payload

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        pass


def extract_member(info: zipfile.ZipInfo) -> tuple[bytes, str]:
    if info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError(f"unexpected ZIP compression method for {info.filename}")
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted ZIP member is not allowed: {info.filename}")
    local_header = fetch_range(info.header_offset, info.header_offset + 29)
    fields = struct.unpack("<IHHHHHIIIHH", local_header)
    if fields[0] != 0x04034B50:
        raise ValueError(f"invalid local ZIP header for {info.filename}")
    name_length, extra_length = fields[-2:]
    local_name = fetch_range(info.header_offset + 30, info.header_offset + 29 + name_length)
    if local_name.decode("utf-8") != info.filename:
        raise ValueError(f"local/central ZIP names differ for {info.filename}")
    data_start = info.header_offset + 30 + name_length + extra_length
    compressed = fetch_range(data_start, data_start + info.compress_size - 1)
    payload = zlib.decompress(compressed, -15)
    if len(payload) != info.file_size:
        raise ValueError(f"uncompressed byte count mismatch for {info.filename}")
    if zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"ZIP CRC mismatch for {info.filename}")
    return payload, sha256_bytes(payload)


def read_manifest(path: Path, expected_split: str) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(row.get("split") != expected_split for row in rows):
        raise ValueError(f"{path} is empty or does not contain only {expected_split} rows")
    if len({row.get("image_filename") for row in rows}) != len(rows):
        raise ValueError(f"{path} repeats an image identity")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    train = read_manifest(args.train_manifest, "train")
    validation = read_manifest(args.validation_manifest, "validation")
    train_names = {str(row["image_filename"]) for row in train}
    validation_names = {str(row["image_filename"]) for row in validation}
    if train_names & validation_names:
        raise ValueError("train/validation image identity overlap")
    rows_by_split = {"train": train, "validation": validation}
    required_names = {
        archive_member(split, str(row["image_filename"]))
        for split, rows in rows_by_split.items()
        for row in rows
    }
    started = datetime.now(UTC)
    wall_start = time.perf_counter()
    with zipfile.ZipFile(RemoteArchive()) as archive:
        infos = {}
        for name in required_names:
            try:
                infos[name] = archive.getinfo(name)
            except KeyError as exc:
                raise ValueError(f"image missing from pinned source archive: {name}") from exc
        expanded_bytes = sum(info.file_size for info in infos.values())
        free_bytes = shutil.disk_usage(args.output.parent).free
        if free_bytes < expanded_bytes + MIN_FREE_RESERVE_BYTES:
            raise OSError(
                f"need {expanded_bytes + MIN_FREE_RESERVE_BYTES} bytes including disk reserve; "
                f"only {free_bytes} bytes are free"
            )

        jobs = []
        for split, rows in rows_by_split.items():
            for row in rows:
                member = archive_member(split, str(row["image_filename"]))
                jobs.append((split, row, infos[member]))
        jobs.sort(key=lambda item: item[2].header_offset)
        args.output.mkdir(parents=True)
        records: dict[str, dict[str, object]] = {}

        def save_job(job):
            split, row, info = job
            payload, digest = extract_member(info)
            if Path(str(row["image_filename"])).name != str(row["image_filename"]):
                raise ValueError("image filename is not a basename")
            path = args.output / split / str(row["image_filename"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return str(row["sample_id"]), {
                "split": split,
                "sample_id": row["sample_id"],
                "image_filename": row["image_filename"],
                "member": info.filename,
                "image_sha256": digest,
                "uncompressed_bytes": len(payload),
                "zip_crc32": f"{info.CRC:08x}",
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(save_job, job) for job in jobs]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                sample_id, record = future.result()
                records[sample_id] = record
                if index % 512 == 0:
                    print(f"saved {index}/{len(jobs)} selected images", flush=True)

    ordered_records = [
        records[str(row["sample_id"])]
        for split in ("train", "validation")
        for row in rows_by_split[split]
    ]
    receipt = {
        "status": "SELECTED_MEDIA_FETCHED_AND_CRC_VERIFIED",
        "source_url": SOURCE_URL,
        "source_version_id": SOURCE_VERSION,
        "source_etag": SOURCE_ETAG,
        "source_size_bytes": SOURCE_BYTES,
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - wall_start,
        "workers": args.workers,
        "train_manifest_sha256": hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(),
        "validation_manifest_sha256": hashlib.sha256(
            args.validation_manifest.read_bytes()
        ).hexdigest(),
        "train_validation_image_overlap": 0,
        "train_images": len(train),
        "validation_images": len(validation),
        "expanded_bytes": expanded_bytes,
        "image_sha256_ordered": ordered_records,
    }
    (args.output / "acquisition.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    printable = {
        key: value for key, value in receipt.items() if key != "image_sha256_ordered"
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
