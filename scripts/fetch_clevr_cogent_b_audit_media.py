"""Fetch all pinned CoGenT-B audit images by verified ZIP byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

SOURCE_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip"
SOURCE_BYTES = 24_719_655_191
SOURCE_ETAG = '"a2125a3153354b3b09849befcdd79ad1-1572"'
SOURCE_VERSION = "haDYjDZdal8NjYimkkMSeCanaOmhAc5y"
EXPECTED_VALB_MEMBER_NAMES_SHA256 = (
    "f79ebb62a5489d396ea288280d76036376d76d757646d07ac22afedadf1d4227"
)
EXPECTED_VALB_IMAGE_COUNT = 15_000
MIN_FREE_RESERVE_BYTES = 2 * 1024**3


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_member(filename: str) -> str:
    if Path(filename).name != filename or not filename.startswith("CLEVR_valB_"):
        raise ValueError("CoGenT-B filename must be a valB image basename")
    return f"CLEVR_CoGenT_v1.0/images/valB/{filename}"


def fetch_range(start: int, end: int) -> bytes:
    if start < 0 or end < start or end >= SOURCE_BYTES:
        raise ValueError("requested byte range is outside the pinned CoGenT archive")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "Range": f"bytes={start}-{end}",
            "If-Match": SOURCE_ETAG,
            "User-Agent": "tiny-omni-clevr-cogent-b-audit/1",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                expected = f"bytes {start}-{end}/{SOURCE_BYTES}"
                if response.status != 206:
                    raise OSError(f"expected HTTP 206, got {response.status}")
                if response.headers.get("Content-Range") != expected:
                    raise OSError("CoGenT archive Content-Range mismatch")
                if response.headers.get("ETag") != SOURCE_ETAG:
                    raise OSError("CoGenT archive ETag mismatch")
                payload = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    if len(payload) != end - start + 1:
        raise OSError("short CoGenT archive byte-range response")
    return payload


class RemoteArchive:
    """Seekable facade that retrieves only requested remote ZIP ranges."""

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
            raise ValueError("seek outside pinned CoGenT archive")
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = SOURCE_BYTES - self.position
        size = min(size, SOURCE_BYTES - self.position)
        if size <= 0:
            return b""
        payload = fetch_range(self.position, self.position + size - 1)
        self.position += len(payload)
        return payload

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        pass


def extract_member(info: zipfile.ZipInfo) -> tuple[bytes, str]:
    if info.compress_type != zipfile.ZIP_DEFLATED or info.flag_bits & 0x1:
        raise ValueError(f"unexpected compression or encryption for {info.filename}")
    local = fetch_range(info.header_offset, info.header_offset + 29)
    fields = struct.unpack("<IHHHHHIIIHH", local)
    if fields[0] != 0x04034B50:
        raise ValueError(f"invalid ZIP local header for {info.filename}")
    name_length, extra_length = fields[-2:]
    local_name = fetch_range(
        info.header_offset + 30, info.header_offset + 29 + name_length
    )
    if local_name.decode("utf-8") != info.filename:
        raise ValueError(f"local/central ZIP names differ for {info.filename}")
    data_start = info.header_offset + 30 + name_length + extra_length
    compressed = fetch_range(data_start, data_start + info.compress_size - 1)
    payload = zlib.decompress(compressed, -15)
    if len(payload) != info.file_size or zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"CoGenT ZIP size/CRC mismatch for {info.filename}")
    return payload, sha256_bytes(payload)


def read_manifest(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != EXPECTED_VALB_IMAGE_COUNT:
        raise ValueError("audit manifest must cover every 15,000 valB image")
    if any(row.get("split") != "cogent_b_audit" for row in rows):
        raise ValueError("manifest must contain only cogent_b_audit rows")
    if len({row.get("image_filename") for row in rows}) != EXPECTED_VALB_IMAGE_COUNT:
        raise ValueError("audit manifest repeats or omits a valB image")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    rows = read_manifest(args.manifest)
    metadata = json.loads(args.manifest_metadata.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if metadata.get("manifest_sha256") != manifest_sha:
        raise ValueError("audit manifest metadata hash mismatch")

    started = datetime.now(UTC)
    wall_start = time.perf_counter()
    required_names = {archive_member(str(row["image_filename"])) for row in rows}
    with zipfile.ZipFile(RemoteArchive()) as archive:
        valb_infos = [
            info
            for info in archive.infolist()
            if info.filename.startswith("CLEVR_CoGenT_v1.0/images/valB/")
            and info.filename.lower().endswith(".png")
        ]
        member_names_hash = sha256_bytes(
            "\n".join(info.filename for info in valb_infos).encode()
        )
        if (
            len(valb_infos) != EXPECTED_VALB_IMAGE_COUNT
            or member_names_hash != EXPECTED_VALB_MEMBER_NAMES_SHA256
        ):
            raise ValueError("pinned valB archive member inventory mismatch")
        infos = {info.filename: info for info in valb_infos}
        if required_names != set(infos):
            raise ValueError("audit manifest does not cover the exact full valB image set")
        expanded_bytes = sum(infos[name].file_size for name in required_names)
        compressed_bytes = sum(infos[name].compress_size for name in required_names)
        free_bytes = shutil.disk_usage(args.output.parent).free
        if free_bytes < expanded_bytes + MIN_FREE_RESERVE_BYTES:
            raise OSError(
                f"need {expanded_bytes + MIN_FREE_RESERVE_BYTES} bytes including reserve; "
                f"only {free_bytes} bytes are free"
            )
        print(
            f"preflight passed: all {len(infos)} valB images, compressed={compressed_bytes}, "
            f"expanded={expanded_bytes}, free={free_bytes}",
            flush=True,
        )

        jobs = [(row, infos[archive_member(str(row["image_filename"]))]) for row in rows]
        jobs.sort(key=lambda job: job[1].header_offset)
        args.output.mkdir(parents=True)
        media_dir = args.output / "cogent_b_audit"
        media_dir.mkdir()
        records: dict[str, dict[str, object]] = {}

        def save_job(job):
            row, info = job
            payload, digest = extract_member(info)
            path = media_dir / str(row["image_filename"])
            path.write_bytes(payload)
            return str(row["sample_id"]), {
                "sample_id": row["sample_id"],
                "image_filename": row["image_filename"],
                "archive_member": info.filename,
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
                    print(f"saved {index}/{len(jobs)} audit images", flush=True)

    ordered = [records[str(row["sample_id"])] for row in rows]
    receipt = {
        "status": "FULL_COGENT_B_MEDIA_FETCHED_AND_CRC_VERIFIED",
        "source_url": SOURCE_URL,
        "source_version_id": SOURCE_VERSION,
        "source_etag": SOURCE_ETAG,
        "source_size_bytes": SOURCE_BYTES,
        "valb_member_names_sha256": EXPECTED_VALB_MEMBER_NAMES_SHA256,
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - wall_start,
        "workers": args.workers,
        "manifest_sha256": manifest_sha,
        "manifest_metadata_sha256": hashlib.sha256(args.manifest_metadata.read_bytes()).hexdigest(),
        "images": len(rows),
        "expanded_bytes": expanded_bytes,
        "compressed_bytes": compressed_bytes,
        "image_sha256_ordered": ordered,
    }
    receipt_path = args.output / "acquisition.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    printable = {key: value for key, value in receipt.items() if key != "image_sha256_ordered"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
