"""Inspect or fetch only per-file allowlisted BSD10k audio via exact HTTP ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import time
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import requests

try:
    from scripts.audit_bsd10k_metadata import (
        ALLOWED_LICENSE_PATHS,
        canonical_license_with_explicit_extras,
    )
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from audit_bsd10k_metadata import (
        ALLOWED_LICENSE_PATHS,
        canonical_license_with_explicit_extras,
    )

EXPECTED_RECORD_ID = "19868804"
EXPECTED_ARCHIVE_BYTES = 8_576_285_410
DEFAULT_ARCHIVE_URL = "https://zenodo.org/api/records/19868804/files/audio.zip/content"
SPLIT_NAMES = {"train", "validation", "test"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HTTPRangeReader:
    """Seekable, fail-closed reader that requests only the exact requested range."""

    def __init__(
        self,
        url: str,
        size: int = EXPECTED_ARCHIVE_BYTES,
        timeout: int = 60,
        session: requests.Session | None = None,
    ):
        self.url = url
        self.size = size
        self.timeout = timeout
        self.position = 0
        self.bytes_read = 0
        self.session = session or requests.Session()
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("invalid seek mode")
        if position < 0 or position > self.size:
            raise ValueError("seek outside pinned archive size")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size == 0 or self.position == self.size:
            return b""
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size > 64 * 1024 * 1024:
            raise OSError("refusing an unexpectedly large metadata range request")
        start, end = self.position, self.position + size - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            timeout=self.timeout,
        )
        try:
            content_range = response.headers.get("Content-Range", "")
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status_code != 206 or content_range != expected:
                raise OSError("archive server did not honor the exact byte-range request")
            payload = response.content
        finally:
            response.close()
        if len(payload) != size:
            raise OSError("range response length did not match the requested byte count")
        self.position += size
        self.bytes_read += size
        return payload


def _normalize_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(ranges)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if start < 0 or end < start:
            raise ValueError("invalid byte range")
        if merged and start <= merged[-1][1]:
            raise ValueError("overlapping archive ranges are not permitted")
        if merged and start == merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _parse_multipart_ranges(
    payload: bytes,
    expected_ranges: list[tuple[int, int]],
    archive_size: int,
) -> dict[tuple[int, int], bytes]:
    marker = b"\n--EOSMULTIPARTBOUNDARY\n"
    final_marker = b"\n--EOSMULTIPARTBOUNDARY--\n"
    position = 0
    parts: dict[tuple[int, int], bytes] = {}
    while position < len(payload):
        if payload.startswith(final_marker, position):
            position += len(final_marker)
            break
        if not payload.startswith(marker, position):
            raise OSError("unexpected multipart range response boundary")
        position += len(marker)
        header_end = payload.find(b"\n\n", position)
        if header_end < 0:
            raise OSError("multipart range part is missing its headers")
        headers = payload[position:header_end].decode("ascii", errors="strict")
        match = re.search(r"(?im)^Content-Range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", headers)
        if match is None:
            raise OSError("multipart range part lacks a valid Content-Range")
        start, end, declared_size = (int(value) for value in match.groups())
        if declared_size != archive_size or end < start:
            raise OSError("multipart range part has the wrong archive size")
        content_start = header_end + 2
        content_end = content_start + (end - start + 1)
        if content_end > len(payload):
            raise OSError("multipart range part is truncated")
        parts[(start, end)] = payload[content_start:content_end]
        position = content_end
    if position != len(payload) or set(parts) != set(expected_ranges):
        raise OSError("multipart response ranges differ from the exact request")
    return parts


def _read_range_batch(
    session: requests.Session,
    url: str,
    archive_size: int,
    ranges: list[tuple[int, int]],
    *,
    timeout: int = 90,
) -> dict[tuple[int, int], bytes]:
    normalized = _normalize_ranges(ranges)
    if not normalized:
        return {}
    range_header = "bytes=" + ",".join(f"{start}-{end}" for start, end in normalized)
    expected_bytes = sum(end - start + 1 for start, end in normalized)
    for attempt in range(6):
        response = session.get(
            url,
            headers={"Range": range_header, "Accept-Encoding": "identity"},
            timeout=timeout,
        )
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 5:
            retry_after = response.headers.get("Retry-After", "")
            response.close()
            if response.status_code == 429:
                try:
                    delay = min(max(float(retry_after), 1.0), 120.0)
                except ValueError:
                    delay = min(2**attempt, 30)
            else:
                delay = min(2**attempt, 30)
            time.sleep(delay)
            continue
        try:
            if response.status_code != 206:
                raise OSError(f"archive range request returned HTTP {response.status_code}")
            content_range = response.headers.get("Content-Range", "")
            body = response.content
            if len(normalized) == 1:
                start, end = normalized[0]
                expected = f"bytes {start}-{end}/{archive_size}"
                if content_range != expected or len(body) != end - start + 1:
                    raise OSError("single-range response does not match the exact requested bytes")
                return {normalized[0]: body}
            if len(body) > expected_bytes + 256 * len(normalized) + 4096:
                raise OSError("multipart response exceeds range bytes plus bounded headers")
            return _parse_multipart_ranges(body, normalized, archive_size)
        finally:
            response.close()
    raise OSError("archive range request exceeded retry limit")


def _extract_member_bytes(
    info: zipfile.ZipInfo,
    read_range: Callable[[int, int], bytes],
) -> bytes:
    """Read and verify one ZIP member without fetching adjacent archive bytes."""
    prefix_length = 30 + len(info.filename.encode("utf-8"))
    prefix = read_range(info.header_offset, prefix_length)
    data_offset = _member_data_offset(info, prefix)
    compressed = read_range(data_offset, info.compress_size)
    return _decode_member(info, compressed)


def _member_data_offset(info: zipfile.ZipInfo, prefix: bytes) -> int:
    name_length_expected = len(info.filename.encode("utf-8"))
    if len(prefix) != 30 + name_length_expected:
        raise OSError("truncated local ZIP header prefix")
    fixed = prefix[:30]
    signature, _, _, method, _, _, _, _, _, name_length, extra_length = struct.unpack(
        "<IHHHHHIIIHH", fixed
    )
    if signature != 0x04034B50 or method != info.compress_type:
        raise OSError("local ZIP header signature or compression method mismatch")
    if name_length != name_length_expected:
        raise OSError("local ZIP filename length differs from central directory")
    local_name = prefix[30:].decode("utf-8")
    if local_name != info.filename:
        raise OSError("local ZIP filename does not match central-directory entry")
    return info.header_offset + 30 + name_length + extra_length


def _decode_member(info: zipfile.ZipInfo, compressed: bytes) -> bytes:
    if len(compressed) != info.compress_size:
        raise OSError("truncated compressed ZIP member")
    method = info.compress_type
    if method == zipfile.ZIP_STORED:
        payload = compressed
    elif method == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise OSError(f"unsupported ZIP compression method {method}")
    if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
        raise OSError("extracted member failed declared size or CRC-32 validation")
    return payload


def _read_ranges_preserving(
    session: requests.Session,
    url: str,
    archive_size: int,
    ranges: list[tuple[int, int]],
) -> dict[tuple[int, int], bytes]:
    normalized = _normalize_ranges(ranges)
    parts = _read_range_batch(session, url, archive_size, ranges)
    result: dict[tuple[int, int], bytes] = {}
    for start, end in ranges:
        container = next(
            (
                candidate
                for candidate in normalized
                if candidate[0] <= start and candidate[1] >= end
            ),
            None,
        )
        if container is None:
            raise OSError("server response did not include an exact requested range")
        offset = start - container[0]
        result[(start, end)] = parts[container][offset : offset + end - start + 1]
    return result


def _allowed_ids(
    split_manifest: Path | None,
    split: str | None,
    extra_allowed_licenses: set[str] | None = None,
) -> set[str] | None:
    if split_manifest is None:
        if split is not None:
            raise ValueError("--split requires --split-manifest")
        return None
    if split not in SPLIT_NAMES:
        raise ValueError("--split must be train, validation, or test")
    document = json.loads(split_manifest.read_text(encoding="utf-8"))
    if document.get("source_record_id") != EXPECTED_RECORD_ID:
        raise ValueError("split manifest is not pinned to the expected BSD10k release")
    if document.get("schema") != "tiny-omni-decision/bsd10k-group-split-v1":
        raise ValueError("split manifest schema is not supported")
    if document.get("metadata_zip_sha256") is None:
        raise ValueError("split manifest lacks the source metadata hash")
    rows = document.get("examples")
    if not isinstance(rows, list):
        raise ValueError("split manifest lacks example assignments")
    selected: set[str] = set()
    seen_ids: set[str] = set()
    group_split: dict[str, str] = {}
    for row in rows:
        if row.get("license") not in set(ALLOWED_LICENSE_PATHS.values()) | (
            extra_allowed_licenses or set()
        ):
            raise ValueError("split manifest contains a non-allowlisted media license")
        if row.get("split") not in SPLIT_NAMES:
            raise ValueError("split manifest contains an invalid split label")
        sound_id = str(row.get("sound_id", ""))
        if not re.fullmatch(r"[0-9]+", sound_id) or sound_id in seen_ids:
            raise ValueError("split manifest contains a malformed or duplicate sound ID")
        seen_ids.add(sound_id)
        group_hash = str(row.get("uploader_group_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", group_hash):
            raise ValueError("split manifest contains a malformed uploader group hash")
        previous_split = group_split.setdefault(group_hash, str(row["split"]))
        if previous_split != row["split"]:
            raise ValueError("split manifest leaks an uploader group across splits")
        if row.get("split") == split:
            selected.add(sound_id)
    if not selected:
        raise ValueError("selected split contains no allowlisted sound IDs")
    return selected


def inspect_archive(
    metadata_zip: Path,
    *,
    archive_url: str = DEFAULT_ARCHIVE_URL,
    archive_size: int = EXPECTED_ARCHIVE_BYTES,
    extra_allowed_licenses: set[str] | None = None,
) -> dict[str, object]:
    import csv
    import io

    with zipfile.ZipFile(metadata_zip) as metadata_archive:
        csv_name = next(
            name
            for name in metadata_archive.namelist()
            if name.casefold().endswith("bsd10k_metadata.csv")
        )
        with metadata_archive.open(csv_name) as source:
            rows = list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig", newline="")))
    allowed = {
        row["sound_id"].strip()
        for row in rows
        if canonical_license_with_explicit_extras(row["license"], extra_allowed_licenses or set())
        in set(ALLOWED_LICENSE_PATHS.values()) | (extra_allowed_licenses or set())
    }
    if not allowed:
        raise ValueError("metadata has no media with an allowlisted per-file license")

    reader = HTTPRangeReader(archive_url, archive_size)
    try:
        with zipfile.ZipFile(reader) as audio_archive:
            by_id: dict[str, zipfile.ZipInfo] = {}
            for info in audio_archive.infolist():
                match = re.fullmatch(r"audio/([0-9]+)\.wav", info.filename)
                if match:
                    sound_id = match.group(1)
                    if sound_id in by_id:
                        raise ValueError("archive contains a duplicate sound ID member")
                    by_id[sound_id] = info
            missing = allowed - by_id.keys()
            if missing:
                raise ValueError("archive is missing one or more allowlisted metadata records")
            selected = [by_id[sound_id] for sound_id in allowed]
        metadata_range_bytes = reader.bytes_read
    finally:
        reader.close()

    return {
        "source_record_id": EXPECTED_RECORD_ID,
        "archive_size_bytes": archive_size,
        "archive_entries": len(by_id),
        "allowlisted_entries": len(selected),
        "allowlisted_compressed_bytes": sum(info.compress_size for info in selected),
        "allowlisted_uncompressed_bytes": sum(info.file_size for info in selected),
        "compression_methods": dict(Counter(str(info.compress_type) for info in selected)),
        "unmatched_allowlisted_ids": 0,
        "zip_metadata_range_bytes_read": metadata_range_bytes,
        "media_payload_or_hashes_inspected": False,
    }


def download_split(
    metadata_zip: Path,
    split_manifest: Path,
    split: str,
    output_dir: Path,
    *,
    archive_url: str = DEFAULT_ARCHIVE_URL,
    archive_size: int = EXPECTED_ARCHIVE_BYTES,
    limit: int | None = None,
    extra_allowed_licenses: set[str] | None = None,
) -> dict[str, object]:
    accepted = set(ALLOWED_LICENSE_PATHS.values()) | (extra_allowed_licenses or set())
    selected_ids = _allowed_ids(split_manifest, split, extra_allowed_licenses)
    assert selected_ids is not None
    import csv
    import io

    with zipfile.ZipFile(metadata_zip) as metadata_archive:
        csv_name = next(
            name
            for name in metadata_archive.namelist()
            if name.casefold().endswith("bsd10k_metadata.csv")
        )
        with metadata_archive.open(csv_name) as source:
            rows = list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig", newline="")))
    license_by_id = {
        row["sound_id"].strip(): canonical_license_with_explicit_extras(
            row["license"], extra_allowed_licenses or set()
        )
        for row in rows
    }
    if any(license_by_id.get(sound_id) not in accepted for sound_id in selected_ids):
        raise ValueError("split manifest IDs do not match allowlisted metadata licenses")
    split_document = json.loads(split_manifest.read_text(encoding="utf-8"))
    if split_document["metadata_zip_sha256"] != sha256_file(metadata_zip):
        raise ValueError("split manifest metadata hash does not match the pinned metadata archive")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {row["sound_id"]: row for row in split_document["examples"]}
    class_by_id = {sound_id: row["class"] for sound_id, row in records.items()}
    downloaded: list[dict[str, object]] = []
    range_bytes_fetched = 0
    member_range_requests = 0
    skipped_existing = 0

    def fetch_batched(
        session: requests.Session,
        ranges: list[tuple[int, int]],
        max_count: int,
        max_bytes: int,
    ):
        nonlocal range_bytes_fetched, member_range_requests
        batch: list[tuple[int, int]] = []
        batch_bytes = 0
        for byte_range in ranges:
            length = byte_range[1] - byte_range[0] + 1
            if length > 64 * 1024 * 1024:
                raise OSError("member range exceeds the bounded memory limit")
            if batch and (len(batch) >= max_count or batch_bytes + length > max_bytes):
                received = _read_ranges_preserving(session, archive_url, archive_size, batch)
                range_bytes_fetched += batch_bytes
                member_range_requests += 1
                yield received
                batch, batch_bytes = [], 0
            batch.append(byte_range)
            batch_bytes += length
            if length > max_bytes:
                received = _read_ranges_preserving(session, archive_url, archive_size, batch)
                range_bytes_fetched += batch_bytes
                member_range_requests += 1
                yield received
                batch, batch_bytes = [], 0
        if batch:
            received = _read_ranges_preserving(session, archive_url, archive_size, batch)
            range_bytes_fetched += batch_bytes
            member_range_requests += 1
            yield received

    with requests.Session() as session:
        reader = HTTPRangeReader(archive_url, archive_size, session=session)
        try:
            with zipfile.ZipFile(reader) as audio_archive:
                infos: dict[str, zipfile.ZipInfo] = {}
                for info in audio_archive.infolist():
                    match = re.fullmatch(r"audio/([0-9]+)\.wav", info.filename)
                    if match and match.group(1) in selected_ids:
                        infos[match.group(1)] = info
                if infos.keys() != selected_ids:
                    raise ValueError("selected split and archive member IDs do not match exactly")
                ids = sorted(selected_ids, key=lambda sound_id: int(sound_id))
                if limit is not None:
                    if limit < 1:
                        raise ValueError("--limit must be positive")
                    ids = ids[:limit]
                pending: list[str] = []
                for sound_id in ids:
                    info = infos[sound_id]
                    if info.file_size > 64 * 1024 * 1024 or info.compress_size > 64 * 1024 * 1024:
                        raise OSError("member exceeds the bounded per-file memory limit")
                    destination = output_dir / f"{sound_id}.wav"
                    partial = output_dir / f"{sound_id}.wav.part"
                    if partial.exists():
                        raise FileExistsError(f"unverified partial file exists: {partial}")
                    if destination.exists():
                        payload_size = destination.stat().st_size
                        crc = 0
                        with destination.open("rb") as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b""):
                                crc = zlib.crc32(block, crc)
                        if payload_size != info.file_size or crc != info.CRC:
                            raise OSError(
                                f"existing audio failed ZIP size/CRC validation: {destination}"
                            )
                        downloaded.append(
                            {
                                "sound_id": sound_id,
                                "bytes": payload_size,
                                "crc32": f"{crc:08x}",
                                "sha256": sha256_file(destination),
                                "license": license_by_id[sound_id],
                                "class": class_by_id[sound_id],
                                "resumed_existing": True,
                            }
                        )
                        skipped_existing += 1
                    else:
                        pending.append(sound_id)
                required_bytes = sum(infos[sound_id].file_size for sound_id in pending)
                free_bytes = shutil.disk_usage(output_dir).free
                if free_bytes - required_bytes < 2 * 1024**3:
                    raise OSError("download would violate the fixed 2 GiB free-space reserve")
                prefix_ranges = [
                    (
                        infos[sound_id].header_offset,
                        infos[sound_id].header_offset
                        + 30
                        + len(infos[sound_id].filename.encode("utf-8"))
                        - 1,
                    )
                    for sound_id in pending
                ]
                prefixes: dict[tuple[int, int], bytes] = {}
                for received in fetch_batched(session, prefix_ranges, 50, 8 * 1024**2):
                    prefixes.update(received)
                offsets: dict[str, int] = {}
                for sound_id, byte_range in zip(pending, prefix_ranges, strict=True):
                    offsets[sound_id] = _member_data_offset(infos[sound_id], prefixes[byte_range])

                range_to_id = {
                    (
                        offsets[sound_id],
                        offsets[sound_id] + infos[sound_id].compress_size - 1,
                    ): sound_id
                    for sound_id in pending
                }
                payload_ranges = list(range_to_id)
                completed = 0
                for received in fetch_batched(session, payload_ranges, 24, 16 * 1024**2):
                    # Decode, verify, and flush one bounded response batch before the next
                    # request. Never retain the multi-gigabyte split in RAM.
                    for byte_range, compressed in received.items():
                        sound_id = range_to_id[byte_range]
                        info = infos[sound_id]
                        payload = _decode_member(info, compressed)
                        destination = output_dir / f"{sound_id}.wav"
                        temporary = output_dir / f"{sound_id}.wav.part"
                        temporary.write_bytes(payload)
                        temporary.replace(destination)
                        downloaded.append(
                            {
                                "sound_id": sound_id,
                                "bytes": len(payload),
                                "crc32": f"{zlib.crc32(payload):08x}",
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "license": license_by_id[sound_id],
                                "class": class_by_id[sound_id],
                                "resumed_existing": False,
                            }
                        )
                        completed += 1
                        if completed % 25 == 0 or completed == len(pending):
                            print(
                                f"downloaded {completed}/{len(pending)} approved "
                                f"{split} audio files"
                            )
        finally:
            reader.close()
    return {
        "source_record_id": EXPECTED_RECORD_ID,
        "split": split,
        "selected_ids": len(selected_ids),
        "downloaded_files": downloaded,
        "downloaded_count": len(downloaded),
        "downloaded_bytes": sum(int(row["bytes"]) for row in downloaded),
        "skipped_existing_count": skipped_existing,
        "member_range_request_count": member_range_requests,
        "member_range_payload_bytes": range_bytes_fetched,
        "archive_whole_file_md5_verified": False,
        "per_member_zip_crc32_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--archive-size", type=int, default=EXPECTED_ARCHIVE_BYTES)
    parser.add_argument("--inspect-output", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=sorted(SPLIT_NAMES))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--allow-license", action="append", default=[])
    args = parser.parse_args()
    if args.split_manifest or args.split:
        if not args.split_manifest or not args.split or not args.output_dir:
            parser.error("downloading requires --split-manifest, --split, and --output-dir")
        result = download_split(
            args.metadata_zip,
            args.split_manifest,
            args.split,
            args.output_dir,
            archive_url=args.archive_url,
            archive_size=args.archive_size,
            limit=args.limit,
            extra_allowed_licenses=set(args.allow_license),
        )
    else:
        result = inspect_archive(
            args.metadata_zip,
            archive_url=args.archive_url,
            archive_size=args.archive_size,
            extra_allowed_licenses=set(args.allow_license),
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered, encoding="utf-8")
    elif args.inspect_output:
        args.inspect_output.parent.mkdir(parents=True, exist_ok=True)
        args.inspect_output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
