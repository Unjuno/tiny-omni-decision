"""Verify train-manifest labels against pinned Physion++ readout metadata.

This audit reads only the supplied train JSONL. It downloads the matching
official ``.pkl`` members through conditional HTTP Range requests and uses a
strict allowlist when decoding NumPy-backed pickle metadata.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pickle
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import requests

ARCHIVE_URL = "https://physion-v2.s3.amazonaws.com/readout_data.zip"
ARCHIVE_ETAG = '"4cf84f667bcc3eaf991dbcc10dd31177-355"'
ARCHIVE_SIZE = 2_971_891_321
DEFAULT_TRAIN_SHA256 = "0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b"
_EOCD = struct.Struct("<4s4H2IH")
_CENTRAL = struct.Struct("<4s6H3I5H2I")
_LOCAL = struct.Struct("<4s5H3I2H")


@dataclass(frozen=True)
class ZipMember:
    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


class RestrictedUnpickler(pickle.Unpickler):
    """Decode the official NumPy metadata while rejecting other globals."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("builtins", "bytes"),
        ("builtins", "str"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")
        return super().find_class(module, name)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def archive_member_for_row(row: dict[str, Any]) -> str:
    media = row.get("media")
    if not isinstance(media, list) or len(media) != 1 or not isinstance(media[0], dict):
        raise ValueError("expected exactly one media record")
    raw_path = media[0].get("path")
    if not isinstance(raw_path, str):
        raise ValueError("media path is missing")
    if "\\" in raw_path:
        raise ValueError(f"media path must use POSIX separators: {raw_path}")
    path = PurePosixPath(raw_path)
    prefix = PurePosixPath("raw/physionpp-readout")
    if path.is_absolute() or ".." in path.parts or path.suffix != ".mp4":
        raise ValueError(f"unsafe or unexpected media path: {raw_path}")
    try:
        relative = path.relative_to(prefix)
    except ValueError as error:
        raise ValueError(f"media path is outside Physion++ readout: {raw_path}") from error
    if not relative.name.endswith("_img.mp4"):
        raise ValueError(f"unexpected Physion++ RGB path: {raw_path}")
    stem = relative.name.removesuffix("_img.mp4")
    return (relative.parent / f"{stem}.pkl").as_posix()


def parse_eocd(tail: bytes, archive_size: int) -> tuple[int, int, int, int]:
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or len(tail) - offset < _EOCD.size:
        raise ValueError("ZIP end-of-central-directory record was not found")
    signature, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, comment_len = (
        _EOCD.unpack_from(tail, offset)
    )
    if signature != b"PK\x05\x06" or disk or cd_disk or disk_entries != entries:
        raise ValueError("multi-disk or malformed ZIP archive is unsupported")
    if entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        raise ValueError("ZIP64 archive is unsupported by this pinned audit")
    if offset + _EOCD.size + comment_len != len(tail):
        raise ValueError("ZIP end record comment length is inconsistent")
    absolute_eocd = archive_size - len(tail) + offset
    if cd_offset + cd_size != absolute_eocd:
        raise ValueError("central directory bounds do not match the end record")
    return entries, cd_size, cd_offset, absolute_eocd


def parse_central_directory(data: bytes, expected_entries: int) -> dict[str, ZipMember]:
    members: dict[str, ZipMember] = {}
    position = 0
    while position < len(data):
        if position + _CENTRAL.size > len(data):
            raise ValueError("truncated ZIP central directory")
        values = _CENTRAL.unpack_from(data, position)
        if values[0] != b"PK\x01\x02":
            raise ValueError("unexpected central-directory signature")
        flags, method = values[3], values[4]
        crc, compressed_size, uncompressed_size = values[7:10]
        filename_len, extra_len, comment_len = values[10:13]
        local_offset = values[16]
        end = position + _CENTRAL.size + filename_len + extra_len + comment_len
        if end > len(data):
            raise ValueError("central-directory entry exceeds its bounds")
        if flags & 0x1:
            raise ValueError("encrypted ZIP entries are unsupported")
        name = data[position + _CENTRAL.size : position + _CENTRAL.size + filename_len]
        decoded_name = name.decode("utf-8")
        if decoded_name in members:
            raise ValueError(f"duplicate ZIP member: {decoded_name}")
        members[decoded_name] = ZipMember(
            decoded_name, method, crc, compressed_size, uncompressed_size, local_offset
        )
        position = end
    if position != len(data) or len(members) != expected_entries:
        raise ValueError("ZIP central-directory entry count is inconsistent")
    return members


def decode_member(member: ZipMember, local_header_and_payload: bytes) -> bytes:
    if len(local_header_and_payload) < _LOCAL.size:
        raise ValueError("truncated ZIP local header")
    values = _LOCAL.unpack_from(local_header_and_payload)
    if values[0] != b"PK\x03\x04":
        raise ValueError(f"invalid local header for {member.name}")
    flags, method, local_crc, local_compressed, local_uncompressed = (
        values[2], values[3], values[6], values[7], values[8]
    )
    filename_len, extra_len = values[9:11]
    name_start = _LOCAL.size
    payload_start = name_start + filename_len + extra_len
    local_name = local_header_and_payload[name_start : name_start + filename_len].decode("utf-8")
    if local_name != member.name or method != member.method or flags & 0x1:
        raise ValueError(f"local and central ZIP headers disagree for {member.name}")
    compressed = local_header_and_payload[payload_start:]
    if len(compressed) != member.compressed_size:
        raise ValueError(f"compressed member length mismatch for {member.name}")
    if not (flags & 0x8) and (
        local_crc != member.crc32
        or local_compressed != member.compressed_size
        or local_uncompressed != member.uncompressed_size
    ):
        raise ValueError(f"local ZIP sizes/CRC disagree for {member.name}")
    if method == 8:
        content = zlib.decompress(compressed, -15)
    elif method == 0:
        content = compressed
    else:
        raise ValueError(f"unsupported ZIP compression method {method}")
    if len(content) != member.uncompressed_size or zlib.crc32(content) & 0xFFFFFFFF != member.crc32:
        raise ValueError(f"uncompressed member CRC/size mismatch for {member.name}")
    return content


def load_train_rows(path: Path, expected_sha256: str) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != expected_sha256.lower():
        raise ValueError(f"train manifest SHA-256 mismatch: {digest}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("this audit accepts only a nonempty train-only JSONL")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("train sample IDs are not unique")
    return rows, digest


def load_resume_results(
    path: Path,
    rows: list[dict[str, Any]],
    expected_sha256: str,
) -> dict[str, dict[str, Any]]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(f"resume results SHA-256 mismatch: {actual_sha256}")
    rows_by_id = {row["id"]: row for row in rows}
    resumed: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        sample_id = result.get("sample_id")
        row = rows_by_id.get(sample_id)
        if row is None or sample_id in resumed:
            raise ValueError(
                f"resume results contain an unknown or duplicate sample ID: {sample_id}"
            )
        if (
            result.get("archive_member") != archive_member_for_row(row)
            or result.get("manifest_target") != row.get("target")
            or result.get("source_target") != row.get("source_target")
            or result.get("official_target") != row.get("target")
            or result.get("label_match") is not True
            or result.get("frame_timeline_match") is not True
            or result.get("stimulus_identity_match") is not True
            or not isinstance(result.get("positive_contact_frames"), list)
            or len(result["positive_contact_frames"])
            != result.get("positive_contact_frame_count")
            or result["positive_contact_frames"]
            != sorted(set(result["positive_contact_frames"]))
            or any(
                type(frame_index) is not int
                or frame_index < 0
                or frame_index >= result.get("frame_count", 0)
                for frame_index in result["positive_contact_frames"]
            )
        ):
            raise ValueError(f"resume result does not verify its train row: {sample_id}")
        resumed[sample_id] = result
    return resumed


def _range_get(session: requests.Session, start: int, end: int, etag: str) -> bytes:
    expected = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
    for attempt in range(4):
        try:
            response = session.get(
                ARCHIVE_URL,
                headers={"If-Match": etag, "Range": f"bytes={start}-{end}"},
                timeout=(20, 90),
            )
            if response.status_code != 206:
                raise ValueError(f"range request returned HTTP {response.status_code}")
            if response.headers.get("ETag") != etag:
                raise ValueError("archive ETag changed during range retrieval")
            if response.headers.get("Content-Range") != expected:
                raise ValueError("archive returned an unexpected Content-Range")
            if len(response.content) != end - start + 1:
                raise ValueError("range response has an unexpected byte count")
            return response.content
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def fetch_member(member: ZipMember, etag: str) -> bytes:
    with requests.Session() as session:
        header = _range_get(
            session,
            member.local_offset,
            member.local_offset + _LOCAL.size - 1,
            etag,
        )
        values = _LOCAL.unpack(header)
        filename_len, extra_len = values[9:11]
        prefix_size = _LOCAL.size + filename_len + extra_len
        data_start = member.local_offset + prefix_size
        compressed_end = data_start + member.compressed_size - 1
        payload = _range_get(session, member.local_offset + _LOCAL.size, compressed_end, etag)
        return decode_member(member, header + payload)


def _safe_load_pickle(raw: bytes) -> dict[str, Any]:
    import io

    value = RestrictedUnpickler(io.BytesIO(raw)).load()
    if not isinstance(value, dict) or not isinstance(value.get("static"), dict):
        raise ValueError("official metadata pickle has an unexpected structure")
    return value


def audit_record(row: dict[str, Any], member: ZipMember, etag: str) -> dict[str, Any]:
    raw = fetch_member(member, etag)
    metadata = _safe_load_pickle(raw)
    static = metadata["static"]
    frames = metadata.get("frames")
    if not isinstance(frames, dict) or not frames:
        raise ValueError(f"official metadata has no frame table: {member.name}")
    if "does_target_contact_zone" not in static:
        raise ValueError(f"official metadata has no contact target: {member.name}")
    official = bool(static["does_target_contact_zone"])
    source_target = row.get("source_target")
    manifest_target = row.get("target")
    if not isinstance(source_target, bool) or manifest_target not in {"Yes", "No"}:
        raise ValueError(f"manifest labels are malformed: {row.get('id')}")
    source_match = official == source_target
    target_match = official == (manifest_target == "Yes")
    if not (source_match and target_match):
        raise ValueError(f"official label mismatch for {row.get('id')}")
    frame_positive = sorted(
        int(key)
        for key, frame in frames.items()
        if bool(frame.get("labels", {}).get("target_contacting_zone", False))
    )
    if bool(frame_positive) != official:
        raise ValueError(f"frame-level contact labels disagree with metadata: {row.get('id')}")
    first_contact = static.get("first_target_contact_zone_frame")
    if official and (first_contact is None or int(first_contact) not in frame_positive):
        raise ValueError(f"first-contact frame is inconsistent: {row.get('id')}")
    target_id = int(static["target_id"])
    zone_id = int(static["zone_id"])
    stimulus_name = static.get("stimulus_name")
    if isinstance(stimulus_name, bytes):
        stimulus_name = stimulus_name.decode("utf-8")
    expected_stimulus = PurePosixPath(member.name).parent.name + "_" + PurePosixPath(
        member.name
    ).stem
    if stimulus_name != expected_stimulus:
        raise ValueError(f"pickle stimulus identity mismatch: {member.name}")
    return {
        "sample_id": row["id"],
        "archive_member": member.name,
        "pickle_sha256": sha256_bytes(raw),
        "manifest_target": manifest_target,
        "source_target": source_target,
        "official_target": "Yes" if official else "No",
        "official_target_id": target_id,
        "official_zone_id": zone_id,
        "frame_count": len(frames),
        "positive_contact_frame_count": len(frame_positive),
        "positive_contact_frames": frame_positive,
        "first_contact_frame": int(first_contact) if first_contact is not None else None,
        "label_match": True,
        "frame_timeline_match": True,
        "stimulus_identity_match": True,
    }


def run_audit(
    train_jsonl: Path,
    output_dir: Path,
    expected_sha256: str = DEFAULT_TRAIN_SHA256,
    workers: int = 8,
    resume_results_path: Path | None = None,
    resume_results_sha256: str | None = None,
) -> dict[str, Any]:
    rows, train_sha = load_train_rows(train_jsonl, expected_sha256)
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output_dir}")
    if bool(resume_results_path) != bool(resume_results_sha256):
        raise ValueError("resume results path and expected SHA-256 must be supplied together")
    resumed = (
        load_resume_results(resume_results_path, rows, resume_results_sha256)
        if resume_results_path and resume_results_sha256
        else {}
    )
    output_dir.mkdir(parents=True)

    with requests.Session() as session:
        tail_start = ARCHIVE_SIZE - 65_536
        tail = _range_get(session, tail_start, ARCHIVE_SIZE - 1, ARCHIVE_ETAG)
        entry_count, cd_size, cd_offset, _ = parse_eocd(tail, ARCHIVE_SIZE)
        cd = _range_get(session, cd_offset, cd_offset + cd_size - 1, ARCHIVE_ETAG)
    members = parse_central_directory(cd, entry_count)
    jobs: list[tuple[dict[str, Any], ZipMember]] = []
    for row in rows:
        member_name = archive_member_for_row(row)
        member = members.get(member_name)
        if member is None:
            raise ValueError(f"official archive lacks train metadata: {member_name}")
        jobs.append((row, member))

    results_by_id = dict(resumed)
    pending_jobs = [(row, member) for row, member in jobs if row["id"] not in resumed]
    progress_path = output_dir / "train-label-verification.progress.jsonl"
    with progress_path.open("x", encoding="utf-8", newline="\n") as sink:
        for sample_id in (row["id"] for row in rows if row["id"] in resumed):
            sink.write(json.dumps(resumed[sample_id], sort_keys=True) + "\n")
        sink.flush()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(audit_record, row, member, ARCHIVE_ETAG): row["id"]
                for row, member in pending_jobs
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                results_by_id[result["sample_id"]] = result
                sink.write(json.dumps(result, sort_keys=True) + "\n")
                sink.flush()
                if index % 50 == 0 or index == len(pending_jobs):
                    print(
                        f"verified {len(resumed) + index}/{len(rows)} train labels",
                        flush=True,
                    )
    if len(results_by_id) != len(rows):
        raise ValueError(f"only {len(results_by_id)}/{len(rows)} train rows were verified")
    results = [results_by_id[row["id"]] for row in rows]
    output_jsonl = output_dir / "train-label-verification.jsonl"
    with output_jsonl.open("x", encoding="utf-8", newline="\n") as sink:
        for result in results:
            sink.write(json.dumps(result, sort_keys=True) + "\n")
    report = {
        "status": "pass" if len(results) == len(rows) else "incomplete",
        "archive_url": ARCHIVE_URL,
        "archive_etag": ARCHIVE_ETAG,
        "archive_size_bytes": ARCHIVE_SIZE,
        "archive_central_directory_entries": entry_count,
        "train_manifest_path": str(train_jsonl),
        "train_manifest_sha256": train_sha,
        "audit_script_sha256": sha256_file(Path(__file__)),
        "train_records": len(rows),
        "official_metadata_records_verified": len(results),
        "label_mismatches": sum(not item["label_match"] for item in results),
        "timeline_mismatches": sum(not item["frame_timeline_match"] for item in results),
        "stimulus_identity_mismatches": sum(
            not item["stimulus_identity_match"] for item in results
        ),
        "manifest_labels": {
            label: sum(item["manifest_target"] == label for item in results)
            for label in ("No", "Yes")
        },
        "official_labels": {
            label: sum(item["official_target"] == label for item in results)
            for label in ("No", "Yes")
        },
        "unique_pickle_sha256": len({item["pickle_sha256"] for item in results}),
        "workers": workers,
        "resumed_result_count": len(resumed),
        "resumed_result_source": str(resume_results_path) if resume_results_path else None,
        "resumed_result_source_sha256": resume_results_sha256,
        "sealed_or_validation_data_opened": False,
        "pickle_globals": sorted(
            f"{module}.{name}" for module, name in RestrictedUnpickler._ALLOWED
        ),
        "results_jsonl_sha256": sha256_bytes(output_jsonl.read_bytes()),
    }
    report_path = output_dir / "audit-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", default=DEFAULT_TRAIN_SHA256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume-results", type=Path)
    parser.add_argument("--resume-results-sha256")
    args = parser.parse_args()
    report = run_audit(
        args.train_jsonl,
        args.output_dir,
        args.expected_train_sha256,
        args.workers,
        args.resume_results,
        args.resume_results_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
