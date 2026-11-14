"""Range-acquire only the frozen FSD50K development manifest, never official eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.build_fsd50k_annotation_decision_manifest import (
    DATASET_REVISION,
    canonical_sha256,
    sha256_file,
    summarize_split,
)
from scripts.fetch_fsd50k_training_smoke import central_entries
from scripts.fetch_starss23_smoke import HttpRangeReader
from scripts.run_starss23_ipv4 import IPv4HTTPSHandler
from scripts.validate_fsd50k_annotation_acquisition import sha256_and_crc32

ARCHIVE_SIZES = (3_221_225_472,) * 5 + (2_306_663_327,)
ARCHIVE_NAMES = tuple(f"FSD50K.dev_audio.z{i:02d}" for i in range(1, 6)) + (
    "FSD50K.dev_audio.zip",
)
EXPECTED_DIRECTORY_END = (
    b"PK\x05\x06",
    5,
    5,
    40_967,
    40_967,
    3_962_717,
    2_302_700_512,
    0,
)
MAX_MEMBER_BYTES = 64 * 1024 * 1024
DISK_RESERVE_BYTES = 2 * 1024**3


class SharedBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.requested = 0

    def reserve(self, count: int) -> None:
        if count < 0 or self.requested + count > self.limit:
            raise ValueError("total FSD50K range-transfer budget exceeded")
        self.requested += count


class BudgetedRangeReader(HttpRangeReader):
    def __init__(self, url: str, size: int, shared_budget: SharedBudget):
        super().__init__(url, size, budget=size, retries=5)
        self.shared_budget = shared_budget

    def read(self, size: int = -1) -> bytes:
        count = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if count == 0:
            return b""
        if count < 0 or self.transferred + count > self.budget:
            raise ValueError("range transfer budget exceeded")
        start = self.position
        end = start + count - 1
        expected_range = f"bytes {start}-{end}/{self.size}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self.shared_budget.reserve(count)
            request = urllib.request.Request(
                self.url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    actual_range = response.headers.get("Content-Range")
                    if response.status != 206 or actual_range != expected_range:
                        raise ValueError(
                            "server did not return the exact requested byte range: "
                            f"status={response.status}, content_range={actual_range!r}, "
                            f"expected={expected_range!r}"
                        )
                    data = response.read(count + 1)
                self.transferred += len(data)
                if len(data) == count:
                    self.position += count
                    return data
                if len(data) > count:
                    raise ValueError(
                        f"range response exceeded requested length: got={len(data)}, "
                        f"expected={count}, start={start}"
                    )
                last_error = ValueError(
                    f"short range response: got={len(data)}, expected={count}, "
                    f"start={start}, end={end}, attempt={attempt + 1}"
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(0.5 * (2**attempt), 8.0))
        raise ValueError(
            f"range request failed after {self.retries + 1} attempts at "
            f"bytes={start}-{end}: {last_error!r}"
        ) from last_error


def load_frozen_manifest(
    preflight_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    report = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "PREFLIGHT_PASS"
        or report.get("dataset_revision") != DATASET_REVISION
    ):
        raise ValueError("FSD50K preflight is not a passing, pinned manifest")
    if report.get("official_eval_labels_read") is not False:
        raise ValueError("preflight says official eval labels were read")
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "calibration", "validation"):
        path = preflight_dir / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if canonical_sha256(rows) != report["splits"][split]["ordered_rows_sha256"]:
            raise ValueError(f"frozen FSD50K {split} manifest hash mismatch")
        if summarize_split(rows) != report["splits"][split]:
            raise ValueError(f"frozen FSD50K {split} summary mismatch")
        splits[split] = rows
    return report, splits


def planned_budget(
    splits: dict[str, list[dict[str, Any]]], index: dict[str, Any], central_directory_bytes: int
) -> tuple[int, int]:
    members = []
    for rows in splits.values():
        for row in rows:
            member_name = f"FSD50K.dev_audio/{row['sample_id']}.wav"
            try:
                info = index[member_name]
            except KeyError as error:
                raise ValueError(
                    f"manifest member missing from pinned ZIP: {member_name}"
                ) from error
            if info.is_dir() or info.flag_bits & 1 or info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"unsupported or oversized FSD50K member: {member_name}")
            members.append(info)
    output_bytes = sum(info.file_size for info in members)
    transfer_budget = sum(
        info.compress_size + len(info.filename.encode("utf-8")) + 4096 + 60 for info in members
    ) + central_directory_bytes + 22
    return output_bytes, transfer_budget


def require_free_space(path: Path, required_bytes: int) -> None:
    if required_bytes > shutil.disk_usage(path).free:
        raise ValueError("insufficient disk space for FSD50K media plus 2 GiB reserve")


def _read_parts(readers: list[BudgetedRangeReader], disk: int, offset: int, count: int) -> bytes:
    chunks = []
    while count:
        if not 0 <= disk < len(readers):
            raise ValueError("invalid multi-disk ZIP volume")
        reader = readers[disk]
        length = min(count, reader.size - offset)
        if length <= 0:
            raise ValueError("invalid multi-disk ZIP offset")
        reader.seek(offset)
        chunks.append(reader.read(length))
        count -= length
        disk += 1
        offset = 0
    return b"".join(chunks)


def _extract_member(
    readers: list[BudgetedRangeReader], info: Any, member_name: str, output: Path
) -> tuple[int, str]:
    header = _read_parts(readers, info.volume, info.header_offset, 30)
    fields = struct.unpack("<4s5H3I2H", header)
    if fields[0] != b"PK\x03\x04" or fields[2] & 1 or fields[3] != info.compress_type:
        raise ValueError(f"invalid or encrypted local FSD50K member header: {member_name}")
    name_length, extra_length = fields[-2], fields[-1]
    raw = _read_parts(
        readers,
        info.volume,
        info.header_offset,
        30 + name_length + extra_length + info.compress_size,
    )
    local_name = raw[30 : 30 + name_length].decode("utf-8")
    if local_name != member_name:
        raise ValueError(f"local/central FSD50K member name mismatch: {member_name}")
    compressed = raw[30 + name_length + extra_length :]
    if info.compress_type == 8:
        data = zlib.decompress(compressed, -15)
    elif info.compress_type == 0:
        data = compressed
    else:
        raise ValueError(f"unsupported FSD50K ZIP compression: {member_name}")
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError(f"FSD50K archive size/CRC mismatch: {member_name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return len(data), hashlib.sha256(data).hexdigest()


def validate_failed_partial(
    preflight_dir: Path, partial_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the known attempt-03 prefix before copying it to a fresh run."""
    report, splits = load_frozen_manifest(preflight_dir)
    prior_path = partial_dir / "result.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (
        prior.get("status") != "FAIL"
        or prior.get("error") != "ValueError('range length mismatch')"
        or prior.get("runner_sha256")
        != "784687424d0720626e89a284a4a9546e585b75e977d5321664d5c022dbe7a71d"
    ):
        raise ValueError("resume source is not the preserved, identified attempt-03 failure")
    if (
        prior.get("dataset_revision") != DATASET_REVISION
        or prior.get("preflight_manifest_sha256") != report["manifest_sha256"]
        or prior.get("archive_directory_sha256")
        != "a52c8e77acf8c01e5108a3872461a65f2a583b2ddc3e0bb6554e51dc7ae12c5f"
        or prior.get("official_eval_labels_read") is not False
        or prior.get("official_eval_media_acquired") is not False
    ):
        raise ValueError("resume source does not match the frozen safe FSD50K corpus")
    receipt_path = partial_dir / "files.jsonl"
    if prior.get("files_jsonl_sha256") != sha256_file(receipt_path):
        raise ValueError("partial attempt-03 receipt hash mismatch")
    records = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
    ordered = [row for split in ("train", "calibration", "validation") for row in splits[split]]
    record_ids = [record.get("sample_id") for record in records]
    expected_prefix_ids = [row["sample_id"] for row in ordered[: len(records)]]
    if (
        prior.get("expected_files") != len(ordered)
        or
        len(records) != prior.get("downloaded_files")
        or record_ids != expected_prefix_ids
        or len(records) >= len(ordered)
    ):
        raise ValueError("partial attempt-03 receipt count/order is invalid")
    expected = {row["sample_id"]: row for row in ordered[: len(records)]}
    for record in records:
        sample_id = record.get("sample_id")
        row = expected.get(sample_id)
        if row is None or any(
            record.get(field) != row.get(field)
            for field in ("source_id", "group_id", "license", "split")
        ):
            raise ValueError("partial attempt-03 receipt differs from frozen manifest")
        media_path = partial_dir / row["split"] / f"{sample_id}.wav"
        size, digest, crc = sha256_and_crc32(media_path)
        if (
            size != record.get("bytes")
            or digest != record.get("sha256")
            or crc != record.get("zip_crc32")
        ):
            raise ValueError(f"partial attempt-03 media failed integrity check: {sample_id}")
    expected_paths = {
        partial_dir / row["split"] / f"{row['sample_id']}.wav"
        for row in ordered[: len(records)]
    }
    if set(partial_dir.rglob("*.wav")) != expected_paths:
        raise ValueError("partial attempt-03 contains unreceipted or missing WAV files")
    return prior, records


def acquire(
    preflight_dir: Path,
    output_dir: Path,
    artifact_root: Path,
    *,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite FSD50K acquisition: {output_dir}")
    report, splits = load_frozen_manifest(preflight_dir)
    expected_ids = {row["sample_id"] for rows in splits.values() for row in rows}
    if len(expected_ids) != sum(len(rows) for rows in splits.values()):
        raise ValueError("duplicate IDs across frozen FSD50K splits")
    prior_result = None
    prior_records: list[dict[str, Any]] = []
    if resume_from is not None:
        prior_result, prior_records = validate_failed_partial(preflight_dir, resume_from)
    output_dir.mkdir(parents=True, exist_ok=False)
    for record in prior_records:
        split_dir = output_dir / record["split"]
        split_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            resume_from / record["split"] / f"{record['sample_id']}.wav",
            split_dir / f"{record['sample_id']}.wav",
        )

    shared = SharedBudget(limit=sum(ARCHIVE_SIZES))
    readers = [
        BudgetedRangeReader(
            f"https://zenodo.org/api/records/4060432/files/{name}/content", size, shared
        )
        for name, size in zip(ARCHIVE_NAMES, ARCHIVE_SIZES, strict=True)
    ]
    result: dict[str, Any] = {
        "status": "RUNNING",
        "dataset_revision": DATASET_REVISION,
        "preflight_manifest_sha256": report["manifest_sha256"],
        "expected_files": len(expected_ids),
        "downloaded_files": 0,
        "resumed_files": len(prior_records),
        "resumed_from_result_sha256": sha256_file(resume_from / "result.json")
        if resume_from is not None
        else None,
        "resumed_range_bytes": prior_result.get("range_bytes", 0)
        if prior_result is not None
        else 0,
        "official_eval_labels_read": False,
        "official_eval_media_acquired": False,
        "archive_directory_sha256": None,
        "output_bytes": 0,
        "range_bytes": 0,
        "runner_sha256": sha256_file(Path(__file__)),
    }
    files_path = output_dir / "files.jsonl"
    progress_path = output_dir / "progress.json"
    urlopen = urllib.request.urlopen
    opener = urllib.request.build_opener(IPv4HTTPSHandler())
    try:
        with patch("urllib.request.urlopen", opener.open):
            zip_reader = readers[-1]
            zip_reader.seek(ARCHIVE_SIZES[-1] - 22)
            footer = struct.unpack("<4s4H2IH", zip_reader.read(22))
            if footer != EXPECTED_DIRECTORY_END:
                raise ValueError("FSD50K archive central-directory footer changed")
            directory = _read_parts(
                readers,
                footer[6] // ARCHIVE_SIZES[-1] if footer[6] > ARCHIVE_SIZES[-1] else 5,
                footer[6] % ARCHIVE_SIZES[-1] if footer[6] > ARCHIVE_SIZES[-1] else footer[6],
                footer[5],
            )
            result["archive_directory_sha256"] = hashlib.sha256(directory).hexdigest()
            if prior_result is not None and result["archive_directory_sha256"] != prior_result.get(
                "archive_directory_sha256"
            ):
                raise ValueError("archive directory hash changed since attempt-03")
            index = central_entries(directory, footer[4])
            output_bytes, transfer_budget = planned_budget(splits, index, footer[5])
            if transfer_budget > sum(ARCHIVE_SIZES):
                raise ValueError("planned FSD50K transfer exceeds source archive size")
            require_free_space(output_dir, output_bytes + DISK_RESERVE_BYTES)
            resumed_range_bytes = int(result["resumed_range_bytes"])
            if resumed_range_bytes > transfer_budget:
                raise ValueError("partial attempt transfer already exceeds the frozen budget")
            shared.limit = transfer_budget - resumed_range_bytes
            result["planned_output_bytes"] = output_bytes
            result["range_transfer_budget_bytes"] = transfer_budget

            count = len(prior_records)
            resume_ids = {record["sample_id"] for record in prior_records}
            with files_path.open("x", encoding="utf-8", newline="\n") as receipt:
                for record in prior_records:
                    receipt.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                receipt.flush()
                os.fsync(receipt.fileno())
            with files_path.open("a", encoding="utf-8", newline="\n") as receipt:
                for split_name, rows in splits.items():
                    for row in rows:
                        if row["sample_id"] in resume_ids:
                            continue
                        member_name = f"FSD50K.dev_audio/{row['sample_id']}.wav"
                        info = index[member_name]
                        destination = output_dir / split_name / f"{row['sample_id']}.wav"
                        size, digest = _extract_member(
                            readers, info, member_name, destination
                        )
                        record = {
                            "sample_id": row["sample_id"],
                            "source_id": row["source_id"],
                            "group_id": row["group_id"],
                            "split": split_name,
                            "license": row["license"],
                            "creator": row["creator"],
                            "title": row["title"],
                            "source_url": row["source_url"],
                            "bytes": size,
                            "sha256": digest,
                            "zip_crc32": info.CRC,
                            "archive_volume": info.volume,
                        }
                        receipt.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        count += 1
                        if count % 50 == 0:
                            receipt.flush()
                            os.fsync(receipt.fileno())
                            result.update(
                                downloaded_files=count,
                                output_bytes=sum(
                                    r.stat().st_size for r in output_dir.rglob("*.wav")
                                ),
                                range_bytes=resumed_range_bytes + shared.requested,
                            )
                            progress_path.write_text(
                                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8",
                            )
                receipt.flush()
                os.fsync(receipt.fileno())
            result.update(
                status="PASS",
                downloaded_files=count,
                output_bytes=sum(r.stat().st_size for r in output_dir.rglob("*.wav")),
                range_bytes=resumed_range_bytes + shared.requested,
                range_budget_bytes=shared.limit,
                archive_bytes_by_volume={
                    name: reader.transferred
                    for name, reader in zip(ARCHIVE_NAMES, readers, strict=True)
                },
            )
            if count != len(expected_ids) or result["output_bytes"] != output_bytes:
                raise ValueError("FSD50K acquisition final file count/size mismatch")
    except BaseException as error:
        result.update(
            status="FAIL",
            error=repr(error),
            downloaded_files=sum(1 for _ in files_path.open(encoding="utf-8"))
            if files_path.exists()
            else 0,
            output_bytes=sum(r.stat().st_size for r in output_dir.rglob("*.wav")),
            range_bytes=int(result.get("resumed_range_bytes", 0)) + shared.requested,
        )
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise
    finally:
        if files_path.exists():
            result["files_jsonl_sha256"] = sha256_file(files_path)
        result["runner_sha256"] = sha256_file(Path(__file__))
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        try:
            urllib.request.urlopen = urlopen
        except Exception:
            pass
    return result


def main() -> None:
    default_root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=default_root / "fsd50k-annotation-presence-v1-preflight-20261115-attempt-01",
    )
    parser.add_argument("--artifact-root", type=Path, default=default_root)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "fsd50k-annotation-presence-v1-media-20261115-attempt-01",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            acquire(
                args.preflight,
                args.output,
                args.artifact_root,
                resume_from=args.resume_from,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
