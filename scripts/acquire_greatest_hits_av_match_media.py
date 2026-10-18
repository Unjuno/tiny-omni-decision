"""Acquire only the frozen TRAIN-only audio/video match assets with strict bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.greatest_hits_source import CurlRangeSource

ARCHIVE_URL = "https://web.eecs.umich.edu/~ahowens/vis/vis-data-256.zip"
ARCHIVE_SIZE = 21_142_601_241
ARCHIVE_ETAG = '"4ec327e19-5cc7fa3687c80"'
ARCHIVE_INDEX_SHA256 = "98673a0101020606537d2850f825ebe08219c496928002f2cf09dcaa83b9824a"
TRAIN_JSONL_SHA256 = "5684d3a07186f8b6db1951ddb1131b1bb90cd74de57307217fe39c05b9185859"
CALIBRATION_JSONL_SHA256 = "4fea517675f3f85d948c8e7a9733bec84b787a0ada680a387589aa21fbe84ee9"
VALIDATION_JSONL_SHA256 = "60fd0ff39a7d164e4082688749def3c224fad613881b932b75e587df9208ce83"
MAX_MEMBER_BYTES = 40 * 1024 * 1024
MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_COMPRESSED_BYTES = 1 * 1024 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_bytes(entry: dict[str, Any], read: Any) -> bytes:
    name = str(entry["name"])
    if (
        not name.startswith("vis-data-256/")
        or ".." in PurePosixPath(name).parts
        or "\\" in name
        or name.count("/") != 1
        or not 0 < int(entry["bytes"]) <= MAX_MEMBER_BYTES
        or not 0 < int(entry["compressed"]) <= MAX_COMPRESSED_BYTES
        or int(entry["offset"]) < 0
        or int(entry["compression"]) not in (0, 8)
    ):
        raise ValueError("unsafe or oversized archive member")
    header = read(int(entry["offset"]), 30)
    if len(header) != 30:
        raise ValueError("truncated local ZIP header")
    import struct

    magic, _, flags, method, _, _, _, _, _, name_length, extra_length = struct.unpack(
        "<4s5H3I2H", header
    )
    if (
        magic != b"PK\x03\x04"
        or flags & 1
        or method != int(entry["compression"])
        or not 0 < name_length <= 512
        or extra_length > 4096
    ):
        raise ValueError("invalid/encrypted local ZIP header")
    remainder_size = name_length + extra_length + int(entry["compressed"])
    tail = read(int(entry["offset"]) + 30, remainder_size)
    if len(tail) != remainder_size:
        raise ValueError("truncated local ZIP body")
    filename_encoding = "utf-8" if flags & 2048 else "cp437"
    if tail[:name_length].decode(filename_encoding) != name:
        raise ValueError("local ZIP name differs from pinned directory index")
    compressed = tail[name_length + extra_length :]
    if method == 8:
        decoder = zlib.decompressobj(-15)
        payload = decoder.decompress(compressed, MAX_MEMBER_BYTES + 1)
        if (
            not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
            or len(payload) > MAX_MEMBER_BYTES
        ):
            raise ValueError("invalid or oversized deflate payload")
    else:
        payload = compressed
    if len(payload) != int(entry["bytes"]):
        raise ValueError("expanded member length differs")
    if f"{zlib.crc32(payload):08x}" != str(entry["crc32"]).lower():
        raise ValueError("expanded member CRC differs")
    return payload


def _asset_plan(preflight_dir: Path, archive_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {
        "train": (TRAIN_JSONL_SHA256, 252),
        "calibration": (CALIBRATION_JSONL_SHA256, 76),
        "validation": (VALIDATION_JSONL_SHA256, 112),
    }
    members = {str(row["name"]): row for row in archive_index}
    plan_by_name: dict[str, dict[str, Any]] = {}
    group_splits: dict[str, str] = {}
    for split, (digest, expected_rows) in expected.items():
        path = preflight_dir / f"{split}.jsonl"
        if _sha256(path) != digest:
            raise ValueError(f"frozen {split} corpus hash mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != expected_rows:
            raise ValueError(f"frozen {split} row count mismatch")
        for row in rows:
            for field, modality, suffix in (
                ("audio_path", "audio", "_denoised.wav"),
                ("video_path", "video", "_denoised.mp4"),
            ):
                filename = str(row[field])
                if not filename.endswith(suffix) or Path(filename).name != filename:
                    raise ValueError("unsafe frozen media filename")
                source_group = row[f"{modality}_source_group"]
                old_split = group_splits.setdefault(source_group, split)
                if old_split != split:
                    raise ValueError("source group appears in multiple frozen splits")
                member_name = f"vis-data-256/{filename}"
                if member_name not in members:
                    raise ValueError(f"pinned archive index lacks {member_name}")
                current = plan_by_name.setdefault(
                    member_name,
                    {
                        "name": member_name,
                        "split": split,
                        "modality": modality,
                        "source_group": source_group,
                        "entry": members[member_name],
                    },
                )
                if current["split"] != split or current["source_group"] != source_group:
                    raise ValueError("media identity crosses split or source-group boundary")
    plan = sorted(plan_by_name.values(), key=lambda item: item["name"])
    total_bytes = sum(int(item["entry"]["bytes"]) for item in plan)
    total_compressed = sum(int(item["entry"]["compressed"]) for item in plan)
    if len(plan) != 440 or total_bytes != 1_910_230_808 or total_compressed != 933_852_565:
        raise ValueError("pinned selected media count or aggregate size differs")
    return plan


def acquire(plan: list[dict[str, Any]], output_dir: Path, workers: int = 2) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "media").mkdir()
    (output_dir / "receipts").mkdir()
    started = time.monotonic()
    lock = threading.Lock()
    completed: dict[str, dict[str, Any]] = {}
    stop = threading.Event()

    def fetch(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = item["name"]
        if stop.is_set():
            return name, {"status": "NOT_DISPATCHED"}
        key = Path(name).name
        destination = output_dir / "media" / item["split"] / item["modality"] / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        range_dir = output_dir / "receipts" / key
        source = None
        try:
            source = CurlRangeSource(
                ARCHIVE_URL,
                ARCHIVE_SIZE,
                ARCHIVE_ETAG,
                range_dir,
                request_seconds=90,
            )
            payload = _safe_member_bytes(item["entry"], source.read)
            digest = hashlib.sha256(payload).hexdigest()
            destination.write_bytes(payload)
            for temporary in range_dir.glob("range-*.bin"):
                temporary.unlink()
            for temporary in range_dir.glob("range-*.headers"):
                temporary.unlink()
            return name, {
                "status": "PASS",
                "split": item["split"],
                "modality": item["modality"],
                "source_group": item["source_group"],
                "file": str(destination.relative_to(output_dir)),
                "bytes": len(payload),
                "crc32": f"{zlib.crc32(payload):08x}",
                "sha256": digest,
                "requested_bytes": source.requested,
                "archive_etag": ARCHIVE_ETAG,
            }
        except Exception as error:
            stop.set()
            return name, {
                "status": "FAILED",
                "split": item["split"],
                "error": repr(error),
                "requested_bytes": source.requested if source else 0,
            }

    result: dict[str, Any] = {
        "status": "RUNNING",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "planned_files": len(plan),
        "workers": workers,
        "archive_url": ARCHIVE_URL,
        "archive_size": ARCHIVE_SIZE,
        "archive_etag": ARCHIVE_ETAG,
        "archive_index_sha256": ARCHIVE_INDEX_SHA256,
        "raw_test_or_quarantine_media_opened": False,
        "sealed_audit_opened": False,
        "weights_loaded": False,
        "optimizer_updates": 0,
    }
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(fetch, item) for item in plan]
            for future in as_completed(pending):
                name, receipt = future.result()
                with lock:
                    completed[name] = receipt
                    progress = {
                        "completed": len(completed),
                        "passed": sum(row["status"] == "PASS" for row in completed.values()),
                        "failed": [
                            key for key, row in completed.items() if row["status"] == "FAILED"
                        ],
                        "last_member": name,
                        "seconds": time.monotonic() - started,
                    }
                    (output_dir / "progress.json").write_text(
                        json.dumps(progress, indent=2), encoding="utf-8"
                    )
        if len(completed) != len(plan) or any(
            item["status"] != "PASS" for item in completed.values()
        ):
            raise RuntimeError("media acquisition incomplete; partial receipts preserved")
        owners: dict[tuple[str, str], str] = {}
        for receipt in completed.values():
            key = (receipt["modality"], receipt["sha256"])
            previous = owners.setdefault(key, receipt["split"])
            if previous != receipt["split"]:
                raise ValueError("identical raw media hash crosses splits")
        result["status"] = "PASS_MEDIA_ACQUISITION_RAW_HASH_DISJOINT"
    except Exception as error:
        result.update(status="FAILED", error=repr(error))
        raise
    finally:
        (output_dir / "members.json").write_text(
            json.dumps(dict(sorted(completed.items())), indent=2), encoding="utf-8"
        )
        result.update(
            finished_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            completed=len(completed),
            passed=sum(item["status"] == "PASS" for item in completed.values()),
            requested_bytes=sum(int(item.get("requested_bytes", 0)) for item in completed.values()),
            members_manifest_sha256=_sha256(output_dir / "members.json"),
            raw_media_sha256_by_split={
                split: {
                    modality: hashlib.sha256(
                        "\n".join(
                            sorted(
                                item["sha256"]
                                for item in completed.values()
                                if item.get("split") == split and item.get("modality") == modality
                            )
                        ).encode()
                    ).hexdigest()
                    for modality in ("audio", "video")
                }
                for split in ("train", "calibration", "validation")
            },
        )
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.archive_index) != ARCHIVE_INDEX_SHA256:
        raise ValueError("pinned archive directory index hash mismatch")
    archive_index = json.loads(args.archive_index.read_text(encoding="utf-8"))
    plan = _asset_plan(args.preflight_dir, archive_index)
    result = acquire(plan, args.output_dir)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "completed", "passed", "elapsed_seconds")}
        )
    )


if __name__ == "__main__":
    main()
