"""Fetch and authenticate all ADL activity ZIPs from the pinned Zenodo record.

Downloads are written outside Git. A failed `.part` file is retained and may
be resumed only when the server honors an exact HTTP byte-range request.
Existing complete files are accepted only after full size and publisher-MD5
revalidation; every resumed file is fully rehashed before it is promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

RECORD_SHA256 = "c01bf60460981280c5063ea54c038d41c751bf3056220059c3896cbc5d2b97c6"
DEFAULT_RECORD = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\adl-audio-vibration-source-20261027-attempt-01\record.json"
)
PRIOR_ARCHIVES = {
    "23_take-from-stove.zip": Path(
        r"C:\CodexArtifacts\pretrained-reuse-r2\adl-audio-vibration-archive-20261027-attempt-01\23_take-from-stove.zip"
    ),
    "10_light-off.zip": Path(
        r"C:\CodexArtifacts\pretrained-reuse-r2\adl-audio-vibration-archive-20261027-attempt-02\10_light-off.zip"
    ),
}
CHUNK_SIZE = 1024 * 1024
DISK_RESERVE = 2 * 1024**3


def archive_jobs(files: list[dict]) -> list[dict]:
    archives = [row for row in files if row.get("key", "").endswith(".zip")]
    if len(archives) != 25:
        raise ValueError("pinned source record must contain exactly 25 activity ZIPs")
    numbers = set()
    for row in archives:
        key = row["key"]
        prefix, separator, rest = key.partition("_")
        if (
            not separator
            or len(prefix) != 2
            or not prefix.isdecimal()
            or not rest.endswith(".zip")
            or int(prefix) in numbers
            or not isinstance(row.get("size"), int)
            or row["size"] <= 0
            or not row.get("checksum", "").startswith("md5:")
            or len(row["checksum"]) != 36
            or urlparse(row.get("links", {}).get("self", "")).scheme != "https"
        ):
            raise ValueError(f"invalid activity archive metadata: {key}")
        try:
            int(row["checksum"][4:], 16)
        except ValueError as exc:
            raise ValueError(f"invalid publisher MD5: {key}") from exc
        numbers.add(int(prefix))
    if numbers != set(range(25)):
        raise ValueError("archive names must cover activity indices 00 through 24")
    return sorted(archives, key=lambda row: (row["size"], row["key"]))


def hash_archive(path: Path, expected_size: int, expected_md5: str) -> dict:
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"archive size mismatch: {path}")
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            md5.update(chunk)
            sha256.update(chunk)
    if md5.hexdigest() != expected_md5.removeprefix("md5:"):
        raise ValueError(f"publisher MD5 mismatch: {path}")
    return {"bytes": expected_size, "publisher_md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def download_one(session: requests.Session, row: dict, output_dir: Path) -> dict:
    key = row["key"]
    target = output_dir / key
    expected_md5 = row["checksum"]
    expected_size = row["size"]
    if target.exists():
        return {"key": key, **hash_archive(target, expected_size, expected_md5), "reused": True}
    part = target.with_suffix(target.suffix + ".part")
    start = time.perf_counter()
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    offset = part.stat().st_size if part.exists() else 0
    received = offset
    transferred = 0
    if offset > expected_size:
        raise ValueError(f"partial file exceeds pinned source size: {part}")
    if offset:
        with part.open("rb") as stream:
            while chunk := stream.read(CHUNK_SIZE):
                md5.update(chunk)
                sha256.update(chunk)
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = session.get(
        row["links"]["self"], headers=headers, stream=True, timeout=(20, 180)
    )
    response.raise_for_status()
    if offset:
        expected_range = f"bytes {offset}-{expected_size - 1}/{expected_size}"
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != expected_range
        ):
            raise ValueError(
                f"server did not honor exact resume range for {key}; partial preserved"
            )
        expected_response_size = expected_size - received
    else:
        if response.status_code != 200:
            raise ValueError(
                f"expected HTTP 200 for full archive, got {response.status_code}: {key}"
            )
        expected_response_size = expected_size
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != expected_response_size:
        raise ValueError(f"Content-Length mismatch for {key}")
    with part.open("ab" if received else "xb") as stream:
        for chunk in response.iter_content(CHUNK_SIZE):
            if not chunk:
                continue
            received += len(chunk)
            if received > expected_size:
                raise ValueError(f"received more bytes than pinned source size: {key}")
            stream.write(chunk)
            md5.update(chunk)
            sha256.update(chunk)
            transferred += len(chunk)
        stream.flush()
    elapsed = time.perf_counter() - start
    if received != expected_size:
        raise ValueError(f"received {received} of {expected_size} bytes for {key}")
    if md5.hexdigest() != expected_md5.removeprefix("md5:"):
        raise ValueError(f"publisher MD5 mismatch for {key}")
    part.replace(target)
    return {
        "key": key,
        "bytes": received,
        "transferred_bytes": transferred,
        "publisher_md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "seconds": elapsed,
        "mib_per_second": transferred / 2**20 / elapsed,
        "reused": False,
        "resumed_bytes": offset,
    }


def verify_receipt_downloads(output_dir: Path, downloads: list[dict], by_key: dict) -> set[str]:
    completed = {item.get("key") for item in downloads}
    if len(completed) != len(downloads):
        raise ValueError("receipt contains duplicate archive entries")
    for item in downloads:
        key = item.get("key")
        if key not in by_key or key in PRIOR_ARCHIVES:
            raise ValueError(f"receipt contains unexpected archive entry: {key}")
        row = by_key[key]
        actual = hash_archive(output_dir / key, row["size"], row["checksum"])
        if (
            item.get("bytes") != actual["bytes"]
            or item.get("publisher_md5") != actual["publisher_md5"]
            or item.get("sha256") != actual["sha256"]
        ):
            raise ValueError(f"completed archive does not match pinned receipt: {key}")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="download at most this many new archives")
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()
    record_raw = args.record.read_bytes()
    if hashlib.sha256(record_raw).hexdigest() != RECORD_SHA256:
        raise ValueError("pinned Zenodo record hash mismatch")
    jobs = archive_jobs(json.loads(record_raw)["files"])
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.output_dir.exists():
        if not (args.output_dir / "download-plan.json").is_file():
            raise FileExistsError("output directory exists without this run's pinned plan")
    else:
        args.output_dir.mkdir(parents=True)
    plan = {
        "source_record_sha256": RECORD_SHA256,
        "source_doi": "10.5281/zenodo.7937591",
        "archives": [
            {"key": row["key"], "size": row["size"], "md5": row["checksum"]} for row in jobs
        ],
        "prior_verified_archives": {key: str(path) for key, path in PRIOR_ARCHIVES.items()},
        "cloud_compute": False,
        "retry_policy": "none; failed partials are retained and never resumed implicitly",
    }
    plan_path = args.output_dir / "download-plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("existing plan does not match pinned source record")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    total_bytes = sum(row["size"] for row in jobs if row["key"] not in PRIOR_ARCHIVES)
    verified_prior = {}
    by_key = {row["key"]: row for row in jobs}
    for key, path in PRIOR_ARCHIVES.items():
        if key not in by_key:
            raise ValueError(f"unexpected existing archive {key}")
        verified_prior[key] = {
            "key": key,
            **hash_archive(path, by_key[key]["size"], by_key[key]["checksum"]),
            "reused": True,
        }
    pending = [row for row in jobs if row["key"] not in PRIOR_ARCHIVES]
    if args.limit is not None:
        pending = pending[: args.limit]

    receipt_path = args.output_dir / "download-receipt.json"
    prior_receipt = {
        "source_record_sha256": RECORD_SHA256,
        "prior_verified_archives": verified_prior,
        "downloads": [],
    }
    if receipt_path.exists():
        prior_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            prior_receipt.get("source_record_sha256") != RECORD_SHA256
            or prior_receipt.get("prior_verified_archives") != verified_prior
        ):
            raise ValueError("existing receipt does not match prior verified archive set")
    completed = verify_receipt_downloads(
        args.output_dir, prior_receipt["downloads"], by_key
    )
    todo = [row for row in pending if row["key"] not in completed]
    required_free = sum(row["size"] for row in todo) + DISK_RESERVE
    if shutil.disk_usage(args.output_dir).free < required_free:
        raise OSError(f"need {required_free} free bytes including fixed 2 GiB reserve")
    if todo:
        with requests.Session() as session:
            for row in todo:
                measured = download_one(session, row, args.output_dir)
                prior_receipt["downloads"].append(measured)
                receipt_path.write_text(
                    json.dumps(prior_receipt, indent=2) + "\n", encoding="utf-8"
                )
                print(json.dumps(measured, sort_keys=True), flush=True)
    measured_rows = [item for item in prior_receipt["downloads"] if "seconds" in item]
    sample = measured_rows[:3]
    sample_bytes = sum(item.get("transferred_bytes", item["bytes"]) for item in sample)
    sample_seconds = sum(item["seconds"] for item in sample)
    rest_bytes = total_bytes - sum(item["bytes"] for item in prior_receipt["downloads"])
    mean_bps = sample_bytes / sample_seconds if sample_seconds else None
    projection = rest_bytes / mean_bps if mean_bps else None
    summary = {
        "status": "PARTIAL"
        if len(prior_receipt["downloads"]) < len(jobs) - len(PRIOR_ARCHIVES)
        else "ALL_ARCHIVES_MD5_VERIFIED",
        "new_archives_verified": len(prior_receipt["downloads"]),
        "new_archives_expected": len(jobs) - len(PRIOR_ARCHIVES),
        "verified_new_bytes": sum(item["bytes"] for item in prior_receipt["downloads"]),
        "remaining_new_bytes": max(0, rest_bytes),
        "first_three_mean_mib_per_second": mean_bps / 2**20 if mean_bps else None,
        "projected_remaining_seconds_from_first_three": projection,
        "publisher_md5_verified": len(prior_receipt["downloads"]) + len(verified_prior),
    }
    prior_receipt["summary"] = summary
    receipt_path.write_text(json.dumps(prior_receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
