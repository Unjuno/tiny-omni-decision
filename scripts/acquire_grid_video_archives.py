"""Acquire pinned GRID video ZIPs for cross-split media duplicate audits."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 * 1024
PROGRESS_INTERVAL = 8 * 1024 * 1024
MIN_FREE_RESERVE = 2 * 1024**3
_progress_lock = threading.Lock()


def file_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def load_inventory(summary_path: Path, expected_sha256: str) -> tuple[str, list[dict]]:
    _, actual_sha256 = file_hashes(summary_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("GRID source summary SHA-256 mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS_DIRECT_SOURCE_SPLIT_BUILT":
        raise ValueError("source summary is not the verified direct-source split")
    receipts = summary.get("source_receipts")
    if not isinstance(receipts, dict) or len(receipts) != 32:
        raise ValueError("expected exactly 32 speaker video receipts")
    rows = []
    for speaker, receipt in sorted(receipts.items()):
        url = receipt.get("video_url", "")
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "zenodo.org"
            or parsed.path != f"/api/records/3625687/files/{speaker}.zip/content"
        ):
            raise ValueError(f"unexpected official video URL for {speaker}")
        checksum = str(receipt.get("video_publisher_md5", ""))
        if not checksum.startswith("md5:") or len(checksum) != 36:
            raise ValueError(f"invalid publisher MD5 for {speaker}")
        size = int(receipt.get("video_archive_bytes", 0))
        if size <= 0:
            raise ValueError(f"invalid archive size for {speaker}")
        rows.append(
            {
                "speaker": speaker,
                "url": url,
                "expected_bytes": size,
                "publisher_md5": checksum.removeprefix("md5:").lower(),
            }
        )
    return actual_sha256, rows


def parse_existing_archive(value: str) -> tuple[str, Path]:
    speaker, separator, raw_path = value.partition("=")
    if not separator or not speaker or not raw_path:
        raise argparse.ArgumentTypeError("expected SPEAKER=PATH")
    return speaker, Path(raw_path).resolve()


def append_jsonl(path: Path, row: dict) -> None:
    encoded = json.dumps(row, sort_keys=True) + "\n"
    with _progress_lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


def validate_existing_archive(item: dict, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != item["expected_bytes"]:
        raise ValueError(f"existing {item['speaker']} archive size mismatch")
    md5, sha256 = file_hashes(path)
    if md5 != item["publisher_md5"]:
        raise ValueError(f"existing {item['speaker']} publisher MD5 mismatch")
    return {
        **item,
        "path": str(path),
        "sha256": sha256,
        "verified_by": "existing_publisher_md5_and_sha256",
    }


def download_one(item: dict, output_dir: Path, progress_path: Path) -> dict:
    speaker = item["speaker"]
    final_path = output_dir / f"{speaker}.zip"
    partial_path = output_dir / f"{speaker}.zip.part"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite {final_path}")
    start = partial_path.stat().st_size if partial_path.exists() else 0
    if start > item["expected_bytes"]:
        raise ValueError(f"partial {speaker} archive exceeds expected size")
    if start == item["expected_bytes"]:
        md5, sha256 = file_hashes(partial_path)
        if md5 != item["publisher_md5"]:
            raise ValueError(f"complete partial {speaker} archive has publisher MD5 mismatch")
        partial_path.replace(final_path)
        return {
            **item,
            "path": str(final_path),
            "sha256": sha256,
            "verified_by": "publisher_md5_and_sha256_after_resume",
            "download_seconds": 0,
        }
    headers = {"User-Agent": "tiny-omni-decision-grid-media-audit/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(item["url"], headers=headers)
    began = time.monotonic()
    last_progress = start
    mode = "ab" if start else "wb"
    with urllib.request.urlopen(request, timeout=180) as response:
        if start:
            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {start}-"
            if response.status != 206 or not content_range.startswith(expected_prefix):
                raise ValueError(
                    f"invalid resume response for {speaker}: "
                    f"{response.status} {content_range}"
                )
        elif response.status != 200:
            raise ValueError(f"unexpected HTTP status for {speaker}: {response.status}")
        with partial_path.open(mode) as output:
            while chunk := response.read(CHUNK_BYTES):
                output.write(chunk)
                current = output.tell()
                if current - last_progress >= PROGRESS_INTERVAL:
                    output.flush()
                    os.fsync(output.fileno())
                    append_jsonl(
                        progress_path,
                        {
                            "speaker": speaker,
                            "downloaded_bytes": current,
                            "expected_bytes": item["expected_bytes"],
                            "elapsed_seconds": round(time.monotonic() - began, 3),
                        },
                    )
                    last_progress = current
    actual_size = partial_path.stat().st_size
    if actual_size != item["expected_bytes"]:
        raise ValueError(
            f"{speaker} size mismatch: expected {item['expected_bytes']}, got {actual_size}"
        )
    md5, sha256 = file_hashes(partial_path)
    if md5 != item["publisher_md5"]:
        raise ValueError(f"{speaker} publisher MD5 mismatch; partial is preserved")
    partial_path.replace(final_path)
    result = {
        **item,
        "path": str(final_path),
        "sha256": sha256,
        "verified_by": "publisher_md5_and_sha256",
        "download_seconds": round(time.monotonic() - began, 3),
    }
    append_jsonl(progress_path, {"speaker": speaker, "status": "verified", **result})
    return result


def acquire(
    *,
    summary_path: Path,
    expected_summary_sha256: str,
    output_dir: Path,
    existing_archives: list[tuple[str, Path]],
    workers: int,
    resume: bool,
) -> dict:
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(REPO):
        raise ValueError("media outputs must remain outside the repository")
    source_sha256, rows = load_inventory(summary_path, expected_summary_sha256)
    items = {row["speaker"]: row for row in rows}
    reused = dict(existing_archives)
    if len(reused) != len(existing_archives) or set(reused) - set(items):
        raise ValueError("existing archive speaker list contains duplicates or unknown speakers")
    if not resume:
        if output_dir.exists():
            raise FileExistsError("use a new output directory or pass --resume for a matching run")
        pending_bytes = sum(
            item["expected_bytes"] for speaker, item in items.items() if speaker not in reused
        )
        if shutil.disk_usage(output_dir.parent).free < pending_bytes + MIN_FREE_RESERVE:
            raise OSError("insufficient free space for archives plus 2 GiB reserve")
        output_dir.mkdir(parents=True)
        (output_dir / "run-manifest.json").write_text(
            json.dumps(
                {
                    "source_summary": str(summary_path.resolve()),
                    "source_summary_sha256": source_sha256,
                    "expected_speakers": sorted(items),
                    "workers": workers,
                    "created_utc_epoch": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        manifest_path = output_dir / "run-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("resume requires an existing run-manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_summary_sha256") != source_sha256
            or manifest.get("expected_speakers") != sorted(items)
        ):
            raise ValueError("resume manifest does not match pinned source inventory")
        remaining_bytes = 0
        for speaker, item in items.items():
            if (output_dir / f"{speaker}.zip").is_file():
                continue
            partial = output_dir / f"{speaker}.zip.part"
            if partial.exists():
                remaining_bytes += max(
                    0, item["expected_bytes"] - partial.stat().st_size
                )
            else:
                remaining_bytes += item["expected_bytes"]
        if shutil.disk_usage(output_dir.parent).free < remaining_bytes + MIN_FREE_RESERVE:
            raise OSError("insufficient free space to resume archives plus 2 GiB reserve")

    progress_path = output_dir / "progress.jsonl"
    results = {
        speaker: validate_existing_archive(items[speaker], path)
        for speaker, path in reused.items()
    }
    already_done = {}
    for speaker, item in items.items():
        path = output_dir / f"{speaker}.zip"
        if path.is_file():
            already_done[speaker] = validate_existing_archive(item, path)
    results.update(already_done)
    pending = [item for speaker, item in items.items() if speaker not in results]
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, item, output_dir, progress_path): item["speaker"]
            for item in pending
        }
        for future in concurrent.futures.as_completed(futures):
            speaker = futures[future]
            try:
                results[speaker] = future.result()
            except Exception as exc:  # keep other speakers progressing and preserve partials
                errors.append({"speaker": speaker, "error": f"{type(exc).__name__}: {exc}"})
    status = "PASS_ARCHIVES_VERIFIED" if not errors and len(results) == len(items) else "INCOMPLETE"
    report = {
        "status": status,
        "source_summary_sha256": source_sha256,
        "expected_speakers": len(items),
        "verified_speakers": len(results),
        "errors": errors,
        "archives": [results[key] for key in sorted(results)],
    }
    (output_dir / "acquisition-result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if errors or len(results) != len(items):
        raise RuntimeError(f"incomplete acquisition; see {output_dir / 'acquisition-result.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--existing-archive",
        action="append",
        type=parse_existing_archive,
        default=[],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    result = acquire(
        summary_path=args.summary,
        expected_summary_sha256=args.expected_summary_sha256,
        output_dir=args.output_dir,
        existing_archives=args.existing_archive,
        workers=args.workers,
        resume=args.resume,
    )
    summary = {
        key: result[key]
        for key in ("status", "expected_speakers", "verified_speakers")
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
