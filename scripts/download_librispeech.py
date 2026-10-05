from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = "https://www.openslr.org/resources/12"
FILES = ("train-clean-100.tar.gz", "dev-clean.tar.gz")
CHUNK_BYTES = 16 * 1024 * 1024
WORKERS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_checksums() -> dict[str, str]:
    with urlopen(f"{BASE_URL}/md5sum.txt", timeout=30) as response:
        content = response.read().decode("ascii")
    return {
        name: digest
        for digest, name in re.findall(r"^([0-9a-f]{32})\s+([^\r\n]+)$", content, re.MULTILINE)
    }


def get_range(name: str, start: int, end: int, total: int) -> tuple[int, bytes]:
    request = Request(
        f"{BASE_URL}/{name}",
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "tiny-omni-decision/1.0"},
    )
    last_error: Exception | None = None
    for _ in range(5):
        try:
            with urlopen(request, timeout=90) as response:
                expected_range = f"bytes {start}-{end}/{total}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected_range
                ):
                    raise RuntimeError("source did not honor the exact requested byte range")
                payload = response.read(end - start + 1)
            if len(payload) != end - start + 1:
                raise RuntimeError("source returned a truncated byte range")
            return start, payload
        except Exception as error:  # noqa: PERF203 - retry remote range failures
            last_error = error
    raise RuntimeError(f"failed byte range {start}-{end}: {last_error}")


def download_one(name: str, destination: Path, expected_md5: str) -> dict[str, object]:
    url = f"{BASE_URL}/{name}"
    with urlopen(Request(url, method="HEAD"), timeout=30) as response:
        total = int(response.headers["Content-Length"])
        range_supported = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    if not range_supported:
        raise RuntimeError(f"{name}: server does not advertise byte-range support")
    if destination.exists() and destination.stat().st_size == total:
        if md5_file(destination) == expected_md5:
            return {
                "path": str(destination),
                "bytes": total,
                "md5": expected_md5,
                "sha256": sha256_file(destination),
                "reused_verified_file": True,
            }

    partial = destination.with_suffix(destination.suffix + ".part")
    journal = partial.with_suffix(partial.suffix + ".json")
    expected_journal = {
        "url": url,
        "size": total,
        "md5": expected_md5,
        "chunk_bytes": CHUNK_BYTES,
    }
    completed: set[int] = set()
    try:
        journal_data = json.loads(journal.read_text(encoding="utf-8"))
        if journal_data.get("source") == expected_journal and partial.stat().st_size == total:
            completed = set(journal_data.get("completed_offsets", []))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    if not completed:
        with partial.open("wb") as handle:
            handle.truncate(total)

    offsets = list(range(0, total, CHUNK_BYTES))
    pending = [offset for offset in offsets if offset not in completed]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, partial.open("r+b") as output:
        futures = {
            pool.submit(
                get_range, name, offset, min(offset + CHUNK_BYTES, total) - 1, total
            ): offset
            for offset in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            offset, payload = future.result()
            output.seek(offset)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            completed.add(offset)
            state = {"source": expected_journal, "completed_offsets": sorted(completed)}
            temporary = journal.with_suffix(journal.suffix + ".tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            temporary.replace(journal)
            if index % 8 == 0 or index == len(pending):
                print(
                    f"{name}: completed {len(completed)}/{len(offsets)} ranges; "
                    f"{min(len(completed) * CHUNK_BYTES, total)} bytes",
                    flush=True,
                )

    md5 = hashlib.md5()
    with partial.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
    if partial.stat().st_size != total or md5.hexdigest() != expected_md5:
        raise RuntimeError(f"{name}: final size or official MD5 verification failed")
    sha256 = sha256_file(partial)
    partial.replace(destination)
    journal.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "bytes": total,
        "md5": md5.hexdigest(),
        "sha256": sha256,
        "reused_verified_file": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch only LibriSpeech train and dev partitions.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/teacher-quality-next/librispeech")
    )
    args = parser.parse_args()
    checksums = official_checksums()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        name: download_one(name, args.output_dir / name, checksums[name]) for name in FILES
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
