from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://thor.robots.ox.ac.uk/clevr4/clevr_4_10k_v1.zip"
EXPECTED_SHA512 = (
    "769465d90b6550a242f6d5940b52f2e0944c65af09e17e6f3f2cb65701af66057"
    "ab06485b2194fa9d8932e03a03e87b3fadece43049e1af1ddd9f8e4990ed19a"
)
CHUNK_BYTES = 16 * 1024 * 1024
WORKERS = 8


def sha512_file(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_range(start: int, end: int, total: int) -> tuple[int, bytes]:
    request = Request(URL, headers={"Range": f"bytes={start}-{end}"})
    last_error: Exception | None = None
    for _ in range(5):
        try:
            with urlopen(request, timeout=120) as response:
                expected = f"bytes {start}-{end}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("server did not honor exact requested byte range")
                payload = response.read(end - start + 1)
            if len(payload) != end - start + 1:
                raise RuntimeError("truncated range response")
            return start, payload
        except Exception as error:  # noqa: PERF203 - retry transient range failures
            last_error = error
    raise RuntimeError(f"failed archive range {start}-{end}: {last_error}")


def download(path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(URL, method="HEAD"), timeout=30) as response:
        size = int(response.headers["Content-Length"])
        if "bytes" not in response.headers.get("Accept-Ranges", "").lower():
            raise RuntimeError("Clevr-4 source does not advertise byte-range support")
    if path.is_file() and path.stat().st_size == size:
        digest = sha512_file(path)
        if digest == EXPECTED_SHA512:
            return {"path": str(path), "bytes": size, "sha512": digest, "reused": True}

    partial = path.with_suffix(path.suffix + ".part")
    journal = path.with_suffix(path.suffix + ".ranges.json")
    source = {"url": URL, "size": size, "sha512": EXPECTED_SHA512, "chunk_bytes": CHUNK_BYTES}
    completed: set[int] = set()
    try:
        state = json.loads(journal.read_text(encoding="utf-8"))
        if state.get("source") == source and partial.stat().st_size == size:
            completed = set(state.get("completed_offsets", []))
    except (OSError, json.JSONDecodeError):
        pass
    if partial.exists() and partial.stat().st_size != size:
        partial.unlink()
        completed.clear()
    if not completed:
        with partial.open("wb") as handle:
            handle.truncate(size)

    offsets = list(range(0, size, CHUNK_BYTES))
    pending = [offset for offset in offsets if offset not in completed]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, partial.open("r+b") as output:
        futures = {
            pool.submit(fetch_range, offset, min(offset + CHUNK_BYTES, size) - 1, size): offset
            for offset in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            offset, payload = future.result()
            output.seek(offset)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            completed.add(offset)
            temporary = journal.with_suffix(journal.suffix + ".tmp")
            temporary.write_text(
                json.dumps({"source": source, "completed_offsets": sorted(completed)}),
                encoding="utf-8",
            )
            temporary.replace(journal)
            if index % 8 == 0 or index == len(pending):
                print(f"verified ranges received: {len(completed)}/{len(offsets)}", flush=True)

    digest = sha512_file(partial)
    if partial.stat().st_size != size or digest != EXPECTED_SHA512:
        raise RuntimeError(f"Clevr-4 archive verification failed (sha512={digest})")
    partial.replace(path)
    journal.unlink(missing_ok=True)
    return {"path": str(path), "bytes": size, "sha512": digest, "reused": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the SHA-512-pinned Clevr-4 image archive.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/teacher-quality-next/clevr4/clevr_4_10k_v1.zip"),
    )
    args = parser.parse_args()
    print(json.dumps(download(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
