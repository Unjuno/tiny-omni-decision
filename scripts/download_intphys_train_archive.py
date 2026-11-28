"""Download a pinned IntPhys train shard with safe HTTP range resumption."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scripts.intphys_train_archive_pins import TRAIN_ARCHIVE_PINS

DEFAULT_DISK_RESERVE_BYTES = 20 * 1024**3
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


class ArchiveIntegrityError(RuntimeError):
    """Raised when an HTTP response does not match the pinned archive."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    bytes: int
    md5: str
    sha256: str
    etag: str
    resumed_from_bytes: int
    http_responses: int


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()  # type: ignore[attr-defined]
    return int(status)


def _hash_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def download_archive(
    *,
    url: str,
    output_path: Path,
    expected_bytes: int,
    expected_md5: str,
    etag: str,
    max_retries: int = 5,
    reserve_bytes: int = DEFAULT_DISK_RESERVE_BYTES,
    chunk_bytes: int = 1 << 20,
    timeout_seconds: int = 120,
    opener: Callable[..., object] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadResult:
    """Download/resume without truncating an existing partial file.

    Every response is bound to the supplied ETag. Resumes require a matching
    206 Content-Range that starts at the local file length. Transient stream
    failures preserve bytes already written and retry from the new length.
    """
    if expected_bytes <= 0 or max_retries < 0 or chunk_bytes <= 0:
        raise ValueError(
            "archive size and chunk size must be positive; retries must be nonnegative"
        )
    if not etag or etag.startswith("W/"):
        raise ValueError("a strong, non-empty ETag is required for safe resumption")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_size = output_path.stat().st_size if output_path.exists() else 0
    if initial_size > expected_bytes:
        raise ArchiveIntegrityError("existing partial exceeds pinned archive size")
    responses = 0
    consecutive_failures = 0

    while True:
        offset = output_path.stat().st_size if output_path.exists() else 0
        if offset == expected_bytes:
            break
        free_bytes = shutil.disk_usage(output_path.parent).free
        if free_bytes < expected_bytes - offset + reserve_bytes:
            raise OSError("insufficient free disk space to finish archive and preserve reserve")

        headers = {"If-Match": etag, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-{expected_bytes - 1}"
        request = urllib.request.Request(url, headers=headers)

        try:
            response = opener(request, timeout=timeout_seconds)
            responses += 1
            with response:  # type: ignore[attr-defined]
                status = _response_status(response)
                response_headers = response.headers  # type: ignore[attr-defined]
                if response_headers.get("ETag") != etag:
                    raise ArchiveIntegrityError("response ETag differs from pinned source")

                if status == 200 and offset == 0:
                    response_start, response_end, total = 0, expected_bytes - 1, expected_bytes
                    mode = "wb"
                    content_length = response_headers.get("Content-Length")
                    if content_length is None or int(content_length) != expected_bytes:
                        raise ArchiveIntegrityError("full response length differs from archive pin")
                elif status == 206:
                    content_range = response_headers.get("Content-Range", "")
                    match = _CONTENT_RANGE.fullmatch(content_range)
                    if match is None:
                        raise ArchiveIntegrityError("missing or malformed Content-Range")
                    response_start, response_end, total = map(int, match.groups())
                    if response_start != offset or total != expected_bytes:
                        raise ArchiveIntegrityError("range start or total differs from archive pin")
                    content_length = response_headers.get("Content-Length")
                    if (
                        content_length is None
                        or int(content_length) != response_end - response_start + 1
                        or response_end >= expected_bytes
                    ):
                        raise ArchiveIntegrityError("range response length is inconsistent")
                    mode = "ab" if offset else "wb"
                else:
                    raise ArchiveIntegrityError(
                        f"unexpected HTTP status {status} for resume offset {offset}"
                    )

                remaining = response_end - response_start + 1
                with output_path.open(mode) as target:
                    while remaining:
                        block = response.read(min(chunk_bytes, remaining))  # type: ignore[attr-defined]
                        if not block:
                            raise urllib.error.URLError("connection ended before range completed")
                        if len(block) > remaining:
                            raise ArchiveIntegrityError("response exceeded its pinned byte range")
                        target.write(block)
                        remaining -= len(block)
                    target.flush()
            consecutive_failures = 0
        except ArchiveIntegrityError:
            raise
        except urllib.error.HTTPError as error:
            if error.code not in {408, 429} and not 500 <= error.code < 600:
                raise ArchiveIntegrityError(
                    f"server rejected pinned range: HTTP {error.code}"
                ) from error
            consecutive_failures += 1
            if consecutive_failures > max_retries:
                raise RuntimeError(
                    f"archive transfer failed after {max_retries} retries; partial preserved"
                ) from error
            sleep(min(2**(consecutive_failures - 1), 30))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            consecutive_failures += 1
            if consecutive_failures > max_retries:
                raise RuntimeError(
                    f"archive transfer failed after {max_retries} retries; partial preserved"
                ) from error
            sleep(min(2**(consecutive_failures - 1), 30))

    actual_size = output_path.stat().st_size
    if actual_size != expected_bytes:
        raise ArchiveIntegrityError(
            f"archive size mismatch: expected {expected_bytes}, received {actual_size}"
        )
    actual_md5, actual_sha256 = _hash_file(output_path)
    if actual_md5 != expected_md5:
        raise ArchiveIntegrityError("archive MD5 differs from pinned source")
    return DownloadResult(
        path=output_path,
        bytes=actual_size,
        md5=actual_md5,
        sha256=actual_sha256,
        etag=etag,
        resumed_from_bytes=initial_size,
        http_responses=responses,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-number", type=int, choices=TRAIN_ARCHIVE_PINS, required=True)
    parser.add_argument("--etag", required=True, help="strong ETag recorded from the pinned source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--reserve-gib", type=float, default=20.0)
    args = parser.parse_args()
    pin = TRAIN_ARCHIVE_PINS[args.shard_number]
    result = download_archive(
        url=pin["url"],
        output_path=args.output,
        expected_bytes=pin["bytes"],
        expected_md5=pin["md5"],
        etag=args.etag,
        max_retries=args.max_retries,
        reserve_bytes=int(args.reserve_gib * 1024**3),
    )
    print(json.dumps({**result.__dict__, "path": str(result.path)}, sort_keys=True))


if __name__ == "__main__":
    main()
