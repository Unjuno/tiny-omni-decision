"""Fetch a Zenodo archive with verified HTTP byte ranges and publisher MD5."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

CHUNK_BYTES = 32 * 1024 * 1024


def chunk_ranges(total_bytes: int, chunk_bytes: int = CHUNK_BYTES) -> list[tuple[int, int]]:
    if total_bytes < 1 or chunk_bytes < 1:
        raise ValueError("archive and chunk sizes must be positive")
    return [
        (start, min(start + chunk_bytes, total_bytes) - 1)
        for start in range(0, total_bytes, chunk_bytes)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _preserve_as_untrusted(path: Path) -> Path:
    candidate = path.with_name(path.name + ".untrusted")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.untrusted.{suffix}")
        suffix += 1
    os.replace(path, candidate)
    return candidate


def _fetch_chunk(
    *,
    curl: str,
    url: str,
    total_bytes: int,
    index: int,
    start: int,
    end: int,
    chunks_dir: Path,
) -> Path:
    chunk = chunks_dir / f"{index:04d}.bin"
    chunk_digest = chunk.with_suffix(".sha256")
    expected_length = end - start + 1
    if chunk.is_file() and chunk.stat().st_size == expected_length:
        actual_digest = _sha256(chunk)
        if not chunk_digest.exists():
            chunk_digest.write_text(actual_digest + "\n", encoding="ascii")
            return chunk
        if chunk_digest.read_text(encoding="ascii").strip() == actual_digest:
            return chunk
    partial = chunk.with_suffix(".part")
    headers = chunk.with_suffix(".headers")
    partial_digest = partial.with_suffix(".part.sha256")
    expected_range = (start, end, total_bytes)
    if partial.is_file():
        prior_ranges = re.findall(
            r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$",
            headers.read_text(encoding="latin-1") if headers.is_file() else "",
        )
        digest_is_valid = (
            partial_digest.is_file()
            and partial_digest.read_text(encoding="ascii").strip() == _sha256(partial)
        )
        if (
            not prior_ranges
            or tuple(map(int, prior_ranges[-1])) != expected_range
            or not digest_is_valid
            or partial.stat().st_size > expected_length
        ):
            _preserve_as_untrusted(partial)
            if partial_digest.exists():
                _preserve_as_untrusted(partial_digest)

    for attempt in range(6):
        partial_size = partial.stat().st_size if partial.is_file() else 0
        if partial_size > expected_length:
            partial.unlink()
            partial_size = 0
        request_start = start + partial_size
        if request_start > end:
            break
        piece = chunk.with_name(f"{chunk.stem}.{attempt}.piece")
        piece_headers = chunk.with_name(f"{chunk.stem}.{attempt}.headers")
        piece.unlink(missing_ok=True)
        piece_headers.unlink(missing_ok=True)
        completed = subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                "180",
                "--range",
                f"{request_start}-{end}",
                "--dump-header",
                str(piece_headers),
                "--output",
                str(piece),
                "--write-out",
                "%{http_code}",
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        header_text = piece_headers.read_text(encoding="latin-1") if piece_headers.is_file() else ""
        content_ranges = re.findall(
            r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", header_text
        )
        range_is_valid = bool(content_ranges) and tuple(map(int, content_ranges[-1])) == (
            request_start,
            end,
            total_bytes,
        )
        piece_size = piece.stat().st_size if piece.is_file() else 0
        expected_piece_size = end - request_start + 1
        if (
            completed.returncode == 0
            and range_is_valid
            and piece_size == expected_piece_size
        ):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=chunk.stem + ".merge.",
                suffix=".part",
                dir=chunks_dir,
                delete=False,
            ) as temporary_merge:
                merge_file = Path(temporary_merge.name)
            with merge_file.open("wb") as stream:
                if partial.is_file():
                    with partial.open("rb") as source:
                        for block in iter(lambda: source.read(8 << 20), b""):
                            stream.write(block)
                with piece.open("rb") as source:
                    for block in iter(lambda: source.read(8 << 20), b""):
                        stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            prior_size = partial.stat().st_size if partial.is_file() else 0
            if merge_file.stat().st_size != prior_size + piece_size:
                raise OSError("merged range length does not match verified byte ranges")
            os.replace(merge_file, partial)
            partial_digest.write_text(_sha256(partial) + "\n", encoding="ascii")
            piece.unlink(missing_ok=True)
            if request_start == start:
                headers.write_text(header_text, encoding="latin-1")
            piece_headers.unlink(missing_ok=True)
        else:
            piece.unlink(missing_ok=True)
            piece_headers.unlink(missing_ok=True)
        if partial.is_file() and partial.stat().st_size == expected_length:
            break
        retry_after = re.findall(r"(?im)^retry-after:\s*(\d+)\s*$", header_text)
        if attempt < 5:
            time.sleep(min(int(retry_after[-1]), 60) if retry_after else min(attempt + 1, 10))

    if not partial.is_file() or partial.stat().st_size != expected_length:
        raise RuntimeError(
            f"range incomplete after resumable retries for bytes {start}-{end}: "
            f"{partial.stat().st_size if partial.is_file() else 0}/{expected_length}"
        )
    os.replace(partial, chunk)
    chunk_digest.write_text(_sha256(chunk) + "\n", encoding="ascii")
    partial_digest.unlink(missing_ok=True)
    headers.unlink(missing_ok=True)
    return chunk


def fetch_archive(
    *,
    url: str,
    expected_bytes: int,
    expected_md5: str,
    output: Path,
    workers: int = 4,
    curl: str = "curl.exe",
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite archive: {output}")
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    ranges = chunk_ranges(expected_bytes)
    chunks_dir = output.with_name(output.name + ".chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fetch_chunk,
                curl=curl,
                url=url,
                total_bytes=expected_bytes,
                index=index,
                start=start,
                end=end,
                chunks_dir=chunks_dir,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            print(f"verified range {index + 1}/{len(ranges)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=output.name + ".", suffix=".part", dir=output.parent, delete=False
    ) as assembled:
        temporary = Path(assembled.name)
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        total = 0
        for index in range(len(ranges)):
            with results[index].open("rb") as chunk_file:
                for block in iter(lambda: chunk_file.read(8 << 20), b""):
                    assembled.write(block)
                    md5.update(block)
                    sha256.update(block)
                    total += len(block)
    if total != expected_bytes or md5.hexdigest().lower() != expected_md5.lower():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"archive integrity mismatch: bytes={total}/{expected_bytes}, "
            f"md5={md5.hexdigest()}/{expected_md5}"
        )
    os.replace(temporary, output)
    for path in chunks_dir.iterdir():
        if path.is_file() and re.fullmatch(
            r"\d{4}\.(bin|sha256|part|headers|part\.sha256)", path.name
        ):
            path.unlink()
        elif path.is_file() and re.fullmatch(r"\d{4}\.\d+\.(piece|headers)", path.name):
            path.unlink()
    if not any(chunks_dir.iterdir()):
        chunks_dir.rmdir()
    return {
        "status": "PASS",
        "url": url,
        "bytes": total,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "output": str(output),
        "chunk_bytes": CHUNK_BYTES,
        "workers": workers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--bytes", required=True, type=int)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--curl", default="curl.exe")
    args = parser.parse_args()
    import json

    print(
        json.dumps(
            fetch_archive(
                url=args.url,
                expected_bytes=args.bytes,
                expected_md5=args.md5,
                output=args.output,
                workers=args.workers,
                curl=args.curl,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
