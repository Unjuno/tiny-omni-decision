"""Index TAR headers over verified HTTP Range requests without downloading payloads."""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import requests


class HttpRangeReader:
    """Small seekable reader that fetches and validates fixed-size HTTP ranges."""

    def __init__(
        self,
        url: str,
        size: int,
        etag: str,
        *,
        block_size: int = 64 * 1024,
        max_cached_blocks: int = 8,
        retries: int = 5,
        timeout: tuple[float, float] = (10.0, 30.0),
        session: requests.Session | None = None,
    ) -> None:
        if min(size, block_size, max_cached_blocks, retries) < 1:
            raise ValueError("size, block size, cache size, and retries must be positive")
        self.url = url
        self.size = size
        self.etag = etag
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.retries = retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.range_requests = 0
        self.range_bytes = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError(f"invalid seek mode {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def _block(self, start: int) -> bytes:
        if start in self.cache:
            self.cache.move_to_end(start)
            return self.cache[start]
        end = min(start + self.block_size, self.size) - 1
        expected_range = f"bytes {start}-{end}/{self.size}"
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise ValueError(f"server ignored Range request: HTTP {response.status_code}")
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError(
                        f"unexpected Content-Range: {response.headers.get('Content-Range')}"
                    )
                if response.headers.get("ETag") != self.etag:
                    raise ValueError(f"archive ETag changed: {response.headers.get('ETag')}")
                data = response.content
                if len(data) != end - start + 1:
                    raise ValueError("range payload length does not match Content-Range")
                self.range_requests += 1
                self.range_bytes += len(data)
                self.cache[start] = data
                while len(self.cache) > self.max_cached_blocks:
                    self.cache.popitem(last=False)
                return data
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"failed verified range {start}-{end}") from last_error

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, max(0, self.size - self.position))
        output = bytearray()
        while size:
            start = (self.position // self.block_size) * self.block_size
            block = self._block(start)
            offset = self.position - start
            count = min(size, len(block) - offset)
            output.extend(block[offset : offset + count])
            self.position += count
            size -= count
        return bytes(output)


def safe_member_path(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def index_tar(
    reader: HttpRangeReader,
    allowed_sessions: set[str],
    *,
    progress_every: int = 500,
) -> dict:
    if not allowed_sessions:
        raise ValueError("at least one allowed train/validation session is required")
    members: list[dict[str, int | str | bool]] = []
    scanned_member_count = 0
    with tarfile.open(fileobj=reader, mode="r:") as archive:
        for index, member in enumerate(archive):
            scanned_member_count += 1
            if not safe_member_path(member.name):
                raise ValueError(f"unsafe archive member path: {member.name}")
            session_id = PurePosixPath(member.name).parts[0]
            if session_id not in allowed_sessions:
                continue
            members.append(
                {
                    "index": index,
                    "name": member.name,
                    "session_id": session_id,
                    "size": member.size,
                    "offset_data": member.offset_data,
                    "is_file": member.isfile(),
                    "is_directory": member.isdir(),
                }
            )
            if progress_every and (index + 1) % progress_every == 0:
                print(
                    f"scanned_members={index + 1} indexed_allowed={len(members)} "
                    f"range_requests={reader.range_requests} "
                    f"range_bytes={reader.range_bytes} tar_offset={archive.offset}",
                    flush=True,
                )
    return {
        "scanned_member_count": scanned_member_count,
        "allowed_member_count": len(members),
        "range_requests": reader.range_requests,
        "range_bytes": reader.range_bytes,
        "video_members": [
            item for item in members if item["name"].lower().endswith(('.mp4', '.mov'))
        ],
        "audio_members": [
            item for item in members if item["name"].lower().endswith(('.wav', '.aac', '.flac'))
        ],
        "members": members,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--expected-etag", required=True)
    parser.add_argument("--allowed-sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite archive index: {args.output}")
    allowed_sessions = {
        line.strip()
        for line in args.allowed_sessions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    reader = HttpRangeReader(args.url, args.expected_size, args.expected_etag)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive_url": args.url,
        "archive_size_bytes": args.expected_size,
        "archive_etag": args.expected_etag,
        "payload_media_downloaded": False,
        "index": index_tar(
            reader, allowed_sessions, progress_every=args.progress_every
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
