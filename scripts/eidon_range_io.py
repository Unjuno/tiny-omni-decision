"""Bounded HTTP byte-range I/O for pinned Eidon media and Parquet shards."""

from __future__ import annotations

from collections import OrderedDict

import requests


class HTTPRangeReader:
    """Seekable file-like reader that coalesces reads into bounded LRU blocks."""

    def __init__(
        self,
        url: str,
        *,
        max_bytes: int,
        block_size: int = 8 * 1024 * 1024,
        max_cached_blocks: int = 4,
        timeout: float = 45,
        session=None,
    ):
        for name, value in (
            ("max_bytes", max_bytes),
            ("block_size", block_size),
            ("max_cached_blocks", max_cached_blocks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._session = requests.Session() if session is None else session
        self._owns_session = session is None
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._block_size = block_size
        self._max_cached_blocks = max_cached_blocks
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self.position = 0
        self.bytes_read = 0
        self.range_requests = 0
        self.closed = False

        response = self._session.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
        try:
            self.size = int(response.headers["content-length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("range source must report a valid content length") from exc
        if self.size < 1:
            raise ValueError("range source must be non-empty")
        if "bytes" not in response.headers.get("accept-ranges", "").casefold():
            raise ValueError("source does not advertise byte-range support")
        self.url = response.url

    def readable(self) -> bool:
        return not self.closed

    def seekable(self) -> bool:
        return not self.closed

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed range reader")
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError("invalid whence")
        if position < 0 or position > self.size:
            raise ValueError("seek outside range source")
        self.position = position
        return position

    def _block(self, index: int) -> bytes:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        start = index * self._block_size
        end = min(start + self._block_size, self.size)
        length = end - start
        if self.bytes_read + length > self._max_bytes:
            raise ValueError("range byte budget exceeded")
        response = self._session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end - 1}"},
            timeout=self._timeout,
        )
        if response.status_code != 206:
            raise ValueError("server did not honor byte range")
        expected_range = f"bytes {start}-{end - 1}/{self.size}"
        if response.headers.get("content-range") != expected_range:
            raise ValueError("server returned an unexpected content range")
        if len(response.content) != length:
            raise ValueError("server returned an incomplete byte range")

        content = response.content
        self.bytes_read += length
        self.range_requests += 1
        self._cache[index] = content
        while len(self._cache) > self._max_cached_blocks:
            self._cache.popitem(last=False)
        return content

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed range reader")
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("read size must be an integer")
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""

        output = bytearray()
        remaining = size
        while remaining:
            block_index = self.position // self._block_size
            content = self._block(block_index)
            block_offset = self.position - block_index * self._block_size
            count = min(remaining, len(content) - block_offset)
            output.extend(content[block_offset : block_offset + count])
            self.position += count
            remaining -= count
        return bytes(output)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._cache.clear()
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> HTTPRangeReader:
        if self.closed:
            raise ValueError("I/O operation on closed range reader")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
