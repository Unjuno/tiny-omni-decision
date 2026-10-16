"""Bounded, version-checked ZIP member reads for the official Greatest Hits source."""

import hashlib
import json
import re
import struct
import subprocess
import time
import zlib

ID = re.compile(r"[0-9]{4}(?:-[0-9]+){5,6}")
MEMBER_LIMIT = 32 * 2**20


def parse_ids(payload):
    values = payload.decode("ascii").splitlines()
    if not values or any(not ID.fullmatch(v) for v in values):
        raise ValueError("unreviewed source ID format")
    return values


def select_train(train, test, parents):
    if (len(set(train)) != len(train) or len(set(test)) != len(test)
            or set(train) & set(test) or set(train) | set(test) != parents or len(train) < 3):
        raise ValueError("complete unique disjoint official parent split required")
    return sorted(train, key=lambda s: hashlib.sha256(("17:" + s).encode()).hexdigest())[:3]


def validate_range(header, body, start, count, size, etag):
    lines = header.splitlines()
    headers = {k.strip().lower(): v.strip() for s in lines if ":" in s
               for k, v in [s.split(":", 1)]}
    if (not lines or " 206 " not in lines[0] or headers.get("etag") != etag
            or headers.get("content-range") != f"bytes {start}-{start + count - 1}/{size}"
            or len(body) != count):
        raise ValueError("exact206 range, source version and length required")


def member_bytes(entry, read):
    name = entry["name"]
    if (not name.startswith("vis-data-256/") or ".." in name or "\\" in name
            or name.count("/") != 1 or not 0 < entry["bytes"] <= MEMBER_LIMIT
            or not 0 < entry["compressed"] <= MEMBER_LIMIT or entry["offset"] < 0
            or entry["compression"] not in (0, 8)):
        raise ValueError("safe bounded member required")
    header = read(entry["offset"], 30)
    if len(header) != 30:
        raise ValueError("truncated ZIP header")
    magic, _, flags, method, _, _, _, _, _, nlen, elen = struct.unpack("<4s5H3I2H", header)
    if (magic != b"PK\x03\x04" or flags & 1 or method != entry["compression"]
            or not 0 < nlen <= 512 or elen > 4096):
        raise ValueError("invalid/encrypted ZIP header")
    tail = read(entry["offset"] + 30, nlen + elen + entry["compressed"])
    if len(tail) != nlen + elen + entry["compressed"]:
        raise ValueError("truncated ZIP body")
    if tail[:nlen].decode("utf-8" if flags & 2048 else "cp437") != name:
        raise ValueError("local/directory member name differs")
    compressed = tail[nlen + elen:]
    if method == 8:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(compressed, entry["bytes"] + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("invalid or oversized deflate member")
    else:
        data = compressed
    if len(data) != entry["bytes"] or f"{zlib.crc32(data):08x}" != entry["crc32"]:
        raise ValueError("member length or CRC differs")
    return data


class CurlRangeSource:
    """IPv4/TLS with an exact response contract; no redirect, retry or full download."""

    def __init__(self, url, size, etag, directory):
        if not url.startswith("https://"):
            raise ValueError("HTTPS required")
        self.url, self.size, self.etag = url, size, etag
        self.directory = directory
        directory.mkdir(exist_ok=False)
        self.requested, self.requests, self.started = 0, [], time.monotonic()

    def read(self, start, count):
        if (start < 0 or count <= 0 or start + count > self.size
                or self.requested + count > 64 * 2**20
                or time.monotonic() - self.started > 300):
            raise ValueError("range,64MiB or300s resource bound exceeded")
        name = f"range-{len(self.requests):03d}"
        body, header = self.directory / (name + ".bin"), self.directory / (name + ".headers")
        item = dict(start=start, count=count, status="STARTED")
        self.requests.append(item)
        self.requested += count
        try:
            cmd = ["curl.exe", "-4", "--fail", "--silent", "--show-error", "--proto", "=https",
                   "--connect-timeout", "10", "--max-time", "25", "--max-filesize", str(count),
                   "--range", f"{start}-{start + count - 1}", "--header", "If-Match: " + self.etag,
                   "--header", "Accept-Encoding: identity", "--dump-header", str(header),
                   "--output", str(body), self.url]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            if run.returncode:
                raise RuntimeError(run.stderr)
            data = body.read_bytes()
            validate_range(header.read_text(encoding="latin1"), data, start, count,
                           self.size, self.etag)
            item.update(status="PASS", sha256=hashlib.sha256(data).hexdigest())
            return data
        except Exception as exc:
            item.update(status="FAILED", error=repr(exc))
            raise
        finally:
            (self.directory / "requests.json").write_text(
                json.dumps(self.requests, indent=2), encoding="utf-8")
