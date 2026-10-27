"""Fetch only frozen GRID validation video members with verified HTTP ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import time
import zlib
from pathlib import Path

import requests
import urllib3.util.connection

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
TAILS = ROOT / "grid-official-speaker-audio-20261028-attempt-01"
METADATA = ROOT / "grid-validation-video-20261028-attempt-01/zenodo-record-3625687.json"
SELECTION_SHA256 = "e897ff02a435a463211f0a157495058adcb5c390bf384437e3e56dc8066da281"
SPEAKERS = ("s13", "s18", "s23", "s24", "s25", "s27", "s4")
ALL_SPEAKERS = (*SPEAKERS, "s30")

# This host has a confirmed stalled IPv6 path to Zenodo; keep this downloader IPv4-only.
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET


def parse_central_directory(
    tail_path: Path, *, range_start: int, total_size: int
) -> dict[str, dict]:
    data = tail_path.read_bytes()
    if range_start < 0 or range_start + len(data) != total_size:
        raise ValueError("metadata range must be contiguous through EOF")
    eocd = data.rfind(b"PK\x05\x06")
    if eocd < 0 or eocd + 22 > len(data):
        raise ValueError("EOCD missing")
    fields = struct.unpack_from("<4s4H2LH", data, eocd)
    _, disk, cd_disk, disk_count, total_count, cd_size, cd_offset, comment_len = fields
    if eocd + 22 + comment_len != len(data) or disk or cd_disk or disk_count != total_count:
        raise ValueError("unsupported or malformed ZIP directory")
    if total_count == 0xFFFF or cd_offset < range_start or cd_offset + cd_size > total_size:
        raise ValueError("ZIP64 or incomplete central directory")
    pos, end, entries = cd_offset - range_start, cd_offset - range_start + cd_size, {}
    count = 0
    while pos < end:
        if data[pos : pos + 4] != b"PK\x01\x02" or pos + 46 > end:
            raise ValueError("bad central-directory entry")
        f = struct.unpack_from("<4s6H3L5H2L", data, pos)
        flags, method, crc, csize, usize = f[3], f[4], f[7], f[8], f[9]
        name_len, extra_len, comment_len = f[10:13]
        name_end = pos + 46 + name_len
        entry_end = name_end + extra_len + comment_len
        if entry_end > end:
            raise ValueError("truncated central-directory entry")
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = data[pos + 46 : name_end].decode(encoding)
        local_offset = f[16]
        if name.lower().endswith(".mpg"):
            entries[name] = dict(
                flags=flags,
                method=method,
                crc32=crc,
                compressed_size=csize,
                size=usize,
                local_header_offset=local_offset,
            )
        pos, count = entry_end, count + 1
    if pos != end or count != total_count:
        raise ValueError("central-directory count mismatch")
    return entries


def ranged_member(url: str, name: str, info: dict, total_size: int, session=None) -> bytes:
    http = session or requests
    head_start = info["local_header_offset"]
    prefix = http.get(url, headers={"Range": f"bytes={head_start}-{head_start + 29}"}, timeout=60)
    if (
        prefix.status_code != 206
        or len(prefix.content) != 30
        or prefix.headers.get("Content-Range")
        != f"bytes {head_start}-{head_start + 29}/{total_size}"
    ):
        raise ValueError("local-header prefix range mismatch")
    if prefix.content[:4] != b"PK\x03\x04":
        raise ValueError("invalid local header")
    values = struct.unpack_from("<4s5H3L2H", prefix.content, 0)
    flags, method, name_len, extra_len = values[2], values[3], values[-2], values[-1]
    if method != info["method"]:
        raise ValueError("local and central ZIP compression differ")
    metadata_start = head_start + 30
    metadata_end = metadata_start + name_len + extra_len - 1
    metadata = http.get(
        url,
        headers={"Range": f"bytes={metadata_start}-{metadata_end}"},
        timeout=60,
    )
    if (
        metadata.status_code != 206
        or len(metadata.content) != name_len + extra_len
        or metadata.headers.get("Content-Range")
        != f"bytes {metadata_start}-{metadata_end}/{total_size}"
    ):
        raise ValueError("local-header name/extra range mismatch")
    local_name = metadata.content[:name_len].decode("utf-8" if flags & 0x800 else "cp437")
    if local_name != name:
        raise ValueError("local and central ZIP headers differ")
    start = metadata_end + 1
    end = start + info["compressed_size"] - 1
    body_response = http.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
    if body_response.status_code != 206 or len(body_response.content) != info["compressed_size"]:
        raise ValueError("member range response length/status mismatch")
    if body_response.headers.get("Content-Range") != f"bytes {start}-{end}/{total_size}":
        raise ValueError("member Content-Range mismatch")
    compressed = body_response.content
    if info["method"] == 0:
        body = compressed
    elif info["method"] == 8:
        body = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported ZIP compression method {info['method']}")
    if len(body) != info["size"] or zlib.crc32(body) & 0xFFFFFFFF != info["crc32"]:
        raise ValueError("uncompressed member size/CRC mismatch")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-new-members", type=int)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    if hashlib.sha256(args.selection.read_bytes()).hexdigest() != SELECTION_SHA256:
        raise ValueError("selection hash differs")
    if not args.estimate_only and args.output is None:
        raise ValueError("output directory required for fetching")
    if args.output and args.output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("external output path required")
    if args.limit_new_members is not None and args.limit_new_members < 1:
        raise ValueError("limit must be positive")
    record = json.loads(METADATA.read_bytes())
    files = {f["key"]: f for f in record["files"]}
    rows = [json.loads(line) for line in args.selection.read_text(encoding="utf-8").splitlines()]
    selected: dict[str, set[str]] = {speaker: set() for speaker in ALL_SPEAKERS}
    for row in rows:
        selected[row["speaker"]].update(row["codes"])
    if args.estimate_only:
        estimates = {}
        for speaker in SPEAKERS:
            item = files[f"{speaker}.zip"]
            total = item["size"]
            tail = TAILS / f"{speaker}-video-tail.bin"
            index = parse_central_directory(
                tail,
                range_start=total - tail.stat().st_size,
                total_size=total,
            )
            estimates[speaker] = sum(
                index[f"{speaker}/{code}.mpg"]["compressed_size"] for code in selected[speaker]
            )
        print(
            json.dumps(
                {
                    "selected_video_count": sum(map(len, selected.values())),
                    "compressed_bytes_by_speaker": estimates,
                    "compressed_bytes_total": sum(estimates.values()),
                },
                indent=2,
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "range-extraction.json"
    summary = (
        json.loads(summary_path.read_bytes())
        if summary_path.exists()
        else {"selection_sha256": SELECTION_SHA256, "speakers": {}, "files": []}
    )
    if summary["selection_sha256"] != SELECTION_SHA256:
        raise ValueError("existing extraction belongs to a different selection")

    def save_summary() -> None:
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(summary_path)

    completed = {(f["speaker"], f["code"]) for f in summary["files"]}
    newly_completed = 0
    for speaker in SPEAKERS:
        item = files[f"{speaker}.zip"]
        total = item["size"]
        tail = TAILS / f"{speaker}-video-tail.bin"
        index = parse_central_directory(
            tail, range_start=total - tail.stat().st_size, total_size=total
        )
        url = item["links"]["self"]
        session = requests.Session()
        wanted = sorted(selected[speaker])
        for idx, code in enumerate(wanted, 1):
            if (speaker, code) in completed:
                continue
            name = f"{speaker}/{code}.mpg"
            started = time.perf_counter()
            body = ranged_member(url, name, index[name], total, session)
            seconds = time.perf_counter() - started
            digest = hashlib.sha256(body).hexdigest()
            dest = args.output / f"{digest}.mpg"
            dest.write_bytes(body)
            summary["files"].append(
                {
                    "speaker": speaker,
                    "code": code,
                    "path": str(dest),
                    "sha256": digest,
                    "bytes": len(body),
                    "crc32": f"{index[name]['crc32']:08x}",
                    "download_seconds": seconds,
                    "mib_per_second": len(body) / seconds / 2**20,
                }
            )
            completed.add((speaker, code))
            newly_completed += 1
            save_summary()
            print(f"{speaker}/{code}: {len(body)} bytes in {seconds:.2f}s", flush=True)
            if idx % 10 == 0 or idx == len(wanted):
                print(f"{speaker}: {idx}/{len(wanted)}", flush=True)
            if args.limit_new_members and newly_completed >= args.limit_new_members:
                break
        summary["speakers"][speaker] = {
            "expected_md5": item["checksum"],
            "archive_size": total,
            "selected_codes": len(wanted),
        }
        session.close()
        if args.limit_new_members and newly_completed >= args.limit_new_members:
            break
    save_summary()
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "members": len(summary["files"]),
                "new_members": newly_completed,
            }
        )
    )


if __name__ == "__main__":
    main()
