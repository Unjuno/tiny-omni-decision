"""Fetch and validate one preregistered RoboMNIST repetition by HTTPS byte ranges."""

import argparse
import hashlib
import io
import json
import statistics
import struct
import zlib
from pathlib import Path
from urllib.parse import urlparse

import requests

MAX_TRANSFER_BYTES = 32 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
LOCAL_HEADER = b"PK\x03\x04"


def fetch_range(session, url, start, end, total, remaining):
    count = end - start + 1
    if count <= 0 or count > remaining[0]:
        raise ValueError("payload range exceeds fixed transfer budget")
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(10, 60),
    )
    try:
        if response.status_code != 206:
            raise ValueError(f"range request returned HTTP {response.status_code}, expected 206")
        if response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
            raise ValueError("unexpected Content-Range")
        if response.headers.get("Content-Length") != str(count):
            raise ValueError("unexpected Content-Length")
        payload = bytearray()
        for chunk in response.iter_content(64 * 1024):
            if chunk:
                payload.extend(chunk)
                if len(payload) > count:
                    raise ValueError("range response exceeded requested length")
        if len(payload) != count:
            raise ValueError("range response was truncated")
        remaining[0] -= count
        return bytes(payload)
    finally:
        response.close()


def extract_member(session, url, archive_size, member, remaining, expanded):
    if urlparse(url).scheme != "https":
        raise ValueError("source archive URL must use HTTPS")
    offset = member["local_header_offset"]
    header = fetch_range(session, url, offset, offset + 29, archive_size, remaining)
    if header[:4] != LOCAL_HEADER:
        raise ValueError("local ZIP header signature mismatch")
    flags, method = struct.unpack_from("<HH", header, 6)
    filename_len, extra_len = struct.unpack_from("<HH", header, 26)
    if flags & 1:
        raise ValueError("encrypted ZIP members are outside this auditor")
    name = fetch_range(
        session,
        url,
        offset + 30,
        offset + 30 + filename_len - 1,
        archive_size,
        remaining,
    ).decode("utf-8")
    if name != member["name"]:
        raise ValueError("local header member name does not match central directory")
    data_offset = offset + 30 + filename_len + extra_len
    compressed_size = member["compressed_size"]
    if compressed_size > MAX_MEMBER_BYTES:
        raise ValueError("compressed member exceeds fixed per-member limit")
    compressed = fetch_range(
        session,
        url,
        data_offset,
        data_offset + compressed_size - 1,
        archive_size,
        remaining,
    )
    if method == 0:
        content = compressed
    elif method == 8:
        decoder = zlib.decompressobj(-15)
        content = decoder.decompress(compressed, MAX_MEMBER_BYTES + 1)
        if len(content) > MAX_MEMBER_BYTES or decoder.unconsumed_tail:
            raise ValueError("deflate stream exceeds the member size limit")
        content += decoder.flush(MAX_MEMBER_BYTES + 1 - len(content))
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("invalid or oversized deflate stream")
    else:
        raise ValueError(f"unsupported ZIP compression method {method}")
    if len(content) != member["uncompressed_size"] or len(content) > MAX_MEMBER_BYTES:
        raise ValueError("uncompressed member size mismatch or limit exceeded")
    if zlib.crc32(content) & 0xFFFFFFFF != int(member["crc32"], 16):
        raise ValueError("member CRC-32 mismatch")
    expanded[0] += len(content)
    if expanded[0] > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("total uncompressed payload exceeds fixed limit")
    return content


def decode_media(name, content):
    import av

    container = av.open(io.BytesIO(content), mode="r")
    try:
        streams = list(container.streams)
        if len(streams) != 1:
            raise ValueError(f"expected one media stream in {name}")
        stream = streams[0]
        if name.endswith(".mp4"):
            pts = []
            frames = 0
            for frame in container.decode(stream):
                frames += 1
                if frame.pts is not None:
                    pts.append(float(frame.pts * stream.time_base))
            if not frames:
                raise ValueError(f"video contains no decodable frames: {name}")
            return {
                "kind": "video",
                "codec": stream.codec_context.name,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "frames_decoded": frames,
                "first_timestamp_seconds": min(pts) if pts else None,
                "last_timestamp_seconds": max(pts) if pts else None,
                "strictly_increasing_timestamps": all(
                    right > left for left, right in zip(pts, pts[1:], strict=False)
                ),
                "duration_seconds": float(stream.duration * stream.time_base)
                if stream.duration is not None
                else None,
            }
        sample_count = 0
        for frame in container.decode(stream):
            sample_count += frame.samples
        if not sample_count:
            raise ValueError(f"audio contains no decodable samples: {name}")
        rate = stream.codec_context.sample_rate
        return {
            "kind": "audio",
            "codec": stream.codec_context.name,
            "sample_rate_hz": rate,
            "channels": getattr(stream.codec_context, "channels", None),
            "samples_decoded": sample_count,
            "duration_seconds": sample_count / rate,
        }
    finally:
        container.close()


def summarize_timestamps(value):
    timestamps = []

    def walk(item, path="$", depth=0):
        if depth > 8:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}"
                if key.lower() in {"time_stamp_ns", "timestamp_ns"} and isinstance(
                    child, list
                ) and child and all(isinstance(number, (int, float)) for number in child):
                    deltas = [
                        right - left for left, right in zip(child, child[1:], strict=False)
                    ]
                    timestamps.append(
                        {
                            "path": child_path,
                            "count": len(child),
                            "first_ns": child[0],
                            "last_ns": child[-1],
                            "strictly_increasing": all(delta > 0 for delta in deltas),
                            "median_interval_ns": statistics.median(deltas) if deltas else None,
                            "duration_seconds": (child[-1] - child[0]) / 1_000_000_000,
                        }
                    )
                else:
                    walk(child, child_path, depth + 1)
        elif isinstance(item, list):
            for index, child in enumerate(item[:16]):
                walk(child, f"{path}[{index}]", depth + 1)

    walk(value)
    return timestamps


def json_shape(value, depth=0):
    if isinstance(value, dict):
        return {key: json_shape(item, depth + 1) for key, item in list(value.items())[:24]}
    if isinstance(value, list):
        if not value:
            return {"type": "list", "length": 0}
        return {
            "type": "list",
            "length": len(value),
            "first_item": json_shape(value[0], depth + 1) if depth < 3 else type(value[0]).__name__,
        }
    if isinstance(value, (int, float, str, bool)) or value is None:
        return type(value).__name__
    return type(value).__name__


def audit_group(session, version, audit, directory_audit_path, receipt_path):
    archive = "00_Ar1_Vhigh_Ac0.zip"
    entry = next(item for item in version["files"] if item["name"] == archive)
    inventory = next(item for item in audit["archives"] if item["archive"] == archive)
    groups = inventory["groups"]
    group = min(groups, key=lambda item: item["group_id"])
    members_by_name = {item["name"]: item for item in inventory["member_inventory"]}
    selected = group["input_members"] + [group["supervision_only_member"]]
    remaining = [MAX_TRANSFER_BYTES]
    expanded = [0]
    decoded = {}
    payloads = {}
    for name in selected:
        member = members_by_name[name]
        payloads[name] = extract_member(
            session, entry["download_url"], entry["size"], member, remaining, expanded
        )
    for name, content in payloads.items():
        if name.endswith(".mp4") or name.endswith(".wav"):
            decoded[name] = decode_media(name, content)
        else:
            value = json.loads(content)
            decoded[name] = {
                "kind": "json",
                "shape": json_shape(value),
                "timestamps": summarize_timestamps(value),
            }
    result = {
        "status": "PASS_ONE_REPETITION_PAYLOAD_DECODED",
        "archive": archive,
        "archive_size": entry["size"],
        "archive_declared_md5": entry.get("computed_md5"),
        "archive_md5_verified": False,
        "source_version_sha256": audit["article_version_sha256"],
        "directory_audit_sha256": hashlib.sha256(
            Path(directory_audit_path).read_bytes()
        ).hexdigest(),
        "selection_rule": "smallest validated group_id in the pinned first archive",
        "group_id": group["group_id"],
        "input_members": group["input_members"],
        "position_member_supervision_only": group["supervision_only_member"],
        "compressed_payload_bytes_transferred": MAX_TRANSFER_BYTES - remaining[0],
        "uncompressed_payload_bytes_checked": expanded[0],
        "transfer_budget_bytes": MAX_TRANSFER_BYTES,
        "per_member_uncompressed_cap_bytes": MAX_MEMBER_BYTES,
        "whole_archive_downloaded": False,
        "decoded": decoded,
        "source_limits": [
            "one filename-derived repetition; independent sessions/generalization are unproven",
            "per-member CRC and decoded shape do not verify the whole-archive MD5",
            "timestamps/schema are inspected; no task score or model quality is measured",
            "three cameras/mics share a group; timestamp alignment is not yet certified",
        ],
        "training": False,
        "model_inference": False,
        "quality_measured": False,
        "sealed_audit_opened": False,
    }
    with Path(receipt_path).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory-audit", type=Path, required=True)
    parser.add_argument("--version-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh output receipt path")
    audit = json.loads(args.directory_audit.read_bytes())
    version = json.loads(args.version_metadata.read_bytes())
    if audit["status"] != "PASS_DIRECTORY_METADATA_ONLY" or audit["source_version"] != 1:
        raise ValueError("pinned version-1 directory audit required")
    if version["id"] != 28179383 or version["version"] != 1:
        raise ValueError("pinned RoboMNIST version-1 metadata required")
    with requests.Session() as session:
        result = audit_group(session, version, audit, args.directory_audit, args.output)
    print(result["status"], result["group_id"], result["compressed_payload_bytes_transferred"])


if __name__ == "__main__":
    main()
