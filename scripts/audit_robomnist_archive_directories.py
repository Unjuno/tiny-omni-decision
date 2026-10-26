"""Read only the central directories of pinned RoboMNIST ZIP archives."""

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests

from scripts.audit_robomnist_source import BASE, group_members, require_class

INITIAL_TAIL_BYTES = 64 * 1024
MAX_RANGE_BYTES_PER_ARCHIVE = 512 * 1024
MAX_TOTAL_TRANSFER_BYTES = 32 * 1024 * 1024
EOCD = b"PK\x05\x06"
CENTRAL = b"PK\x01\x02"
CLASS_NAME = re.compile(
    r"(?P<index>\d{2})_Ar(?P<robot>[12])_V(?P<velocity>high|med|low)_Ac(?P<action>[0-9])\.zip"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_eocd(tail, tail_start, archive_size):
    eocd_local = tail.rfind(EOCD)
    if eocd_local < 0 or eocd_local + 22 > len(tail):
        raise ValueError("ZIP EOCD absent")
    comment_size = struct.unpack_from("<H", tail, eocd_local + 20)[0]
    if eocd_local + 22 + comment_size != len(tail):
        raise ValueError("EOCD is not at the end of the bounded tail")
    eocd = tail[eocd_local : eocd_local + 22]
    _, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, _ = struct.unpack(
        "<4s4H2IH", eocd
    )
    if (
        disk
        or cd_disk
        or disk_entries != entries
        or entries == 0xFFFF
        or cd_size == 0xFFFFFFFF
        or cd_offset == 0xFFFFFFFF
    ):
        raise ValueError("multi-disk or ZIP64 archive is outside this bounded auditor")
    if entries == 0 or cd_size == 0 or cd_offset + cd_size > archive_size:
        raise ValueError("invalid central directory bounds")
    return eocd_local, entries, cd_size, cd_offset


def fetch_exact_range(session, url, start, end, file_size, budget):
    expected = end - start + 1
    if expected <= 0 or expected > budget["per_archive_remaining"]:
        raise ValueError("range exceeds per-archive transfer budget")
    if expected > budget["total_remaining"]:
        raise ValueError("range exceeds total transfer budget")
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(10, 30),
    )
    try:
        if response.status_code != 206:
            raise ValueError(f"range request returned HTTP {response.status_code}, expected 206")
        if response.headers.get("Content-Range") != f"bytes {start}-{end}/{file_size}":
            raise ValueError("server returned an unexpected Content-Range")
        if response.headers.get("Content-Length") != str(expected):
            raise ValueError("server returned an unexpected Content-Length")
        body = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                body.extend(chunk)
                if len(body) > expected:
                    raise ValueError("range response exceeded its declared length")
        if len(body) != expected:
            raise ValueError("range response body length mismatch")
        budget["per_archive_remaining"] -= len(body)
        budget["total_remaining"] -= len(body)
        budget["transferred"] += len(body)
        return bytes(body)
    finally:
        response.close()


def read_zip_directory(session, file_record, budget):
    size = int(file_record["size"])
    url = file_record["download_url"]
    if urlparse(url).scheme != "https":
        raise ValueError("archive download URL must use HTTPS")
    budget["per_archive_remaining"] = MAX_RANGE_BYTES_PER_ARCHIVE
    tail_size = min(INITIAL_TAIL_BYTES, size)
    tail_start = size - tail_size
    tail = fetch_exact_range(session, url, tail_start, size - 1, size, budget)
    while True:
        try:
            eocd_local, entries, cd_size, cd_offset = parse_eocd(tail, tail_start, size)
            break
        except ValueError as error:
            if str(error) not in {"ZIP EOCD absent", "EOCD is not at the end of the bounded tail"}:
                raise
            new_size = min(tail_size * 2, MAX_RANGE_BYTES_PER_ARCHIVE, size)
            if new_size == tail_size:
                raise ValueError("ZIP EOCD absent within the bounded tail scan") from error
            new_start = size - new_size
            prefix = fetch_exact_range(session, url, new_start, tail_start - 1, size, budget)
            tail = prefix + tail
            tail_start = new_start
            tail_size = new_size

    if cd_offset >= tail_start and cd_offset + cd_size <= size:
        central = tail[cd_offset - tail_start : cd_offset - tail_start + cd_size]
    else:
        central = fetch_exact_range(session, url, cd_offset, cd_offset + cd_size - 1, size, budget)
    if len(central) != cd_size:
        raise ValueError("central directory length mismatch")

    names = []
    members = []
    offset = 0
    while offset < len(central):
        if central[offset : offset + 4] != CENTRAL or offset + 46 > len(central):
            raise ValueError("invalid central directory record")
        flags = struct.unpack_from("<H", central, offset + 8)[0]
        name_size, extra_size, comment_size = struct.unpack_from("<3H", central, offset + 28)
        record_end = offset + 46 + name_size + extra_size + comment_size
        if record_end > len(central):
            raise ValueError("truncated central directory member")
        name_bytes = central[offset + 46 : offset + 46 + name_size]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        names.append(name)
        compressed_size, uncompressed_size = struct.unpack_from("<II", central, offset + 20)
        crc32 = struct.unpack_from("<I", central, offset + 16)[0]
        local_header_offset = struct.unpack_from("<I", central, offset + 42)[0]
        members.append(
            {
                "name": name,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "crc32": f"{crc32:08x}",
                "local_header_offset": local_header_offset,
            }
        )
        offset = record_end
    if len(names) != entries:
        raise ValueError("central-directory entry count mismatch")
    groups = group_members(names, expected_folder=Path(file_record["name"]).stem)
    real_members = [
        name for name in names if not name.endswith("/") and not name.startswith("__MACOSX/")
    ]
    return {
        "archive": file_record["name"],
        "declared_bytes": size,
        "central_directory_entries": entries,
        "real_members": len(real_members),
        "resource_forks": sum(name.startswith("__MACOSX/") for name in names),
        "repetition_groups": len(groups),
        "groups": groups,
        "member_inventory": members,
        "download_host": urlparse(url).hostname,
        "range_bytes": MAX_RANGE_BYTES_PER_ARCHIVE - budget["per_archive_remaining"],
    }


def validate_version(version, article, receipt, directory):
    version_path = BASE / "article-version-1.json"
    if (
        sha(BASE / "article.json") != receipt["article_sha256"]
        or sha(BASE / "upstream-README.md") != receipt["readme_sha256"]
        or sha(version_path) != directory["version_metadata_sha256"]
        or version["id"] != 28179383
        or version["version"] != 1
        or version["files"] != article["files"]
        or version["license"]["url"] != "https://creativecommons.org/licenses/by/4.0/"
        or len(version["files"]) != 60
    ):
        raise ValueError("the pinned CC BY RoboMNIST version-1 receipt is required")
    for item in version["files"]:
        match = CLASS_NAME.fullmatch(item["name"])
        if not match or int(item["size"]) <= 0:
            raise ValueError("unknown archive class or invalid declared size")
        require_class(**match.groupdict())


def audit(session, version, max_total_bytes=MAX_TOTAL_TRANSFER_BYTES):
    budget = {"total_remaining": max_total_bytes, "transferred": 0}
    results = []
    for item in version["files"]:
        result = read_zip_directory(session, item, budget)
        result["computed_md5_missing"] = not bool(item["computed_md5"])
        results.append(result)
    group_ids = [group["group_id"] for item in results for group in item["groups"]]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("underlying repetition group identity overlaps archives")
    return results, budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; choose a fresh path")
    receipt = json.loads((BASE / "source-receipt.json").read_bytes())
    article = json.loads((BASE / "article.json").read_bytes())
    version_path = BASE / "article-version-1.json"
    version = json.loads(version_path.read_bytes())
    directory = json.loads((BASE / "first-archive-directory.json").read_bytes())
    validate_version(version, article, receipt, directory)
    with requests.Session() as session:
        results, budget = audit(session, version)
    group_ids = [group["group_id"] for item in results for group in item["groups"]]
    group_counts = [item["repetition_groups"] for item in results]
    group_sizes = Counter(group_counts)
    stream_count_by_archive = {}
    total_groups = 0
    for item in results:
        streams = Counter()
        inventory = {member["name"]: member for member in item["member_inventory"]}
        for group in item["groups"]:
            total_groups += 1
            members = group["input_members"] + [group["supervision_only_member"]]
            if len(members) != 8 or any(member not in inventory for member in members):
                raise ValueError("group member inventory is incomplete")
            if len(group["input_members"]) != 7:
                raise ValueError("group must have seven runtime input streams")
            for name in members:
                if "_cam." in name:
                    streams["video"] += 1
                elif "_mic." in name:
                    streams["audio"] += 1
                elif "_csi." in name:
                    streams["csi"] += 1
                elif "_pos." in name:
                    streams["position_supervision"] += 1
        stream_count_by_archive[item["archive"]] = dict(streams)
    output = {
        "schema_version": 1,
        "status": "PASS_DIRECTORY_METADATA_ONLY",
        "source": "RoboMNIST",
        "source_version": 1,
        "license": version["license"],
        "article_version_sha256": sha(version_path),
        "source_receipt_sha256": sha(BASE / "source-receipt.json"),
        "archive_count": len(results),
        "declared_archive_bytes": sum(item["declared_bytes"] for item in results),
        "directory_range_bytes_transferred": budget["transferred"],
        "range_transfer_cap_bytes": MAX_TOTAL_TRANSFER_BYTES,
        "per_archive_cap_bytes": MAX_RANGE_BYTES_PER_ARCHIVE,
        "real_members": sum(item["real_members"] for item in results),
        "resource_forks": sum(item["resource_forks"] for item in results),
        "repetition_groups": len(group_ids),
        "unique_repetition_group_ids": len(set(group_ids)),
        "repetition_group_count_distribution": {
            str(count): number for count, number in sorted(group_sizes.items())
        },
        "complete_repetition_groups": total_groups,
        "incomplete_repetition_groups": 0,
        "runtime_input_streams_per_group": {"video": 3, "audio": 3, "csi": 1},
        "supervision_only_streams_per_group": {"position": 1},
        "stream_counts_by_archive": stream_count_by_archive,
        "archives_missing_computed_md5": [
            item["archive"] for item in results if item["computed_md5_missing"]
        ],
        "archives": results,
        "media_payload_downloaded": False,
        "media_payload_opened": False,
        "training": False,
        "quality_evaluation": False,
        "sealed_audit_opened": False,
        "limitations": [
            "central-directory names/counts only; no ZIP member payload was fetched or opened",
            "group identifiers are filename-derived repetitions, not verified independent sessions",
            "or subjects",
            "source sessions/subjects and temporal independence across repetitions are unverified",
            "no official train/validation split is asserted by this inventory",
            "both-input necessity and task label validity remain untested",
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        "PASS_DIRECTORY_METADATA_ONLY",
        "archives",
        output["archive_count"],
        "groups",
        output["repetition_groups"],
        "range_bytes",
        output["directory_range_bytes_transferred"],
    )


if __name__ == "__main__":
    main()
