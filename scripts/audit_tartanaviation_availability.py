"""Read-only TartanAviation audio/ADS-B object-index feasibility audit.

This requests Swift JSON object listings and bounded ZIP central-directory
tails only. It never downloads or extracts audio, trajectory rows, or archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_URL = (
    "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/"
    "AUTH_ac8533a83cff4d48bc8c608ad222d330"
)
USER_AGENT = "tiny-omni-decision-r2-metadata-audit/1.0"
RANGE_BYTES = 8 * 1024 * 1024
AUDIO_KEY = re.compile(
    r"^(?P<airport>kagc|kbtp)/(?P<year>\d{4})/(?P<month>\d{1,2})/"
    r"(?P<date>\d{2}-\d{2}-\d{2})_audio\.zip$"
)
ADSB_ARCHIVE = re.compile(r"^(?P<airport>kagc|kbtp)/raw/(?P<year>20\d{2})\.zip$")
ADSB_MEMBER_DATE = re.compile(r"(?<!\d)(?P<date>\d{2}-\d{2}-\d{2})(?:/|$)")


def _request(url: str, *, headers: dict[str, str] | None = None) -> Any:
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, response.headers, response.read()


def list_objects(bucket: str, prefix: str = "") -> list[dict[str, Any]]:
    """Read all public object names/sizes using paged JSON listings."""
    objects: list[dict[str, Any]] = []
    marker = ""
    while True:
        query = urllib.parse.urlencode(
            {"format": "json", "prefix": prefix, "marker": marker, "limit": 10000}
        )
        url = f"{BASE_URL}/{bucket}/?{query}"
        status, headers, body = _request(url)
        if status != 200 or headers.get_content_type() != "application/json":
            raise RuntimeError("Swift listing did not return HTTP 200 JSON")
        page = json.loads(body)
        if not page:
            return objects
        if not isinstance(page, list) or not all(
            isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and isinstance(row.get("bytes"), int)
            for row in page
        ):
            raise RuntimeError("unexpected Swift listing schema")
        objects.extend(page)
        next_marker = page[-1]["name"]
        if next_marker == marker:
            raise RuntimeError("Swift listing marker did not advance")
        marker = next_marker


def _canonical_date(date_text: str) -> str:
    return datetime.strptime(date_text, "%m-%d-%y").date().isoformat()


def audio_day_groups(rows: list[dict[str, Any]]) -> set[str]:
    groups: set[str] = set()
    for row in rows:
        match = AUDIO_KEY.fullmatch(row["name"])
        if match:
            groups.add(f"{match['airport']}/{_canonical_date(match['date'])}")
    return groups


def parse_zip_central_names(tail: bytes, *, object_size: int) -> list[str]:
    """Parse only a ZIP central directory present in a bounded range tail."""
    tail_start = max(0, object_size - len(tail))
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0 or eocd + 22 > len(tail):
        raise ValueError("ZIP end-of-central-directory missing from bounded tail")
    disk, cd_disk, disk_count, total_count, cd_size, cd_offset, comment_len = (
        struct.unpack_from("<HHHHIIH", tail, eocd + 4)
    )
    if eocd + 22 + comment_len != len(tail):
        raise ValueError("ZIP comment or trailing bytes do not match EOCD")
    if disk or cd_disk or disk_count != total_count:
        raise ValueError("multi-disk or inconsistent ZIP is unsupported")
    if 0xFFFF in (disk_count, total_count) or 0xFFFFFFFF in (cd_size, cd_offset):
        raise ValueError("ZIP64 central directories are unsupported")
    cd_start = cd_offset - tail_start
    cd_end = cd_start + cd_size
    if cd_start < 0 or cd_end > eocd:
        raise ValueError("central directory is not fully present in bounded tail")

    names: list[str] = []
    cursor = cd_start
    while cursor < cd_end:
        if tail[cursor : cursor + 4] != b"PK\x01\x02":
            raise ValueError("invalid ZIP central-directory entry signature")
        filename_len, extra_len, entry_comment_len = struct.unpack_from(
            "<HHH", tail, cursor + 28
        )
        entry_end = cursor + 46 + filename_len + extra_len + entry_comment_len
        if entry_end > cd_end:
            raise ValueError("truncated ZIP central-directory entry")
        names.append(
            tail[cursor + 46 : cursor + 46 + filename_len].decode(
                "utf-8", errors="replace"
            )
        )
        cursor = entry_end
    if cursor != cd_end or len(names) != total_count:
        raise ValueError("ZIP central-directory entry count/size mismatch")
    return names


def fetch_zip_central_names(row: dict[str, Any]) -> tuple[list[str], int]:
    """Fetch a strict HTTP Range suffix and parse names, never member bytes."""
    size = row["bytes"]
    start = max(0, size - RANGE_BYTES)
    end = size - 1
    url = f"{BASE_URL}/tartanaviation-adsb/{urllib.parse.quote(row['name'])}"
    status, headers, body = _request(
        url, headers={"Range": f"bytes={start}-{end}"}
    )
    expected_range = f"bytes {start}-{end}/{size}"
    if status != 206 or headers.get("Content-Range") != expected_range:
        raise RuntimeError("server did not honor the exact bounded Range request")
    if len(body) != end - start + 1:
        raise RuntimeError("Range response length mismatch")
    return parse_zip_central_names(body, object_size=size), len(body)


def adsb_day_groups(
    object_rows: list[dict[str, Any]],
) -> tuple[set[str], list[dict[str, Any]], int]:
    groups: set[str] = set()
    archives: list[dict[str, Any]] = []
    range_bytes = 0
    for row in object_rows:
        match = ADSB_ARCHIVE.fullmatch(row["name"])
        if not match:
            continue
        names, received = fetch_zip_central_names(row)
        range_bytes += received
        airport = match["airport"]
        archive_dates = {
            f"{airport}/{_canonical_date(date_match['date'])}"
            for name in names
            if (date_match := ADSB_MEMBER_DATE.search(name))
        }
        groups.update(archive_dates)
        archives.append(
            {
                "name": row["name"],
                "object_bytes": row["bytes"],
                "central_directory_member_count": len(names),
                "distinct_airport_date_groups": len(archive_dates),
            }
        )
    return groups, archives, range_bytes


def normalized_listing_hash(rows: list[dict[str, Any]]) -> str:
    normalized = "".join(
        f"{row['name']}\t{row['bytes']}\n"
        for row in sorted(rows, key=lambda row: row["name"])
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def run_audit() -> dict[str, Any]:
    audio_rows = [
        row
        for airport in ("kagc", "kbtp")
        for row in list_objects("tartanaviation-audio", f"{airport}/")
    ]
    adsb_rows = list_objects("tartanaviation-adsb")
    audio_groups = audio_day_groups(audio_rows)
    adsb_groups, archives, range_bytes = adsb_day_groups(adsb_rows)
    intersections = sorted(audio_groups & adsb_groups)
    return {
        "schema_version": 1,
        "source": {
            "repository": "https://github.com/castacks/TartanAviation",
            "revision": "077a04abac1e50953c45b72524edd79e3a21d9bd",
            "audio_bucket": "tartanaviation-audio",
            "adsb_bucket": "tartanaviation-adsb",
            "listing_endpoint": f"{BASE_URL}/<bucket>/?format=json",
        },
        "method": {
            "audio": "Swift JSON object names and byte sizes only",
            "adsb": (
                "Swift JSON object names/sizes plus exact 8 MiB HTTP Range "
                "suffixes; parse ZIP central-directory filenames only"
            ),
            "payload_downloaded": False,
            "archive_extracted": False,
            "audio_or_track_rows_opened": False,
        },
        "counts": {
            "audio_daily_archives": len(audio_groups),
            "audio_daily_archives_by_airport": {
                airport: sum(group.startswith(f"{airport}/") for group in audio_groups)
                for airport in ("kagc", "kbtp")
            },
            "adsb_airport_date_groups": len(adsb_groups),
            "adsb_airport_date_groups_by_airport": {
                airport: sum(group.startswith(f"{airport}/") for group in adsb_groups)
                for airport in ("kagc", "kbtp")
            },
            "audio_adsb_intersection_groups": len(intersections),
            "intersection_groups_by_airport": {
                airport: sum(group.startswith(f"{airport}/") for group in intersections)
                for airport in ("kagc", "kbtp")
            },
            "intersection_airport_date_groups": intersections,
            "source_example_count": None,
            "human_verified_task_example_count": None,
            "independence_after_aircraft_or_speaker_identity_grouping": None,
        },
        "listing_hashes_sha256": {
            "audio": normalized_listing_hash(audio_rows),
            "adsb": normalized_listing_hash(adsb_rows),
        },
        "adsb_archives": archives,
        "range_bytes_received": range_bytes,
        "warning": (
            "Airport-date intersections are only an availability upper bound. "
            "They are not usable-example counts or proven independent groups."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
