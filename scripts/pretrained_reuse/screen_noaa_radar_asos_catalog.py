"""Read public NOAA catalog metadata for a fixed ASOS/MRMS screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ASOS_INDEX_TEMPLATE = (
    "https://www.ncei.noaa.gov/data/automated-surface-observing-system-"
    "one-minute-pg2/access/{year}/{month:02d}/"
)
MRMS_BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com/"
MRMS_PRODUCT = "CONUS/RadarOnly_QPE_01H_00.00"
USER_AGENT = "tiny-omni-decision-metadata-screen/1.0"
S3_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


class _DirectoryIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, str]]] = []
        self._row: list[dict[str, str]] | None = None
        self._cell: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = {"text": "", **attributes}
        elif tag == "a" and self._cell is not None:
            self._cell["href"] = attributes.get("href", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._cell is not None and self._row is not None:
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"] += data


def parse_asos_index(html: str, *, year: int, month: int) -> list[dict[str, Any]]:
    parser = _DirectoryIndexParser()
    parser.feed(html)
    pattern = re.compile(rf"asos-1min-pg2-([^-]+)-{year}{month:02d}\.dat\Z")
    rows: list[dict[str, Any]] = []
    for cells in parser.rows:
        if not cells:
            continue
        href = cells[0].get("href", "")
        match = pattern.fullmatch(href)
        if not match:
            continue
        size_text = next(
            (cell["text"].strip() for cell in cells[1:]
             if cell.get("align", "").lower() == "right"
             and cell["text"].strip().isdigit()),
            None,
        )
        if size_text is None:
            raise ValueError(f"ASOS index row has no numeric byte size: {href}")
        rows.append({"station_id": match.group(1), "filename": href,
                     "listed_bytes": int(size_text)})
    if not rows:
        raise ValueError("ASOS directory index contains no matching station files")
    station_ids = [row["station_id"] for row in rows]
    if len(station_ids) != len(set(station_ids)):
        raise ValueError("ASOS index contains duplicate station-month files")
    return sorted(rows, key=lambda row: row["station_id"])


def parse_mrms_listing(xml_bytes: bytes, *, date: str) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    truncated = root.findtext("s3:IsTruncated", namespaces=S3_NAMESPACE)
    if truncated != "false":
        raise ValueError("MRMS S3 listing is truncated or lacks a truncation flag")
    contents = root.findall("s3:Contents", S3_NAMESPACE)
    if not contents:
        raise ValueError("MRMS S3 listing has no objects")
    pattern = re.compile(rf"_{date.replace('-', '')}-(\d{{6}})\.grib2\.gz\Z")
    objects: list[dict[str, Any]] = []
    for item in contents:
        key = item.findtext("s3:Key", namespaces=S3_NAMESPACE) or ""
        size = item.findtext("s3:Size", namespaces=S3_NAMESPACE)
        match = pattern.search(key)
        if not match or size is None or not size.isdigit():
            raise ValueError(f"Unexpected MRMS object metadata: {key}")
        objects.append({"key": key, "time_utc": match.group(1), "listed_bytes": int(size)})
    keys = [row["key"] for row in objects]
    timestamps = [row["time_utc"] for row in objects]
    if len(keys) != len(set(keys)) or len(timestamps) != len(set(timestamps)):
        raise ValueError("MRMS listing contains duplicate object keys or timestamps")
    return {
        "date": date,
        "object_count": len(objects),
        "unique_timestamp_count": len(set(timestamps)),
        "listed_compressed_payload_bytes": sum(row["listed_bytes"] for row in objects),
        "min_object_bytes": min(row["listed_bytes"] for row in objects),
        "max_object_bytes": max(row["listed_bytes"] for row in objects),
        "objects": objects,
    }


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"metadata request failed with HTTP {response.status}: {url}")
        return response.read()


def _save_receipt(output: Path, filename: str, url: str, body: bytes) -> dict[str, Any]:
    path = output / filename
    path.write_bytes(body)
    return {
        "filename": filename,
        "url": url,
        "response_body_bytes": len(body),
        "response_body_sha256": hashlib.sha256(body).hexdigest(),
    }


def run_screen(*, output: Path, year: int = 2024, month: int = 5,
               day: int = 1) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if not 2000 <= year <= 2100:
        raise ValueError("invalid catalog date")
    try:
        datetime(year, month, day)
    except ValueError:
        raise ValueError("invalid catalog date") from None
    output.mkdir(parents=True)
    asos_url = ASOS_INDEX_TEMPLATE.format(year=year, month=month)
    radar_date = f"{year:04d}-{month:02d}-{day:02d}"
    prefix = f"{MRMS_PRODUCT}/{radar_date.replace('-', '')}/"
    radar_url = f"{MRMS_BUCKET_URL}?list-type=2&prefix={urllib.parse.quote(prefix, safe='/')}"
    asos_body = _fetch(asos_url)
    radar_body = _fetch(radar_url)
    asos_receipt = _save_receipt(output, "asos-directory-index.html", asos_url, asos_body)
    radar_receipt = _save_receipt(output, "mrms-object-list.xml", radar_url, radar_body)
    asos_rows = parse_asos_index(asos_body.decode("utf-8", errors="strict"),
                                 year=year, month=month)
    mrms = parse_mrms_listing(radar_body, date=radar_date)
    result = {
        "format": "noaa_radar_asos_catalog_screen_v1",
        "status": "metadata_only_not_admitted",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "sources": [asos_receipt, radar_receipt],
        "asos": {
            "year": year,
            "month": month,
            "station_month_file_count": len(asos_rows),
            "unique_station_id_count": len({row["station_id"] for row in asos_rows}),
            "listed_monthly_payload_bytes": sum(row["listed_bytes"] for row in asos_rows),
            "sample_station_ids": [row["station_id"] for row in asos_rows[:10]],
        },
        "mrms": {
            "product": MRMS_PRODUCT,
            "product_type": "rolling 1-hour radar-only accumulation, cataloged every 2 minutes",
            "listing": {key: value for key, value in mrms.items() if key != "objects"},
            "first_keys": [row["key"] for row in mrms["objects"][:3]],
            "last_keys": [row["key"] for row in mrms["objects"][-3:]],
        },
        "payload_objects_downloaded": 0,
        "media_or_observations_opened": False,
        "independent_storm_groups_established": False,
        "task_admitted": False,
        "limitations": [
            "Station-month file count is not an independent station or storm count.",
            "MRMS and ASOS directory metadata do not establish exact station/grid/time matching.",
            (
                "The proposed radar-versus-gauge three-way comparison can be computed by a "
                "deterministic rule and is not a meaningful learned-Decision quality test "
                "by itself."
            ),
            (
                "ASOS metadata identifies NOAA, FAA and DOD as originators and provides "
                "citation/use-liability conditions, but an explicit downstream license grant "
                "for this mixed-source archive was not established."
            ),
        ],
    }
    (output / "screen.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=5)
    parser.add_argument("--day", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_screen(output=args.output, year=args.year,
                                month=args.month, day=args.day), indent=2))


if __name__ == "__main__":
    main()
