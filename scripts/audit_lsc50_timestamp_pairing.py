"""Audit LSC50 timestamp and video archive metadata without opening media."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
EXPECTED_HEADERS = ("N", "Sign", "Time Start (s)", "Time End (s)")
VIDEO_NAME = re.compile(r"^(\d{4})_(\d{4})_(\d{4})\.avi$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workbook_sheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
    sheets = []
    for sheet in workbook.findall(f"{{{MAIN_NS}}}sheets/{{{MAIN_NS}}}sheet"):
        relation_id = sheet.attrib[f"{{{REL_NS}}}id"]
        target = targets[relation_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets.append((sheet.attrib["name"], target))
    return sheets


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(text.text or "" for text in item.findall(f".//{{{MAIN_NS}}}t"))
        for item in root.findall(f"{{{MAIN_NS}}}si")
    ]


def _worksheet_rows(
    archive: zipfile.ZipFile, target: str, shared: list[str]
) -> list[dict[str, str]]:
    root = ET.fromstring(archive.read(target))
    rows: list[dict[str, str]] = []
    for row in root.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"):
        values: dict[str, str] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            column_match = re.match(r"[A-Z]+", cell.attrib.get("r", ""))
            if column_match is None:
                raise ValueError(f"invalid cell reference in {target}")
            value_node = cell.find(f"{{{MAIN_NS}}}v")
            value = value_node.text or "" if value_node is not None else ""
            cell_type = cell.attrib.get("t")
            if cell_type == "s":
                value = shared[int(value)]
            elif cell_type == "inlineStr":
                value = "".join(text.text or "" for text in cell.findall(f".//{{{MAIN_NS}}}t"))
            values[column_match.group()] = value
        rows.append(values)
    return rows


def _audit_timestamp_sheet(name: str, rows: list[dict[str, str]]) -> dict:
    if not rows or tuple(rows[0].get(column, "") for column in "ABCD") != EXPECTED_HEADERS:
        raise ValueError(f"unexpected timestamp columns in sheet {name!r}")
    records = rows[1:]
    ids: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    invalid_ids: list[int] = []
    for row in records:
        try:
            record_id = int(row["A"])
            start = float(row["C"])
            end = float(row["D"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed timestamp row in sheet {name!r}") from error
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            invalid_ids.append(record_id)
        ids.append(record_id)
        starts.append(start)
        ends.append(end)
    durations = [end - start for start, end in zip(starts, ends, strict=True) if end > start]
    return {
        "sheet": name,
        "sheet_volunteer_number": _volunteer_number(name),
        "row_count": len(records),
        "unique_N_count": len(set(ids)),
        "N_min": min(ids) if ids else None,
        "N_max": max(ids) if ids else None,
        "duplicate_N_count": len(ids) - len(set(ids)),
        "invalid_interval_count": len(invalid_ids),
        "invalid_interval_N": invalid_ids,
        "start_times_monotonic_in_sheet_order": all(
            left <= right for left, right in zip(starts, starts[1:], strict=False)
        ),
        "valid_duration_min_seconds": min(durations) if durations else None,
        "valid_duration_max_seconds": max(durations) if durations else None,
    }


def _volunteer_number(sheet_name: str) -> int:
    match = re.fullmatch(r"Volunteer\s+(\d+)", sheet_name)
    if match is None:
        raise ValueError(f"unrecognized participant sheet name {sheet_name!r}")
    return int(match.group(1))


def audit_lsc50_pairing(timestamp_xlsx: Path, video_inventory_json: Path) -> dict:
    timestamp_xlsx = Path(timestamp_xlsx)
    video_inventory_json = Path(video_inventory_json)
    with zipfile.ZipFile(timestamp_xlsx) as archive:
        shared = _shared_strings(archive)
        sheets = [
            _audit_timestamp_sheet(name, _worksheet_rows(archive, target, shared))
            for name, target in _workbook_sheets(archive)
        ]

    entries = json.loads(video_inventory_json.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("video inventory must be a JSON list")
    triplets_by_view: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    triplet_counts: Counter[tuple[str, str, str]] = Counter()
    video_count = 0
    for entry in entries:
        name = entry.get("name", "")
        if not name.lower().endswith(".avi"):
            continue
        match = VIDEO_NAME.fullmatch(PurePosixPath(name).name)
        if match is None:
            raise ValueError(f"unrecognized video member name {name!r}")
        triplet = match.groups()
        view = PurePosixPath(name).parent.name
        triplets_by_view[view].add(triplet)
        triplet_counts[triplet] += 1
        video_count += 1

    if not triplet_counts:
        raise ValueError("video inventory contains no AVI members")
    sign_volunteer_counts = Counter((sign, volunteer) for sign, volunteer, _ in triplet_counts)
    repetitions_per_pair = Counter(sign_volunteer_counts.values())
    expected_views = set(triplets_by_view)
    all_triplets = set(triplet_counts)
    complete_view_triplets = (
        set.intersection(*triplets_by_view.values()) if triplets_by_view else set()
    )
    return {
        "timestamp_xlsx_sha256": _sha256(timestamp_xlsx),
        "video_inventory_sha256": _sha256(video_inventory_json),
        "timestamp_sheet_count": len(sheets),
        "timestamp_row_count": sum(sheet["row_count"] for sheet in sheets),
        "timestamp_invalid_interval_count": sum(
            sheet["invalid_interval_count"] for sheet in sheets
        ),
        "timestamp_sheets": sheets,
        "video_avi_count": video_count,
        "video_view_directories": sorted(expected_views),
        "video_unique_sign_volunteer_repetition_triplets": len(all_triplets),
        "video_unique_sign_volunteer_pairs": len(sign_volunteer_counts),
        "video_repetition_count_histogram": {
            str(count): n for count, n in sorted(repetitions_per_pair.items())
        },
        "video_participant_id_values": sorted({triplet[1] for triplet in all_triplets}),
        "video_sign_id_min": min(triplet[0] for triplet in all_triplets),
        "video_sign_id_max": max(triplet[0] for triplet in all_triplets),
        "complete_view_triplet_count": len(complete_view_triplets),
        "timestamp_participant_id_convention": (
            "1-based sheet names; mapping to video IDs unresolved"
        ),
        "video_repetition_to_timestamp_mapping": "unresolved from metadata inspected",
        "media_payload_opened": False,
        "sensor_signal_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamps-xlsx", type=Path, required=True)
    parser.add_argument("--video-inventory-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit_lsc50_pairing(args.timestamps_xlsx, args.video_inventory_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
