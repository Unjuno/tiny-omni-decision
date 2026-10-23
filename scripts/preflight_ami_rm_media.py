"""Fetch the pinned low-bitrate AMI close-up streams for CPU-only preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OFFICIAL_HOST = "groups.inf.ed.ac.uk"
OFFICIAL_ROOT = f"https://{OFFICIAL_HOST}/ami/AMICorpusMirror/amicorpus"
REQUIRED_MEETINGS = {
    "train": ("TS3009a", "ES2010b", "IS1004b"),
    "validation": ("TS3005b", "ES2008b", "IS1006b"),
}
CAMERAS = (1, 2, 3, 4)
CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    size_label: str


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[tuple[str, str, list[str]]] = []
        self._in_row = False
        self._in_cell = False
        self._cells: list[str] = []
        self._cell_parts: list[str] = []
        self._link_href: str | None = None
        self._link_parts: list[str] = []
        self._link_text = ""
        self._in_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        if tag == "tr":
            self._in_row = True
            self._cells = []
            self._link_href = None
            self._link_parts = []
            self._link_text = ""
        elif tag == "td" and self._in_row:
            self._in_cell = True
            self._cell_parts = []
        elif tag == "a" and self._in_row:
            self._link_href = attrs_map.get("href")
            self._link_parts = []
            self._in_link = True

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)
        if self._in_link:
            self._link_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._in_cell:
            self._cells.append(" ".join("".join(self._cell_parts).split()))
            self._in_cell = False
        elif tag == "a" and self._in_link:
            self._link_text = " ".join("".join(self._link_parts).split())
            self._in_link = False
        elif tag == "tr" and self._in_row:
            if self._link_href is not None:
                self.entries.append((self._link_href, self._link_text, self._cells[:]))
            self._in_row = False
            self._in_cell = False
            self._link_href = None


def _is_safe_relative_filename(href: str, label: str) -> bool:
    return (
        href == label
        and "/" not in href
        and "\\" not in href
        and ".." not in href
        and ":" not in href
    )


def parse_directory_index(html: str, meeting: str) -> dict[int, DirectoryEntry]:
    if meeting not in {item for values in REQUIRED_MEETINGS.values() for item in values}:
        raise ValueError(f"meeting is outside the pinned cohort: {meeting}")

    parser = _IndexParser()
    parser.feed(html)
    camera_pattern = re.compile(rf"^{re.escape(meeting)}\.Closeup([1-4])\.rm$")
    found: dict[int, DirectoryEntry] = {}
    for href, label, cells in parser.entries:
        match = camera_pattern.fullmatch(label)
        if match is None:
            continue
        if not _is_safe_relative_filename(href, label):
            raise ValueError(f"unsafe media link for {label}: {href}")
        camera = int(match.group(1))
        if camera in found:
            raise ValueError(f"duplicate camera {camera} in {meeting} index")
        if len(cells) < 4 or not cells[3]:
            raise ValueError(f"missing directory size for {label}")
        found[camera] = DirectoryEntry(name=label, size_label=cells[3])

    if set(found) != set(CAMERAS):
        missing = sorted(set(CAMERAS) - set(found))
        raise ValueError(f"index must list all four close-up RM cameras; missing={missing}")
    return found


def build_stream_url(meeting: str, camera: int) -> str:
    if meeting not in {item for values in REQUIRED_MEETINGS.values() for item in values}:
        raise ValueError(f"meeting is outside the pinned cohort: {meeting}")
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}: {camera}")
    filename = f"{meeting}.Closeup{camera}.rm"
    return f"{OFFICIAL_ROOT}/{meeting}/video/{filename}"


def _open_official(url: str, method: str = "GET") -> BinaryIO:
    request = Request(url, method=method, headers={"User-Agent": "tiny-omni-decision-r2-preflight"})
    response = urlopen(request, timeout=45)
    if response.status != 200:
        response.close()
        raise OSError(f"official source returned HTTP {response.status}: {url}")
    if response.geturl().split("/", 3)[2].lower() != OFFICIAL_HOST:
        response.close()
        raise OSError(f"unexpected media host after redirect: {response.geturl()}")
    return response


def _read_index(meeting: str) -> str:
    url = f"{OFFICIAL_ROOT}/{meeting}/video/"
    with _open_official(url) as response:
        return response.read().decode("utf-8", errors="strict")


def _content_length(meeting: str, camera: int) -> int:
    url = build_stream_url(meeting, camera)
    with _open_official(url, method="HEAD") as response:
        value = response.headers.get("Content-Length")
    if value is None or not value.isdigit() or int(value) <= 0:
        raise OSError(f"missing valid Content-Length for {url}")
    return int(value)


def write_verified_stream(
    source: BinaryIO, destination: Path, expected_bytes: int
) -> str:
    """Write a new file only when its body length matches the trusted HEAD size."""
    if destination.exists():
        raise FileExistsError(destination)
    if expected_bytes <= 0:
        raise ValueError("expected byte count must be positive")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)

    digest = hashlib.sha256()
    count = 0
    try:
        with partial.open("xb") as output:
            while chunk := source.read(CHUNK_BYTES):
                count += len(chunk)
                if count > expected_bytes:
                    raise ValueError("download exceeds declared byte count")
                digest.update(chunk)
                output.write(chunk)
        if count != expected_bytes:
            raise ValueError(
                f"download byte count {count} does not match declared {expected_bytes}"
            )
        partial.rename(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def download_cohort(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    files: list[dict[str, object]] = []
    for split, meetings in REQUIRED_MEETINGS.items():
        for meeting in meetings:
            entries = parse_directory_index(_read_index(meeting), meeting)
            for camera in CAMERAS:
                url = build_stream_url(meeting, camera)
                expected_bytes = _content_length(meeting, camera)
                destination = output_dir / f"{meeting}.Closeup{camera}.rm"
                with _open_official(url) as response:
                    digest = write_verified_stream(response, destination, expected_bytes)
                files.append(
                    {
                        "split": split,
                        "meeting_id": meeting,
                        "camera": camera,
                        "filename": destination.name,
                        "source_url": url,
                        "index_size_label": entries[camera].size_label,
                        "content_length_bytes": expected_bytes,
                        "sha256": digest,
                    }
                )

    manifest = {
        "schema_version": 1,
        "captured_utc": datetime.now(UTC).isoformat(),
        "source": "AMI official mirror",
        "source_root": OFFICIAL_ROOT,
        "ami_audio_metadata_revision": "46f28f2503e2ec48f8867a84eef356c70476beab",
        "experiment_split_by_meeting": REQUIRED_MEETINGS,
        "camera_stream_count": len(files),
        "media_downloaded": True,
        "heldout_or_sealed_data_accessed": False,
        "files": files,
    }
    manifest_path = output_dir / "media-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new, nonexistent external artifact directory; existing files are never overwritten",
    )
    args = parser.parse_args()
    try:
        manifest = download_cohort(args.output_dir)
    except (OSError, HTTPError, URLError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "camera_stream_count": manifest["camera_stream_count"],
                "manifest": str(args.output_dir / "media-manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
