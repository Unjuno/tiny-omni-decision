"""Acquire only the preregistered CC0 Clotho publisher-validation audio clips."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

from scripts.clotho_media_identity import media_basename
from scripts.fetch_starss23_smoke import HttpRangeReader
from scripts.run_starss23_ipv4 import IPv4HTTPSHandler

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
ATTEMPT = ROOT / "clotho-aqa-publisher-val-20261126-attempt-01"
MANIFEST = ATTEMPT / "inclusion-manifest.jsonl"
OUTPUT = ATTEMPT / "media"
VAL_SHA256 = "c69b32095d70740cee49d66dc33690d0267cf7bc821d71d834725f9d24c0a491"
MANIFEST_SHA256 = "100630d59a92f461760a83aac1bfd6684d511904a42cf6ea1c71819b0fe5fb19"
METADATA_SHA256 = "7505ba93c291d344b31e5c33e872b95d84255852cff93278bbdf549402acc1f5"
CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
ARCHIVE_URL = "https://zenodo.org/api/records/6473207/files/audio_files.zip/content"
ARCHIVE_SIZE = 3_140_677_494


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    if sha256(ATTEMPT / "clotho_aqa_val.csv") != VAL_SHA256:
        raise ValueError("publisher validation CSV hash mismatch")
    metadata_path = ROOT / "clotho-aqa-admission-01" / "clotho_aqa_metadata.csv"
    if sha256(metadata_path) != METADATA_SHA256:
        raise ValueError("per-clip metadata hash mismatch")
    if sha256(path) != MANIFEST_SHA256:
        raise ValueError("frozen inclusion manifest hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("empty validation inclusion manifest")
    owners, filenames, content = {}, {}, set()
    for row in rows:
        source_id = row["source_id"]
        uploader = row["uploader"]
        identity = (source_id, row["normalized_question"])
        if (
            not source_id.isdecimal()
            or not uploader
            or row["license"] != CC0
            or len(row["votes"]) != 3
            or not set(row["votes"]) <= {"yes", "no"}
            or row["target"] != int(row["votes"].count("yes") >= 2)
            or identity in content
        ):
            raise ValueError("invalid or duplicate preregistered media row")
        if source_id in owners and owners[source_id] != uploader:
            raise ValueError("source has conflicting uploader identities")
        filename = " ".join(row["file_name"].casefold().split())
        if source_id in filenames and filenames[source_id] != filename:
            raise ValueError("source has conflicting media filenames")
        owners[source_id] = uploader
        filenames[source_id] = filename
        content.add(identity)
    return rows


def unique_media_rows(rows: list[dict]) -> list[dict]:
    by_source = {}
    for row in rows:
        previous = by_source.get(row["source_id"])
        if previous is None:
            by_source[row["source_id"]] = row
        elif (
            previous["file_name"] != row["file_name"]
            or previous["uploader"] != row["uploader"]
            or previous["license"] != row["license"]
        ):
            raise ValueError("one audio source has inconsistent identity metadata")
    return [by_source[source] for source in sorted(by_source)]


def acquire(output: Path = OUTPUT) -> dict:
    rows = load_manifest()
    media_rows = unique_media_rows(rows)
    uploaders = {row["uploader"] for row in rows}
    if len(uploaders) < 200:
        # Allowed only as a descriptive one-shot; the result must not be called a quality pass.
        group_scope = "descriptive_only_below_200_independent_uploaders"
    else:
        group_scope = "provisional_200_uploader_floor_met"
    output.mkdir(parents=True, exist_ok=False)
    reader = HttpRangeReader(ARCHIVE_URL, ARCHIVE_SIZE, budget=512 * 1024**2, retries=2)
    report = {
        "status": "RUNNING",
        "scope": group_scope,
        "manifest_sha256": MANIFEST_SHA256,
        "publisher_csv_sha256": VAL_SHA256,
        "metadata_sha256": METADATA_SHA256,
        "selected_questions": len(rows),
        "selected_sources": len(media_rows),
        "selected_uploaders": len(uploaders),
        "archive": {"url": ARCHIVE_URL, "expected_size": ARCHIVE_SIZE},
        "files": [],
        "started_utc_epoch": time.time(),
        "script_sha256": sha256(Path(__file__)),
    }

    def save() -> None:
        report["transferred_bytes"] = reader.transferred
        temporary = output / "result.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output / "result.json")

    save()
    try:
        opener = urllib.request.build_opener(IPv4HTTPSHandler())
        with patch("urllib.request.urlopen", opener.open):
            with zipfile.ZipFile(reader) as archive:
                by_basename = {}
                for info in archive.infolist():
                    is_audio = info.filename.startswith("audio_files/")
                    if is_audio and info.filename.lower().endswith(".wav"):
                        basename = html.unescape(Path(info.filename).name)
                        if basename in by_basename:
                            raise ValueError("duplicate publisher audio basename")
                        by_basename[basename] = info
                for row in media_rows:
                    member = "audio_files/" + media_basename(row)
                    info = archive.getinfo(member)
                    if info.file_size > 16 * 1024**2:
                        raise ValueError("audio member exceeds 16 MiB safety bound")
                    payload = archive.read(info)
                    crc_matches = (zlib.crc32(payload) & 0xFFFFFFFF) == info.CRC
                    if len(payload) != info.file_size or not crc_matches:
                        raise ValueError("publisher archive member CRC/length mismatch")
                    if shutil.disk_usage(output).free < len(payload) + 2 * 1024**3:
                        raise ValueError("2 GiB free-space reserve would be violated")
                    path = output / f"{row['source_id']}.wav"
                    if path.exists():
                        raise FileExistsError(path)
                    path.write_bytes(payload)
                    report["files"].append(
                        {
                            "source_id": row["source_id"],
                            "uploader": row["uploader"],
                            "license": row["license"],
                            "member": info.filename,
                            "bytes": len(payload),
                            "crc32": f"{info.CRC:08x}",
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
                    save()
        expected_sources = {row["source_id"] for row in media_rows}
        actual_sources = {row["source_id"] for row in report["files"]}
        if actual_sources != expected_sources:
            raise ValueError("acquired audio source set differs from frozen manifest")
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_utc_epoch"] = time.time()
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(acquire(args.output), indent=2))


if __name__ == "__main__":
    main()
