"""Prepare only frozen train/validation media from the SoundActions v2 archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import TextIO

from scripts.freeze_soundactions_split import MEDIA_PATH

LABELS_MEMBER = "Labels/SoundActions_labels_master.csv"
CLASSES = ("impulsive", "sustained", "iterative")
ARCHIVE_MD5 = {
    "sound-actions-audio.zip": "61a43a32e1ea8a69399a82ef7e25bcec",
    "sound-actions-video.zip": "300fd4c64d367482bbbff59ab6b1d1ca",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_split_ids(path: Path) -> dict[str, set[int]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_digest = manifest.pop("manifest_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    actual_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if expected_digest != actual_digest:
        raise ValueError("frozen split manifest hash mismatch")
    split_rows = manifest.get("splits", {})
    train = set(split_rows.get("train", []))
    validation = set(split_rows.get("validation", []))
    sealed = set(split_rows.get("sealed_audit", []))
    if not train or not validation or train & validation or (train | validation) & sealed:
        raise ValueError("frozen split IDs are missing or overlap")
    return {"train": train, "validation": validation, "sealed_audit": sealed}


def selected_csv_rows(stream: TextIO, allowed_ids: set[int]) -> dict[int, dict[str, str]]:
    """Parse allowed CSV rows; discard excluded labels without retaining whole rows."""
    header_line = stream.readline()
    if not header_line:
        raise ValueError("CSV is empty")
    header = next(csv.reader([header_line]))
    if not header or header[0] != "number":
        raise ValueError("CSV does not start with the numeric recording ID")
    selected: dict[int, dict[str, str]] = {}
    line_number = 1
    while True:
        first_chars = []
        while True:
            char = stream.read(1)
            if not char:
                if first_chars:
                    raise ValueError(f"CSV line {line_number + 1} has no column delimiter")
                return selected
            if char == ",":
                break
            if char in {"\n", "\r"}:
                raise ValueError(f"CSV line {line_number + 1} has no first-column delimiter")
            first_chars.append(char)
        line_number += 1
        first_field = "".join(first_chars).strip().strip('"')
        try:
            record_id = int(first_field)
        except ValueError as error:
            raise ValueError(f"invalid first-column ID at CSV line {line_number}") from error
        if record_id not in allowed_ids:
            while char and char != "\n":
                char = stream.read(1)
            continue
        cells = next(csv.reader([first_field + "," + stream.readline()]))
        if len(cells) != len(header):
            raise ValueError(
                f"selected CSV row has an unexpected column count at line {line_number}"
            )
        if record_id in selected:
            raise ValueError(f"duplicate selected CSV ID: {record_id}")
        selected[record_id] = dict(zip(header, cells, strict=True))
    return selected


def _load_rows(overview_path: Path, labels_zip_path: Path, split_ids: dict[str, set[int]]):
    allowed_ids = split_ids["train"] | split_ids["validation"]
    with overview_path.open(encoding="utf-8-sig", newline="") as stream:
        overview = selected_csv_rows(stream, allowed_ids)
    with zipfile.ZipFile(labels_zip_path) as archive:
        with archive.open(LABELS_MEMBER) as raw:
            text_stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            labels = selected_csv_rows(text_stream, allowed_ids)
    if set(overview) != allowed_ids or set(labels) != allowed_ids:
        raise ValueError("overview or label master is missing an allowed split ID")

    result = []
    excluded_unknown = {"train": 0, "validation": 0}
    for split_name in ("train", "validation"):
        for record_id in sorted(split_ids[split_name]):
            overview_row = overview[record_id]
            label_row = labels[record_id]
            if int(overview_row["number"]) != record_id or int(label_row["number"]) != record_id:
                raise ValueError(f"CSV row ID mismatch for recording {record_id}")
            if overview_row["perception_type"] != label_row["perception_type"]:
                raise ValueError(f"overview and label-master consensus differ for {record_id}")
            label = label_row["perception_type"].strip().lower()
            if label not in CLASSES:
                if label != "unknown":
                    raise ValueError(f"unrecognized consensus label for recording {record_id}")
                excluded_unknown[split_name] += 1
                continue
            paths = {"wav": overview_row["wav"], "mp4": overview_row["mp4"]}
            for modality, name in paths.items():
                match = MEDIA_PATH.fullmatch(f"{'WAV' if modality == 'wav' else 'MP4'}/{name}")
                if not match or int(match.group(2)) != record_id:
                    raise ValueError(f"invalid media path for recording {record_id}")
            result.append(
                {
                    "recording_id": record_id,
                    "split": split_name,
                    "target": label,
                    "wav_member": f"WAV/{paths['wav']}",
                    "mp4_member": f"MP4/{paths['mp4']}",
                }
            )
    return result, excluded_unknown


def _publisher_hashes(checksums_path: Path, allowed_ids: set[int]) -> dict[tuple[str, int], str]:
    hashes = {}
    for line_number, line in enumerate(checksums_path.open(encoding="utf-8"), start=1):
        digest, relative_path = line.rstrip("\n").split("  ", maxsplit=1)
        match = MEDIA_PATH.fullmatch(relative_path)
        if not match:
            continue
        modality, raw_id, _ = match.groups()
        if modality == "JPG":
            continue
        record_id = int(raw_id)
        if record_id not in allowed_ids:
            continue
        key = (modality, record_id)
        if key in hashes or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid or duplicate publisher checksum at line {line_number}")
        hashes[key] = digest
    expected = {(modality, record_id) for record_id in allowed_ids for modality in ("WAV", "MP4")}
    if set(hashes) != expected:
        raise ValueError("publisher checksum list is incomplete for allowed recordings")
    return hashes


def _verify_archive(path: Path, expected_md5: str, receipt: dict[str, object]) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt_rows = {row["name"]: row for row in receipt.get("archives", [])}
    row = receipt_rows.get(path.name)
    if row is None or row.get("status") != "PASS" or row.get("md5") != expected_md5:
        raise ValueError(f"archive lacks a successful official checksum receipt: {path.name}")
    if path.stat().st_size != row.get("bytes"):
        raise ValueError(f"archive byte count differs from receipt: {path.name}")
    return sha256_file(path)


def _extract_verified_member(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    if destination.exists():
        actual = sha256_file(destination)
        if destination.stat().st_size == expected_bytes and actual == expected_sha256:
            return
        raise FileExistsError(
            f"staged media exists but does not match publisher hash: {destination}"
        )

    temporary: Path | None = None
    digest = hashlib.sha256()
    written = 0
    with archive.open(member) as source:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=destination.name + ".",
            suffix=".partial",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for block in iter(lambda: source.read(1 << 20), b""):
                output.write(block)
                digest.update(block)
                written += len(block)
            output.flush()
            os.fsync(output.fileno())

    if written != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError(f"publisher media checksum mismatch: {member}")
    os.replace(temporary, destination)


def prepare_media(
    *,
    metadata_dir: Path,
    split_manifest: Path,
    audio_archive: Path,
    video_archive: Path,
    output: Path,
    resume_partial: bool = False,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite media output: {output}")
    started = time.perf_counter()
    split_ids = _manifest_split_ids(split_manifest)
    rows, excluded_unknown = _load_rows(
        metadata_dir / "overview.csv", metadata_dir / "sound-actions-labels.zip", split_ids
    )
    allowed_ids = split_ids["train"] | split_ids["validation"]
    publisher_hashes = _publisher_hashes(metadata_dir / "SHA256SUMS.txt", allowed_ids)
    receipt_path = audio_archive.parent / "archive-download-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    archive_hashes = {
        audio_archive.name: _verify_archive(
            audio_archive, ARCHIVE_MD5[audio_archive.name], receipt
        ),
        video_archive.name: _verify_archive(
            video_archive, ARCHIVE_MD5[video_archive.name], receipt
        ),
    }

    staging = output.with_name(output.name + ".partial")
    if staging.exists() and not resume_partial:
        raise FileExistsError(f"refusing to overwrite partial media output: {staging}")
    if staging.exists() and not staging.is_dir():
        raise NotADirectoryError(staging)
    staging.mkdir(parents=True, exist_ok=True)
    media_rows = []
    with zipfile.ZipFile(audio_archive) as audio_zip, zipfile.ZipFile(video_archive) as video_zip:
        for row in rows:
            record_id = row["recording_id"]
            split_name = row["split"]
            row_out = {
                key: value for key, value in row.items() if key not in {"wav_member", "mp4_member"}
            }
            for modality, archive, member_key, extension, publisher_modality in (
                ("audio", audio_zip, "wav_member", ".wav", "WAV"),
                ("video", video_zip, "mp4_member", ".mp4", "MP4"),
            ):
                member = row[member_key]
                info = archive.getinfo(member)
                if info.file_size > 512 * 1024 * 1024:
                    raise ValueError(f"media member is unexpectedly large: {member}")
                expected_hash = publisher_hashes[(publisher_modality, record_id)]
                relative = Path("media") / split_name / modality / f"{record_id:03d}{extension}"
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                _extract_verified_member(
                    archive,
                    member,
                    destination,
                    expected_hash,
                    info.file_size,
                )
                actual_hash = expected_hash
                row_out[f"{modality}_path"] = relative.as_posix()
                row_out[f"{modality}_sha256"] = actual_hash
                row_out[f"{modality}_bytes"] = info.file_size
            media_rows.append(row_out)

    with (staging / "train-validation.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in media_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    counts = {
        split_name: sum(row["split"] == split_name for row in media_rows)
        for split_name in ("train", "validation")
    }
    report = {
        "status": "PASS_TRAIN_VALIDATION_ONLY",
        "source_record": "zenodo:23971067",
        "source_version": 2,
        "split_manifest_sha256": sha256_file(split_manifest),
        "archive_sha256": archive_hashes,
        "publisher_checksums_sha256": sha256_file(metadata_dir / "SHA256SUMS.txt"),
        "counts_with_majority_label": counts,
        "unknown_excluded_by_split": excluded_unknown,
        "media_files_extracted": len(media_rows) * 2,
        "sealed_audit_read_or_extracted": False,
        "resumed_partial_staging": resume_partial,
        "inputs_exposed_to_model": ["audio", "video"],
        "excluded_metadata_fields": ["title", "action", "objects", "materials", "environment"],
        "wall_seconds": time.perf_counter() - started,
    }
    (staging / "acquisition-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--video-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-partial", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_media(
                metadata_dir=args.metadata_dir,
                split_manifest=args.split_manifest,
                audio_archive=args.audio_archive,
                video_archive=args.video_archive,
                output=args.output,
                resume_partial=args.resume_partial,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
