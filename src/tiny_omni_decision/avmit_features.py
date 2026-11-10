"""Whitelisted, hash-checked caching of non-sealed AVMIT TFRecord features."""

from __future__ import annotations

import hashlib
import json
import tarfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from .avmit_metadata import canonical_jsonl_bytes
from .avmit_tfrecord import (
    decode_float32_steps,
    iter_tfrecord_payloads,
    parse_avmit_sequence_example,
)

ALLOWED_SPLITS = ("train", "calibration", "validation")
EXPECTED_AVMIT_ARCHIVE_BYTES = 2_063_124_480
EXPECTED_AVMIT_ARCHIVE_MD5 = "45c51034fb418eb0bbfdae2d17afb29b"


def normalize_archive_path(value: str) -> str:
    """Normalize a publisher-relative path and reject absolute/traversal paths."""
    value = unicodedata.normalize("NFC", value.strip().replace("\\", "/"))
    if not value or value.startswith("/") or ":" in value:
        raise ValueError("feature record must be a nonempty relative path")
    parts = [part for part in value.split("/") if part not in {"", "."}]
    if not parts or ".." in parts:
        raise ValueError("feature record cannot contain parent traversal")
    return "/".join(parts).casefold()


def _pair_source_allowlist(
    pair_manifest_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, Any]]]:
    if not pair_manifest_dir.is_dir():
        raise ValueError("pair manifest directory must exist")
    by_split: dict[str, dict[str, str]] = {}
    record_to_asset: dict[str, str] = {}
    asset_to_split: dict[str, str] = {}
    manifest_info: dict[str, dict[str, Any]] = {}
    for split in ALLOWED_SPLITS:
        path = pair_manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"required non-sealed pair manifest is missing: {split}")
        records: dict[str, str] = {}
        payload = path.read_bytes()
        lines = payload.decode("utf-8").splitlines()
        manifest_info[split] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": sum(bool(line.strip()) for line in lines),
        }
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                raise ValueError(f"pair row split mismatch at {split}:{line_number}")
            for role in ("audio", "video"):
                asset_id = row.get(f"{role}_asset_id_sha256")
                feature_record = row.get(f"{role}_feature_record")
                if not isinstance(asset_id, str) or len(asset_id) != 64:
                    raise ValueError(f"invalid {role} asset identity at {split}:{line_number}")
                try:
                    int(asset_id, 16)
                except ValueError as error:
                    raise ValueError(
                        f"invalid {role} asset identity at {split}:{line_number}"
                    ) from error
                normalized = normalize_archive_path(feature_record)
                previous_split = asset_to_split.setdefault(asset_id, split)
                if previous_split != split:
                    raise ValueError("a source asset appears in more than one split")
                if normalized in records and records[normalized] != asset_id:
                    raise ValueError("one feature record maps to multiple source identities")
                other_split_asset = record_to_asset.get(normalized)
                if other_split_asset is not None and other_split_asset != asset_id:
                    raise ValueError("feature record identity maps inconsistently across splits")
                records[normalized] = asset_id
                record_to_asset[normalized] = asset_id
        if not records:
            raise ValueError(f"non-sealed pair manifest is empty: {split}")
        by_split[split] = records
    split_by_record: dict[str, dict[str, str]] = {}
    for split, records in by_split.items():
        for record, asset_id in records.items():
            if record in split_by_record:
                raise ValueError("a feature record appears in more than one split")
            split_by_record[record] = {"split": split, "asset_id_sha256": asset_id}
    return by_split, record_to_asset, manifest_info


def _archive_member_map(archive: tarfile.TarFile, required: set[str]) -> dict[str, tarfile.TarInfo]:
    suffix_lookup: dict[tuple[str, ...], set[str]] = {}
    for record in required:
        components = tuple(record.split("/"))
        for start in range(len(components)):
            suffix_lookup.setdefault(components[start:], set()).add(record)

    matched: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        name = member.name.replace("\\", "/")
        components = tuple(part.casefold() for part in name.split("/") if part not in {"", "."})
        record_matches: set[str] = set()
        for start in range(len(components)):
            candidates = suffix_lookup.get(components[start:])
            if candidates:
                record_matches = candidates
                break
        if not record_matches:
            continue
        if len(record_matches) != 1:
            raise ValueError(
                f"archive path ambiguously matches required feature records: {member.name}"
            )
        record = next(iter(record_matches))
        if record in matched:
            raise ValueError(f"archive contains duplicate feature record: {record}")
        if not member.isfile():
            raise ValueError(f"required feature record is not a regular file: {member.name}")
        matched[record] = member

    missing = required - set(matched)
    if missing:
        preview = sorted(missing)[:5]
        raise ValueError(f"archive is missing {len(missing)} required records: {preview}")
    return matched


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_and_cache_avmit_features(
    archive_path: Path,
    pair_manifest_dir: Path,
    output_dir: Path,
    *,
    expected_archive_sha256: str,
    expected_archive_size: int = EXPECTED_AVMIT_ARCHIVE_BYTES,
    expected_archive_md5: str = EXPECTED_AVMIT_ARCHIVE_MD5,
) -> dict[str, Any]:
    """Verify the licensed feature tar and cache only train/calibration/validation records.

    The function never opens a sealed manifest and never writes archive member
    paths directly to disk. It refuses an existing output directory and leaves
    a `.partial` directory for diagnosis if any integrity check fails.
    """
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature cache: {output_dir}")
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite partial cache: {partial_dir}")
    if archive_path.stat().st_size != expected_archive_size:
        raise ValueError("AVMIT feature archive byte size differs from the pinned Zenodo record")
    archive_sha256 = _hash_file(archive_path, "sha256")
    if archive_sha256.casefold() != expected_archive_sha256.casefold():
        raise ValueError("AVMIT feature archive SHA-256 mismatch")
    archive_md5 = _hash_file(archive_path, "md5")
    if archive_md5.casefold() != expected_archive_md5.casefold():
        raise ValueError("AVMIT feature archive MD5 mismatch")

    by_split, record_to_asset, manifest_info = _pair_source_allowlist(pair_manifest_dir)
    partial_dir.mkdir(parents=True, exist_ok=False)
    widths: dict[str, int] = {}
    reports: dict[str, Any] = {}

    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            member_map = _archive_member_map(archive, set(record_to_asset))
            for split in ALLOWED_SPLITS:
                split_dir = partial_dir / split
                split_dir.mkdir()
                index_rows = []
                offsets = {"audio": 0, "video": 0}
                paths = {name: split_dir / f"{name}.f32" for name in offsets}
                with paths["audio"].open("wb") as audio_out, paths["video"].open("wb") as video_out:
                    for record in sorted(by_split[split]):
                        member = member_map[record]
                        record_file = archive.extractfile(member)
                        if record_file is None:
                            raise ValueError(f"cannot read selected feature record: {record}")
                        with record_file:
                            payloads = list(iter_tfrecord_payloads(record_file))
                        if len(payloads) != 1:
                            raise ValueError(f"expected exactly one SequenceExample in {record}")
                        parsed = parse_avmit_sequence_example(payloads[0])
                        embedded = normalize_archive_path(str(parsed["filename"]))
                        expected_stem = str(PurePosixPath(record).with_suffix(""))
                        embedded_stem = str(PurePosixPath(embedded).with_suffix(""))
                        if expected_stem.casefold() != embedded_stem.casefold():
                            raise ValueError(
                                f"embedded filename does not match feature record: {record}"
                            )
                        audio = decode_float32_steps(parsed["audio"])
                        video = decode_float32_steps(parsed["visual_object"])
                        for name, values in (("audio", audio), ("video", video)):
                            width = int(values.shape[1])
                            previous = widths.setdefault(name, width)
                            if previous != width:
                                raise ValueError(f"{name} feature width changes across records")

                        audio_out.write(audio.astype("<f4", copy=False).tobytes(order="C"))
                        video_out.write(video.astype("<f4", copy=False).tobytes(order="C"))
                        index_rows.append(
                            {
                                "asset_id_sha256": record_to_asset[record],
                                "feature_record": record,
                                "audio_offset_bytes": offsets["audio"],
                                "audio_steps": int(audio.shape[0]),
                                "audio_width": int(audio.shape[1]),
                                "video_offset_bytes": offsets["video"],
                                "video_steps": int(video.shape[0]),
                                "video_width": int(video.shape[1]),
                            }
                        )
                        offsets["audio"] += int(audio.nbytes)
                        offsets["video"] += int(video.nbytes)
                index_path = split_dir / "index.jsonl"
                index_path.write_bytes(canonical_jsonl_bytes(index_rows))
                split_files = [*paths.values(), index_path]
                reports[split] = {
                    "unique_source_clips": len(index_rows),
                    "pair_manifest_sha256": manifest_info[split]["sha256"],
                    "pair_manifest_rows": manifest_info[split]["rows"],
                    "feature_record_count": len(index_rows),
                    "audio_width": widths["audio"],
                    "video_width": widths["video"],
                    "audio_steps_total": sum(row["audio_steps"] for row in index_rows),
                    "video_steps_total": sum(row["video_steps"] for row in index_rows),
                    "files": {
                        file.name: {
                            "bytes": file.stat().st_size,
                            "sha256": _hash_file(file, "sha256"),
                        }
                        for file in split_files
                    },
                }

        report = {
            "source": "Audiovisual Moments in Time (AVMIT)",
            "source_revision": "e3c06ec0e215b1934d141f86008e92e567366d6c",
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha256,
            "archive_md5": archive_md5,
            "sealed_payloads_opened": 0,
            "raw_media_opened": 0,
            "feature_cache_splits": list(ALLOWED_SPLITS),
            "feature_cache": reports,
        }
        report_path = partial_dir / "cache-report.json"
        report_path.write_bytes(
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        partial_dir.rename(output_dir)
        return report
    except Exception:
        # Preserve the partial directory as evidence; never delete on failure.
        raise
