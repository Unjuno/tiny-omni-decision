"""Extract AVATAR train/validation JSON only; never open held-out audit labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

METADATA_BYTES = 1_624_245_410
METADATA_SHA256 = "97e9f09e9f4325ac888db2886da22ca4650eafb8848721230a4e5d80fb6a4183"
SPLIT_FILE_SHA256 = "57766c748e2d567060685cc16d8be554201f602224b81e6b3ee13f61d987b9af"
SPLIT_MANIFEST_SHA256 = "bd01932a05952e303b9d0d97c2c15870d8c4315e3e50dc9b08db7e4e3b4cdc41"
EXPECTED_JSON_COUNTS = {"train": 19_159, "validation": 2_523, "audit": 2_584}
CLIP_ID = re.compile(r"([A-Za-z0-9_-]{11})_(\d+)\Z")
FRAME_FILE = re.compile(r"\d+\.json\Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _verified_split(path: Path, *, check_pins: bool) -> tuple[dict, dict[str, str]]:
    raw_hash = _file_sha256(path)
    if check_pins and raw_hash != SPLIT_FILE_SHA256:
        raise ValueError("AVATAR source-video group split file hash differs")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if claimed != _canonical_hash(unsigned):
        raise ValueError("AVATAR source-video group split canonical hash differs")
    if check_pins and claimed != SPLIT_MANIFEST_SHA256:
        raise ValueError("AVATAR source-video group split manifest pin differs")
    owners: dict[str, str] = {}
    for split_name, split in manifest.get("splits", {}).items():
        for source_id in split.get("source_video_ids", []):
            if source_id in owners:
                raise ValueError("source video appears in multiple split groups")
            owners[source_id] = split_name
    if set(manifest.get("splits", {})) != {"train", "validation", "audit"}:
        raise ValueError("train, validation, and audit groups are all required")
    return manifest, owners


def extract_annotations(
    metadata_archive: Path,
    split_manifest: Path,
    output_dir: Path,
    *,
    check_pins: bool = True,
) -> dict:
    metadata_archive, split_manifest, output_dir = map(
        Path, (metadata_archive, split_manifest, output_dir)
    )
    metadata_bytes = metadata_archive.stat().st_size
    metadata_hash = _file_sha256(metadata_archive)
    if check_pins and (metadata_bytes != METADATA_BYTES or metadata_hash != METADATA_SHA256):
        raise ValueError("AVATAR metadata archive does not match pinned revision")
    split, owner = _verified_split(split_manifest, check_pins=check_pins)

    selected: list[tuple[zipfile.ZipInfo, str]] = []
    counts: Counter[str] = Counter()
    audit_count = 0
    unknown_count = 0
    with zipfile.ZipFile(metadata_archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".json"):
                continue
            parts = PurePosixPath(info.filename).parts
            if len(parts) != 2 or not FRAME_FILE.fullmatch(parts[1]):
                raise ValueError(f"unrecognized AVATAR annotation path: {info.filename}")
            match = CLIP_ID.fullmatch(parts[0])
            if match is None:
                raise ValueError(f"unrecognized AVATAR clip ID: {parts[0]}")
            group = owner.get(match.group(1))
            if group is None:
                unknown_count += 1
                continue
            counts[group] += 1
            if group == "audit":
                audit_count += 1
            elif group in {"train", "validation"}:
                selected.append((info, group))
            else:
                raise ValueError(f"unexpected split owner: {group}")

    if unknown_count:
        raise ValueError(f"{unknown_count} annotation frames have unknown source-video groups")
    if check_pins and dict(counts) != EXPECTED_JSON_COUNTS:
        raise ValueError("AVATAR annotation frame counts differ from pinned ZIP-member audit")
    if output_dir.exists():
        raise FileExistsError(f"new output directory required: {output_dir}")

    output_dir.mkdir(parents=True)
    extracted: list[dict[str, str | int]] = []
    with zipfile.ZipFile(metadata_archive) as bundle:
        for info, group in selected:
            target = output_dir / group / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            written = 0
            with bundle.open(info, "r") as source, target.open("xb") as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
                    digest.update(block)
                    written += len(block)
            if written != info.file_size:
                raise ValueError(f"annotation member size mismatch: {info.filename}")
            extracted.append(
                {
                    "path": target.relative_to(output_dir).as_posix(),
                    "bytes": written,
                    "sha256": digest.hexdigest(),
                }
            )

    file_manifest = output_dir / "extracted-files.json"
    encoded_files = json.dumps(extracted, sort_keys=True, separators=(",", ":"))
    file_manifest.write_text(encoded_files + "\n", encoding="utf-8")
    return {
        "status": "PASS_TRAIN_VALIDATION_ANNOTATIONS_EXTRACTED_AUDIT_UNOPENED",
        "metadata_archive_sha256": metadata_hash,
        "split_artifact_sha256": _file_sha256(split_manifest),
        "split_manifest_sha256": split["manifest_sha256"],
        "annotation_json_count_by_split": dict(counts),
        "withheld_audit_annotation_json_count": audit_count,
        "extracted_train_validation_json_count": len(extracted),
        "extracted_annotation_bytes": sum(int(item["bytes"]) for item in extracted),
        "extracted_files_manifest_sha256": _file_sha256(file_manifest),
        "audit_annotation_members_opened": False,
        "jpg_annotation_previews_extracted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-archive", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = extract_annotations(args.metadata_archive, args.split_manifest, args.output_dir)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
