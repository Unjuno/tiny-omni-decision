"""Verify and index only the official training split of Clevr-4 100k."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

EXPECTED_ARCHIVE_BYTES = 36_057_238_315
EXPECTED_ARCHIVE_SHA512 = (
    "7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b"
    "64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c"
)
EXPECTED_IMAGE_COUNT = 100_000


def sha512_file(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(Counter(str(row[field]) for row in rows).items()))
        for field in ("color", "texture", "count", "shape")
    }


def build_index(
    archive: Path,
    output: Path,
    *,
    expected_archive_bytes: int = EXPECTED_ARCHIVE_BYTES,
    expected_sha512: str = EXPECTED_ARCHIVE_SHA512,
    expected_image_count: int = EXPECTED_IMAGE_COUNT,
    validation_fraction: float = 0.1,
    seed: int = 17,
) -> dict[str, Any]:
    """Create deterministic train/validation manifests from official train only."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tiny_omni_decision.clevr4_data import read_training_annotations, split_records_by_asset

    if output.exists():
        raise FileExistsError(output)
    if archive.stat().st_size != expected_archive_bytes:
        raise ValueError("Clevr-4 archive size does not match the pinned source")
    actual_sha512 = sha512_file(archive)
    if actual_sha512 != expected_sha512:
        raise ValueError("Clevr-4 archive SHA-512 does not match the official checksum")

    with ZipFile(archive) as zipped:
        infos = zipped.infolist()
        annotation_infos = [info for info in infos if info.filename == "clevr_4_annots.json"]
        if len(annotation_infos) != 1:
            raise ValueError("archive must contain exactly one clevr_4_annots.json")
        image_infos: dict[str, Any] = {}
        for info in infos:
            member = PurePosixPath(info.filename)
            if member.parent == PurePosixPath("images") and member.suffix.lower() == ".png":
                if member.stem in image_infos:
                    raise ValueError(f"duplicate image identity in archive: {member.stem}")
                image_infos[member.stem] = info
        if len(image_infos) != expected_image_count:
            raise ValueError(
                f"expected {expected_image_count} image entries, found {len(image_infos)}"
            )

        with zipped.open(annotation_infos[0], "r") as annotations_stream:
            official_train_rows, split_counts = read_training_annotations(annotations_stream)

    if sum(split_counts.values()) != expected_image_count:
        raise ValueError("annotation split counts do not cover the image inventory")
    if len(official_train_rows) != split_counts.get("train", 0):
        raise ValueError("decoded train rows do not match the official train count")
    if not official_train_rows:
        raise ValueError("official training split is empty")

    train_ids, validation_ids = split_records_by_asset(
        ({"id": row["id"], "asset_id": row["id"]} for row in official_train_rows),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    rows_by_id = {str(row["id"]): row for row in official_train_rows}
    if len(rows_by_id) != len(official_train_rows):
        raise ValueError("official training split contains duplicate image IDs")
    missing_images = set(rows_by_id) - set(image_infos)
    if missing_images:
        raise ValueError(f"official train annotation has no image member: {len(missing_images)}")

    def labels_for(ids: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "id": image_id,
                "asset_id": image_id,
                "source_split": "train",
                "color": rows_by_id[image_id]["color"],
                "texture": rows_by_id[image_id]["texture"],
                "count": rows_by_id[image_id]["count"],
                "shape": rows_by_id[image_id]["shape"],
            }
            for image_id in ids
        ]

    train_labels = labels_for(train_ids)
    validation_labels = labels_for(validation_ids)

    def media_for(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": row["id"],
                "asset_id": row["asset_id"],
                "archive_member": f"images/{row['id']}.png",
                "bytes": image_infos[row["id"]].file_size,
                "media_sha256": None,
            }
            for row in labels
        ]

    train_media = media_for(train_labels)
    validation_media = media_for(validation_labels)
    output.mkdir(parents=True)
    files = {
        "train-labels.jsonl": train_labels,
        "validation-labels.jsonl": validation_labels,
        "train-media-manifest.jsonl": train_media,
        "validation-media-manifest.jsonl": validation_media,
    }
    manifest_hashes: dict[str, str] = {}
    for filename, rows in files.items():
        _write_jsonl(output / filename, rows)
        manifest_hashes[filename] = _jsonl_sha256(rows)

    result = {
        "status": "TRAIN_INDEX_READY_MEDIA_HASH_PENDING",
        "archive_bytes": archive.stat().st_size,
        "archive_sha512": actual_sha512,
        "annotation_member": "clevr_4_annots.json",
        "archive_image_entries": len(image_infos),
        "official_split_counts": split_counts,
        "official_test_annotation_values_decoded": 0,
        "official_test_image_members_opened": 0,
        "decoded_official_train_examples": len(official_train_rows),
        "internal_validation_fraction": validation_fraction,
        "seed": seed,
        "asset_split_policy": "SHA256(seed:asset_id) rank, first ceil(fraction*unique assets)",
        "train_examples": len(train_labels),
        "validation_examples": len(validation_labels),
        "train_asset_ids_unique": len({row["asset_id"] for row in train_labels}),
        "validation_asset_ids_unique": len({row["asset_id"] for row in validation_labels}),
        "train_validation_asset_overlap": 0,
        "media_hashes_computed": False,
        "train_label_counts": _label_counts(train_labels),
        "validation_label_counts": _label_counts(validation_labels),
        "train_id_order_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_id_order_sha256": hashlib.sha256(
            "\n".join(validation_ids).encode()
        ).hexdigest(),
        "manifest_sha256": manifest_hashes,
        "official_test_split_used_for_selection": False,
        "media_payloads_extracted": 0,
    }
    (output / "index-report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_index(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
