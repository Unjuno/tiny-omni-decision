"""Create an immutable train/dev image split from unused CLEVR-4 train assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile


def deterministic_group_split(
    rows: list[dict[str, Any]],
    *,
    train_count: int,
    validation_count: int,
    seed: int,
    excluded_ids: set[str],
    excluded_media_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split image-level groups deterministically after strict identity checks."""
    ids = [str(row["id"]) for row in rows]
    hashes = [str(row["sha256"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source image identity")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate media identity")
    candidates = [
        row
        for row in rows
        if str(row["id"]) not in excluded_ids and str(row["sha256"]) not in excluded_media_hashes
    ]
    if train_count < 1 or validation_count < 1 or len(candidates) < train_count + validation_count:
        raise ValueError(
            f"insufficient unused source images: available={len(candidates)}, "
            f"required={train_count + validation_count}"
        )
    ordered = sorted(
        candidates,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
            str(row["id"]),
        ),
    )
    train = ordered[:train_count]
    validation = ordered[train_count : train_count + validation_count]
    return train, validation


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=30000)
    parser.add_argument("--validation-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    preflight = json.loads(args.source_preflight.read_text(encoding="utf-8"))
    if (
        preflight.get("archive_sha512_match") is not True
        or preflight.get("archive_sha512_actual") != preflight.get("archive_sha512")
        or args.archive.stat().st_size != preflight.get("archive_bytes")
    ):
        raise ValueError("source archive does not match its verified preflight")
    excluded_ids: set[str] = set()
    excluded_hashes: set[str] = set()
    exclusion_manifest_hashes = {}
    for path in args.exclude_manifest:
        exclusion_manifest_hashes[str(path.resolve())] = sha256_file(path)
        for row in read_jsonl(path):
            excluded_ids.add(str(row["id"]))
            excluded_hashes.add(str(row.get("sha256", row.get("media_sha256"))).lower())
    source_assets = read_jsonl(args.source_manifest)
    labels_by_id = {str(row["id"]): row for row in read_jsonl(args.source_labels)}
    source_by_id: dict[str, dict[str, Any]] = {}
    candidates = []
    for row in source_assets:
        source_label = labels_by_id.get(str(row["id"]))
        if source_label is None or source_label.get("source_split") != "train":
            raise ValueError("source manifest must contain only the predeclared train pool")
        candidates.append(
            {
                "id": str(row["id"]),
                "sha256": str(row.get("sha256", row.get("media_sha256"))).lower(),
                "split": "train",
            }
        )
        source_by_id[str(row["id"])] = row
    with ZipFile(args.archive) as archive:
        annotations_payload = archive.read("clevr_4_annots.json")
        annotation_obj = json.loads(annotations_payload)
        if not isinstance(annotation_obj, dict):
            raise TypeError("CLEVR-4 annotations must be an ID-keyed object")
        selected_train, selected_validation = deterministic_group_split(
            candidates,
            train_count=args.train_count,
            validation_count=args.validation_count,
            seed=args.seed,
            excluded_ids=excluded_ids,
            excluded_media_hashes=excluded_hashes,
        )
        manifest_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        label_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
        seen_hashes = set(excluded_hashes)
        for split, selected in (
            ("train", selected_train),
            ("validation", selected_validation),
        ):
            for item in selected:
                sample_id = str(item["id"])
                member = f"images/{sample_id}.png"
                payload = archive.read(member)
                media_hash = hashlib.sha256(payload).hexdigest()
                expected_hash = str(
                    source_by_id[sample_id].get(
                        "sha256", source_by_id[sample_id].get("media_sha256")
                    )
                ).lower()
                if media_hash != expected_hash:
                    raise ValueError(f"source manifest media hash mismatch: {sample_id}")
                if media_hash in seen_hashes:
                    raise ValueError(f"duplicate or previously used media identity: {sample_id}")
                seen_hashes.add(media_hash)
                manifest_rows[split].append(
                    {
                        "id": sample_id,
                        "split": split,
                        "member": member,
                        "bytes": len(payload),
                        "sha256": media_hash,
                    }
                )
                labels_for_sample = labels_by_id.get(sample_id)
                if labels_for_sample is None:
                    raise ValueError(f"source labels missing {sample_id}")
                if any(
                    str(labels_for_sample[key]) != str(annotation_obj[sample_id][key])
                    for key in ("color", "texture", "count", "shape")
                ):
                    raise ValueError(f"source labels differ from pinned annotations: {sample_id}")
                label_rows[split].append(
                    {
                        "id": sample_id,
                        **{
                            key: labels_for_sample[key]
                            for key in ("color", "texture", "count", "shape")
                        },
                    }
                )
        write_jsonl(output / "train-image-manifest.jsonl", manifest_rows["train"])
        write_jsonl(output / "validation-image-manifest.jsonl", manifest_rows["validation"])
        write_jsonl(output / "train-labels.jsonl", label_rows["train"])
        write_jsonl(output / "validation-labels.jsonl", label_rows["validation"])
        annotation_sha256 = hashlib.sha256(annotations_payload).hexdigest()
    summary = {
        "seed": args.seed,
        "requested_train_images": args.train_count,
        "requested_validation_images": args.validation_count,
        "source_train_images": len(candidates),
        "source_archive_sha512": preflight["archive_sha512_actual"],
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "source_labels_sha256": sha256_file(args.source_labels),
        "exclusion_manifest_sha256": exclusion_manifest_hashes,
        "annotation_sha256": annotation_sha256,
        "excluded_ids": len(excluded_ids),
        "excluded_media_hashes": len(excluded_hashes),
        "selected_train_images": len(selected_train),
        "selected_validation_images": len(selected_validation),
        "train_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in manifest_rows["train"]).encode()
        ).hexdigest(),
        "validation_id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in manifest_rows["validation"]).encode()
        ).hexdigest(),
    }
    summary_path = output / "split-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
