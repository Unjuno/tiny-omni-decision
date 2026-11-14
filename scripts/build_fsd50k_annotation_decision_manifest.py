"""Build a rights-filtered, uploader-disjoint FSD50K annotation-prediction split.

This task predicts whether FSD50K assigned one of four broad labels. A missing
annotation is a negative *for annotation prediction*, not proof that the sound
is absent. This script reads only the FSD50K dev metadata and dev labels; it
never opens the official eval labels or media.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

DATASET_REVISION = "10.5281/zenodo.4060432"
SOURCE_METADATA_DIR = "fsd50k-metadata-audit-01"
OUTPUT_DIRNAME = "fsd50k-annotation-presence-v1-preflight-20261115"
TARGET_LABELS = ("Animal", "Human_voice", "Vehicle", "Music")
ALLOWED_LICENSES = {
    "http://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
}
PREVIOUS_SELECTION_DIRS = (
    "fsd50k-training-128-01",
    "fsd50k-training-128-02",
    "fsd50k-training-smoke-01",
    "fsd50k-training-smoke-02",
)
SPLIT_RULE = (
    "sha256('tiny-omni-fsd50k-annotation-v1:17:' + uploader), first 8 hex mod 10; "
    "0-6 train, 7 calibration, 8-9 validation"
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uploader_split(uploader: str) -> str:
    if not isinstance(uploader, str) or not uploader.strip():
        raise ValueError("FSD50K uploader identity is required for grouped splitting")
    payload = f"tiny-omni-fsd50k-annotation-v1:17:{uploader}".encode()
    bucket = int(hashlib.sha256(payload).hexdigest()[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket == 7:
        return "calibration"
    return "validation"


def collect_previous_exposure(
    artifact_root: Path, metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    used_ids: set[str] = set()
    for dirname in PREVIOUS_SELECTION_DIRS:
        selection_path = artifact_root / dirname / "selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"required preserved exposure receipt is missing: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not isinstance(selection, list):
            raise ValueError(f"invalid previous FSD selection: {selection_path}")
        for row in selection:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"invalid previous FSD selection row: {selection_path}")
            used_ids.add(row["id"])

    missing = sorted(used_ids - metadata.keys())
    if missing:
        raise ValueError(
            f"previously exposed FSD IDs missing from pinned dev metadata: {missing[:5]}"
        )
    used_uploaders = {metadata[clip_id].get("uploader") for clip_id in used_ids}
    if None in used_uploaders or "" in used_uploaders:
        raise ValueError("previously exposed clips have missing uploader identity")
    return {
        "selection_directories": list(PREVIOUS_SELECTION_DIRS),
        "unique_exposed_clip_ids": sorted(used_ids),
        "unique_exposed_clip_count": len(used_ids),
        "quarantined_uploader_count": len(used_uploaders),
        "quarantined_uploaders": sorted(used_uploaders),
    }


def build_split_rows(
    dev_csv_text: str,
    metadata: dict[str, dict[str, Any]],
    *,
    excluded_ids: set[str],
    excluded_uploaders: set[str],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("train", "calibration", "validation")
    }
    seen_ids: set[str] = set()
    reader = csv.DictReader(io.StringIO(dev_csv_text))
    required_fields = {"fname", "labels", "split"}
    if not required_fields.issubset(reader.fieldnames or []):
        raise ValueError("FSD50K dev labels lack required columns")

    for source_row in reader:
        if source_row["split"] != "train":
            continue
        clip_id = source_row["fname"]
        if clip_id in seen_ids:
            raise ValueError(f"duplicate FSD50K dev training ID: {clip_id}")
        seen_ids.add(clip_id)
        item = metadata.get(clip_id)
        if item is None:
            raise ValueError(f"FSD50K training ID missing from dev metadata: {clip_id}")
        license_url = item.get("license")
        if license_url not in ALLOWED_LICENSES:
            continue
        uploader = item.get("uploader")
        if not isinstance(uploader, str) or not uploader.strip():
            raise ValueError(f"FSD50K uploader missing for {clip_id}")
        if clip_id in excluded_ids or uploader in excluded_uploaders:
            continue

        annotated = set(source_row["labels"].split(","))
        split = uploader_split(uploader)
        result[split].append(
            {
                "sample_id": clip_id,
                "source_id": f"freesound:{clip_id}",
                "group_id": uploader,
                "split": split,
                "license": license_url,
                "title": item.get("title", ""),
                "creator": uploader,
                "source_url": f"https://freesound.org/people/{uploader}/sounds/{clip_id}/",
                "targets": {label: label in annotated for label in TARGET_LABELS},
            }
        )

    for rows in result.values():
        rows.sort(key=lambda row: row["sample_id"])
    return result


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {row["group_id"] for row in rows}
    return {
        "examples": len(rows),
        "unique_source_ids": len({row["source_id"] for row in rows}),
        "unique_uploaders": len(groups),
        "positives_by_label": {
            label: {
                "examples": sum(bool(row["targets"][label]) for row in rows),
                "positive_uploaders": len(
                    {row["group_id"] for row in rows if row["targets"][label]}
                ),
            }
            for label in TARGET_LABELS
        },
        "ordered_sample_ids_sha256": canonical_sha256([row["sample_id"] for row in rows]),
        "ordered_rows_sha256": canonical_sha256(rows),
    }


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    all_ids: set[str] = set()
    all_groups: set[str] = set()
    for name, rows in splits.items():
        ids = [row["sample_id"] for row in rows]
        groups = {row["group_id"] for row in rows}
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate sample IDs in {name}")
        if all_ids.intersection(ids):
            raise ValueError("sample ID overlap across FSD50K splits")
        if all_groups.intersection(groups):
            raise ValueError("uploader overlap across FSD50K splits")
        all_ids.update(ids)
        all_groups.update(groups)
        if name == "validation" and (len(rows) < 1000 or len(groups) < 200):
            raise ValueError("validation misses provisional 1000-example/200-uploader gate")


def load_dev_source(source_dir: Path) -> tuple[dict[str, dict[str, Any]], str, dict[str, str]]:
    downloads = json.loads((source_dir / "downloads.json").read_text(encoding="utf-8"))
    verified_hashes: dict[str, str] = {}
    for filename, expected in downloads.items():
        path = source_dir / filename
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise ValueError(f"FSD50K pinned source hash mismatch: {filename}")
        verified_hashes[filename] = digest
    with zipfile.ZipFile(source_dir / "FSD50K.metadata.zip") as archive:
        dev_metadata = json.loads(archive.read("FSD50K.metadata/dev_clips_info_FSD50K.json"))
    with zipfile.ZipFile(source_dir / "FSD50K.ground_truth.zip") as archive:
        dev_csv_text = archive.read("FSD50K.ground_truth/dev.csv").decode("utf-8-sig")
    return dev_metadata, dev_csv_text, verified_hashes


def write_manifest(artifact_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite an FSD50K attempt: {output_dir}")
    source_dir = artifact_root / SOURCE_METADATA_DIR
    metadata, dev_csv_text, source_hashes = load_dev_source(source_dir)
    exposure = collect_previous_exposure(artifact_root, metadata)
    excluded_ids = set(exposure["unique_exposed_clip_ids"])
    excluded_uploaders = set(exposure["quarantined_uploaders"])
    splits = build_split_rows(
        dev_csv_text,
        metadata,
        excluded_ids=excluded_ids,
        excluded_uploaders=excluded_uploaders,
    )
    validate_splits(splits)

    output_dir.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for name, rows in splits.items():
        path = output_dir / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        summaries[name] = summarize_split(rows)

    report = {
        "status": "PREFLIGHT_PASS",
        "dataset": "FSD50K",
        "dataset_revision": DATASET_REVISION,
        "task": (
            "predict whether FSD50K assigned each supplied broad category label; "
            "not a claim of verified acoustic absence"
        ),
        "target_labels": list(TARGET_LABELS),
        "option_schema": {"response": ["no_annotation", "annotated"]},
        "license_policy": {
            "allowed": sorted(ALLOWED_LICENSES),
            "excluded": ["CC BY-NC", "CC Sampling+"],
        },
        "group_field": "FSD50K dev metadata uploader",
        "seed": 17,
        "split_rule": SPLIT_RULE,
        "prior_exposure": {
            key: value for key, value in exposure.items() if key != "quarantined_uploaders"
        },
        "source_sha256": source_hashes,
        "official_eval_labels_read": False,
        "media_downloaded": False,
        "feature_extraction": False,
        "model_loaded": False,
        "gpu_used": False,
        "splits": summaries,
        "limitations": [
            (
                "FSD50K dev labels can be incomplete; absence means no annotation assigned "
                "for this task, not verified physical absence."
            ),
            (
                "Official eval labels remain unopened and must only be used after the "
                "candidate, preprocessing, and selector are frozen."
            ),
            (
                "Cross-uploader near-duplicate audio has not yet been checked; exact "
                "byte/media hashes are required before fitting."
            ),
        ],
    }
    report["manifest_sha256"] = canonical_sha256(
        {"splits": summaries, "source_sha256": source_hashes}
    )
    (output_dir / "preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("C:/CodexArtifacts/pretrained-reuse-r2")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.artifact_root / OUTPUT_DIRNAME
    print(json.dumps(write_manifest(args.artifact_root, output), ensure_ascii=False))


if __name__ == "__main__":
    main()
