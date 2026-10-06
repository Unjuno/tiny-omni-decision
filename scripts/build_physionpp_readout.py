from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("C:/CodexArtifacts/tqpp")
SOURCE_AUDIT_DIR = (
    Path.home() / "AppData/Local/CodexArtifacts/teacher-quality-reloads/"
    "physionpp-readout-metadata-audit-20261007"
)
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.dataset import (  # noqa: E402
    audit_manifest,
    check_train_eval_splits,
    normalize_jsonl,
    sha256_file,
)
from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.media import _HttpRangeReader, materialize_physionpp_videos  # noqa: E402
from tiny_omni_decision.physionpp import (  # noqa: E402
    PHYSIONPP_ETAG,
    PHYSIONPP_LAST_MODIFIED,
    PHYSIONPP_READOUT_URL,
    PHYSIONPP_REVISION,
    build_readout_rows,
    split_readout_examples,
)
from tiny_omni_decision.schema import DatasetManifest, DecisionExample  # noqa: E402

EXPECTED_SIZE = 2_971_891_321
EXPECTED_METADATA_ROWS = 832
EXPECTED_METADATA_RECORDS_SHA256 = (
    "e673b6ba5ee3f96a75051b946003e0dafafdeb8bc12b034c7699cf2934d0b84c"
)
EXPECTED_METADATA_FILES_SHA256 = "381bd6fb786d75adac2abad32b59d3c27e97bc78cc3a16caf7961598df619331"
EXPECTED_MAPPED_MEDIA_IDS_SHA256 = (
    "59abe4999b4485139ce925108dd96eca8bdb1f08e2ebf2aae04ae9768f39dc77"
)
DISK_RESERVE_BYTES = 1024**3


def metadata_from_archive(
    archive: zipfile.ZipFile,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for info in archive.infolist():
        if info.is_dir() or not (
            info.filename.startswith("readout_data_v1/")
            and info.filename.endswith("/metadata.json")
        ):
            continue
        raw = archive.read(info)
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError(f"Physion++ metadata file must contain a list: {info.filename}")
        scenario = info.filename.split("/")[1]
        config_group = info.filename.rsplit("/", 2)[1]
        records.extend(
            {
                "zip_metadata_path": info.filename,
                "scenario": scenario,
                "config_group": config_group,
                **row,
            }
            for row in rows
            if row.get("stimulus_name")
        )
        files.append(
            {
                "path": info.filename,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "row_count": len(rows),
            }
        )
    encoded_records = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records).encode(
        "utf-8"
    )
    records_hash = hashlib.sha256(encoded_records).hexdigest()
    files_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if records_hash != EXPECTED_METADATA_RECORDS_SHA256:
        raise ValueError(f"Physion++ metadata record hash mismatch: {records_hash}")
    if files_hash != EXPECTED_METADATA_FILES_SHA256:
        raise ValueError(f"Physion++ metadata file inventory hash mismatch: {files_hash}")
    if len(records) != EXPECTED_METADATA_ROWS:
        raise ValueError(
            f"expected {EXPECTED_METADATA_ROWS} labeled metadata rows, got {len(records)}"
        )
    return records, files, records_hash


def metadata_from_audit_cache(
    audit_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    records_path = audit_dir / "readout-metadata-records.jsonl"
    manifest_path = audit_dir / "source-metadata-manifest.json"
    if not records_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Physion++ source audit cache is incomplete: {audit_dir}")
    if sha256_file(records_path) != EXPECTED_METADATA_RECORDS_SHA256:
        raise ValueError("Physion++ cached metadata record hash differs from the pinned audit")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_metadata = source_manifest.get("archive", {})
    if (
        archive_metadata.get("archive_size_bytes") != EXPECTED_SIZE
        or archive_metadata.get("etag") != PHYSIONPP_ETAG
        or archive_metadata.get("last_modified") != PHYSIONPP_LAST_MODIFIED
        or archive_metadata.get("metadata_records_sha256") != EXPECTED_METADATA_RECORDS_SHA256
        or archive_metadata.get("metadata_files_sha256") != EXPECTED_METADATA_FILES_SHA256
    ):
        raise ValueError("Physion++ source audit manifest differs from its pinned identifiers")
    files = source_manifest.get("files")
    files_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if not isinstance(files, list) or files_hash != EXPECTED_METADATA_FILES_SHA256:
        raise ValueError("Physion++ cached metadata file inventory hash mismatch")
    with records_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != EXPECTED_METADATA_ROWS:
        raise ValueError(f"expected {EXPECTED_METADATA_ROWS} cached records, got {len(records)}")
    return records, files, EXPECTED_METADATA_RECORDS_SHA256


def _write_examples(path: Path, examples: list[DecisionExample]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(example.model_dump_json(exclude_none=True) + "\n")
    return sha256_file(path)


def _counts(examples: list[DecisionExample]) -> dict[str, Any]:
    sources = Counter(example.source for example in examples)
    labels = Counter(example.target for example in examples)
    groups = {example.source_asset_group_id for example in examples}
    return {
        "examples": len(examples),
        "unique_videos": len({example.source_record_id for example in examples}),
        "scenario_seed_groups": len(groups),
        "labels": dict(sorted(labels.items())),
        "sources": dict(sorted(sources.items())),
    }


def build(
    output_dir: Path,
    *,
    seed: int,
    validation_fraction: float,
    source_audit_dir: Path,
    reuse_media_dir: Path | None,
) -> dict[str, Any]:
    local_artifacts = ARTIFACT_ROOT.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(local_artifacts):
        raise ValueError(f"output must be outside Git under {local_artifacts}")
    staging_dir = output_dir.with_name(f"{output_dir.name}.staging")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"refusing to overwrite output or staging directory: {output_dir}")
    staging_dir.mkdir(parents=True)
    reused_media_count = 0
    if reuse_media_dir is not None:
        source_media_dir = reuse_media_dir.resolve()
        if not source_media_dir.is_dir() or source_media_dir.is_symlink():
            raise ValueError(
                f"reused media directory is not a regular directory: {source_media_dir}"
            )
        media_files = list(source_media_dir.rglob("*"))
        if any(path.is_symlink() for path in media_files):
            raise ValueError("reused Physion++ media cache contains a symlink")
        reused_media_count = sum(path.is_file() for path in media_files)
        target_media_dir = staging_dir / "raw" / "physionpp-readout"
        target_media_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_media_dir, target_media_dir)

    manifest_path = ROOT / "manifests/candidates/physionpp-readout.yaml"
    manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
    if manifest.dataset_id != "physionpp/readout" or manifest.revision != PHYSIONPP_REVISION:
        raise ValueError("Physion++ manifest identity does not match the pinned adapter")
    if audit_manifest(manifest)["project_policy"] != "ALLOW":
        raise ValueError("Physion++ manifest does not pass the project training-source gate")

    if source_audit_dir.exists():
        records, metadata_files, metadata_records_hash = metadata_from_audit_cache(source_audit_dir)
        metadata_origin = "pinned local source audit cache"
    else:
        reader = _HttpRangeReader(PHYSIONPP_READOUT_URL)
        if reader.size != EXPECTED_SIZE:
            raise ValueError(f"Physion++ archive size changed: {reader.size}")
        with zipfile.ZipFile(reader) as archive:
            records, metadata_files, metadata_records_hash = metadata_from_archive(archive)
        metadata_origin = "range-read archive metadata"

    reader = _HttpRangeReader(PHYSIONPP_READOUT_URL)
    if reader.size != EXPECTED_SIZE:
        raise ValueError(f"Physion++ archive size changed: {reader.size}")
    with zipfile.ZipFile(reader) as archive:
        archive_names = {info.filename for info in archive.infolist() if not info.is_dir()}
        rows, mapping_audit = build_readout_rows(
            records,
            archive_names,
            source_revision=PHYSIONPP_REVISION,
            split="readout-development",
        )
        rgb_members = {
            unquote(row["media"][0]["uri"].rsplit(f"{PHYSIONPP_REVISION}/", 1)[1]) for row in rows
        }
        raw_bytes = sum(archive.getinfo(member).file_size for member in rgb_members)

    needed_bytes = raw_bytes + DISK_RESERVE_BYTES
    free_bytes = shutil.disk_usage(staging_dir).free
    if free_bytes < needed_bytes:
        raise OSError(
            "insufficient local disk for Physion++ RGB videos: "
            f"need about {needed_bytes:,} bytes including 1 GiB reserve, "
            f"have {free_bytes:,} bytes"
        )

    if mapping_audit["mapped_rows"] != 800 or mapping_audit["scene_seed_groups"] != 615:
        raise ValueError(
            f"Physion++ mapped inventory differs from its reviewed snapshot: {mapping_audit}"
        )
    mapped_id_hash = str(mapping_audit["record_id_sha256"])
    if mapped_id_hash != EXPECTED_MAPPED_MEDIA_IDS_SHA256:
        raise ValueError(f"Physion++ mapped media identity hash mismatch: {mapped_id_hash}")

    examples = list(normalize_jsonl(rows, manifest, "generic"))
    train, validation, split_report = split_readout_examples(
        examples, seed=seed, validation_fraction=validation_fraction
    )
    print(
        f"Verified {len(train)} train and {len(validation)} validation records; "
        "materializing only their RGB video members.",
        flush=True,
    )
    materialized, media_report = materialize_physionpp_videos(
        [*train, *validation],
        data_root=staging_dir,
        archive_url=PHYSIONPP_READOUT_URL,
    )
    train_ids = {example.id for example in train}
    train_materialized = [example for example in materialized if example.id in train_ids]
    validation_materialized = [example for example in materialized if example.id not in train_ids]
    integrity = check_train_eval_splits(train_materialized, validation_materialized)
    if integrity["status"] != "disjoint":
        raise ValueError(f"Physion++ train/development split failed integrity checks: {integrity}")

    train_hash = _write_examples(staging_dir / "train.jsonl", train_materialized)
    validation_hash = _write_examples(staging_dir / "validation.jsonl", validation_materialized)
    frame_counts = Counter(
        str(example.media[0].num_frames)
        for example in materialized
        if example.media and example.media[0].num_frames is not None
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": manifest.dataset_id,
        "source_revision": manifest.revision,
        "source_manifest_sha256": sha256_file(manifest_path),
        "archive": {
            "url": PHYSIONPP_READOUT_URL,
            "size_bytes": EXPECTED_SIZE,
            "etag": PHYSIONPP_ETAG,
            "last_modified": PHYSIONPP_LAST_MODIFIED,
            "metadata_records_sha256": metadata_records_hash,
            "metadata_files_sha256": EXPECTED_METADATA_FILES_SHA256,
            "metadata_file_count": len(metadata_files),
            "metadata_row_count": len(records),
            "metadata_origin": metadata_origin,
            "mapped_record_id_sha256": mapped_id_hash,
            "mapping_audit": mapping_audit,
        },
        "split": split_report,
        "post_materialization_integrity": integrity,
        "train": {**_counts(train_materialized), "corpus_sha256": train_hash},
        "validation": {
            **_counts(validation_materialized),
            "corpus_sha256": validation_hash,
        },
        "video_frame_count_distribution": dict(
            sorted(frame_counts.items(), key=lambda item: int(item[0]))
        ),
        "media": media_report,
        "reused_media_file_count": reused_media_count,
        "reused_media_source": str(reuse_media_dir.resolve()) if reuse_media_dir else None,
        "audit_limitation": (
            "This source's metadata and labels were inspected during source audit; these "
            "partitions are train/development only and cannot serve as a fresh sealed audit."
        ),
        "task_definition": (
            "Retrospective temporal event classification over the full supplied clip. "
            "No future prediction claim or post-cutoff masking is made."
        ),
        "sealed_test_archive_opened": False,
    }
    (staging_dir / "corpus-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    staging_dir.replace(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build grouped Physion++ readout train/dev corpora."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_ROOT / "fullclip-v1",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--source-audit-dir",
        type=Path,
        default=SOURCE_AUDIT_DIR,
    )
    parser.add_argument("--reuse-media-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output_dir,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
                source_audit_dir=args.source_audit_dir,
                reuse_media_dir=args.reuse_media_dir,
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
