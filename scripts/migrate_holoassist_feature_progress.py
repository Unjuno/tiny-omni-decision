"""Migrate a pinned HoloAssist feature checkpoint after a semantics-preserving fix."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_holoassist_metadata import sha256_file
from scripts.extract_holoassist_frozen_features import (
    VIDEO_CHECKPOINT_SHA256,
    VIDEO_SOURCE_COMMIT,
    _media_paths,
    _require_frozen_selection,
    _save_progress,
    _selected_rows,
    _verify_provenance,
    build_feature_fingerprint,
)

LEGACY_EXTRACTOR_SOURCE_COMMIT = "637794f4f56352ec79f2fbd1ba3e5782fad94f5f"
LEGACY_EXTRACTOR_SCRIPT_SHA256 = "d51532a8d93e68811e29ad497d5787298c4d8e77a46e5ff5a93fe83eea93ba38"


def migrate_progress_file(
    source_progress: Path,
    destination_progress: Path,
    *,
    expected_ids: list[str],
    expected_fingerprint: dict[str, Any],
    legacy_source_commit: str = LEGACY_EXTRACTOR_SOURCE_COMMIT,
    legacy_script_sha256: str = LEGACY_EXTRACTOR_SCRIPT_SHA256,
) -> dict[str, Any]:
    source_progress = source_progress.resolve(strict=True)
    destination_progress = destination_progress.resolve()
    destination_bank = destination_progress.with_name("video-features.npz")
    receipt_path = destination_progress.with_name("video-progress-migration.json")
    if (
        destination_progress.exists()
        or destination_progress.with_suffix(destination_progress.suffix + ".tmp").exists()
    ):
        raise FileExistsError(f"refusing existing destination progress: {destination_progress}")
    if destination_bank.exists() or receipt_path.exists():
        raise FileExistsError(
            f"refusing existing destination artifact beside {destination_progress}"
        )

    with np.load(source_progress, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_id", "features", "completed", "fingerprint_json"}:
            raise ValueError("source progress archive has unexpected entries")
        source_ids = list(map(str, archive["sample_id"].tolist()))
        features = archive["features"].copy()
        completed = archive["completed"].copy()
        source_fingerprint = json.loads(str(archive["fingerprint_json"].item()))

    if source_ids != expected_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("source progress IDs differ from the frozen ordered corpus")
    if (
        features.shape != (len(expected_ids), 768)
        or features.dtype != np.float32
        or completed.shape != (len(expected_ids),)
        or completed.dtype != np.bool_
    ):
        raise ValueError("source progress arrays have unexpected shape or dtype")
    completed_indices = np.flatnonzero(completed)
    if not len(completed_indices) or not np.array_equal(
        completed_indices, np.arange(len(completed_indices))
    ):
        raise ValueError("source progress is not a nonempty contiguous prefix")
    if not np.isfinite(features[completed]).all() or np.any(features[~completed] != 0):
        raise ValueError("source progress contains invalid completed or uncompleted features")

    expected_legacy_fingerprint = dict(expected_fingerprint)
    expected_legacy_fingerprint["extractor_source_commit"] = legacy_source_commit
    expected_legacy_fingerprint["extractor_script_sha256"] = legacy_script_sha256
    if source_fingerprint != expected_legacy_fingerprint:
        changed_fields = sorted(
            key
            for key in set(source_fingerprint) | set(expected_legacy_fingerprint)
            if source_fingerprint.get(key) != expected_legacy_fingerprint.get(key)
        )
        raise ValueError(
            "source progress fingerprint differs beyond the approved code revision: "
            + ", ".join(changed_fields)
        )

    source_sha256 = sha256_file(source_progress)
    destination_progress.parent.mkdir(parents=True, exist_ok=True)
    _save_progress(
        destination_progress,
        ids=expected_ids,
        features=features,
        completed=completed,
        fingerprint=expected_fingerprint,
    )
    destination_sha256 = sha256_file(destination_progress)
    receipt = {
        "status": "PASS",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_progress": str(source_progress),
        "source_progress_sha256": source_sha256,
        "destination_progress": str(destination_progress),
        "destination_progress_sha256": destination_sha256,
        "reused_completed_rows": int(completed.sum()),
        "total_rows": len(expected_ids),
        "ordered_sample_ids_sha256": expected_fingerprint.get("ordered_sample_ids_sha256"),
        "legacy_extractor_source_commit": legacy_source_commit,
        "legacy_extractor_script_sha256": legacy_script_sha256,
        "new_extractor_source_commit": expected_fingerprint["extractor_source_commit"],
        "new_extractor_script_sha256": expected_fingerprint["extractor_script_sha256"],
        "unchanged_fingerprint_fields": [
            "modality",
            "device",
            "config_sha256",
            "model_spec",
            "ordered_sample_ids_sha256",
            "source_window_hashes",
            "av_data_provenance",
            "software_versions",
        ],
        "completed_prefix_verified": True,
        "completed_features_finite": True,
        "uncompleted_features_zero": True,
        "reason": (
            "Resume the same frozen CPU VideoPrism extraction after adding a fallback to the "
            "latest preceding frame for a tail window with no in-window video frame. The old "
            "attempt is retained unchanged; only the two explicitly listed code identity fields "
            "are rebound to this committed revision."
        ),
    }
    temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    with temporary_receipt.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary_receipt.replace(receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-progress", type=Path, required=True)
    parser.add_argument("--destination-progress", type=Path, required=True)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--av-validation-report", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise ValueError("commit the migration and extractor fix before rebinding progress")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, rows_by_split, _window_receipt = _selected_rows(args.windows_dir.resolve(strict=True))
    av_report = json.loads(args.av_validation_report.read_text(encoding="utf-8"))
    if av_report.get("status") != "PASS_HOLOASSIST_TRAINVAL_AV_MEDIA":
        raise ValueError("passing full train/validation AV validation report required")
    _require_frozen_selection(av_report, args.selection_sha256)
    provenance = _verify_provenance(
        events_dir=args.events_dir.resolve(strict=True),
        windows_dir=args.windows_dir.resolve(strict=True),
        av_report_path=args.av_validation_report.resolve(strict=True),
        rows_by_split=rows_by_split,
        av_report=av_report,
        selection_sha256=args.selection_sha256,
    )
    _media_paths(rows, args.media_root.resolve(strict=True))

    model_spec = config["encoders"]["video"]
    upstream_source = args.upstream_source.resolve(strict=True)
    upstream_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        upstream_commit != VIDEO_SOURCE_COMMIT
        or model_spec.get("upstream_source_commit") != upstream_commit
    ):
        raise ValueError("VideoPrism source revision differs from the frozen checkpoint")
    upstream_dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=upstream_source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if upstream_dirty.strip():
        raise ValueError("tracked VideoPrism source files differ from the pinned revision")
    checkpoint = args.checkpoint.resolve(strict=True)
    if sha256_file(checkpoint) != VIDEO_CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash differs from the frozen source")

    fingerprint = build_feature_fingerprint(
        modality="video",
        device="cpu",
        config_path=args.config.resolve(strict=True),
        windows_dir=args.windows_dir.resolve(strict=True),
        rows=rows,
        provenance=provenance,
        model_spec=model_spec,
    )
    receipt = migrate_progress_file(
        args.source_progress,
        args.destination_progress,
        expected_ids=[str(row["id"]) for row in rows],
        expected_fingerprint=fingerprint,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
