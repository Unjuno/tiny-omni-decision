"""Independently reload and verify the frozen AV input corpus without models."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cross_split_overlap_count(rows: list[dict[str, Any]], field: str) -> int:
    splits_by_hash: dict[str, set[str]] = {}
    for row in rows:
        value = row[field]
        digest = value["normalized_content_sha256"] if isinstance(value, dict) else value
        identity = f"{row.get('modality', '')}\0{digest}"
        splits_by_hash.setdefault(identity, set()).add(str(row["split"]))
    return sum(len(splits) > 1 for splits in splits_by_hash.values())


def audit(
    inputs_dir: Path, media_dir: Path, preflight_dir: Path, output_dir: Path
) -> dict[str, Any]:
    import numpy as np

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    inputs_result_path = inputs_dir / "result.json"
    inputs_result = json.loads(inputs_result_path.read_text(encoding="utf-8"))
    groups_path = inputs_dir / "groups.jsonl"
    groups = [json.loads(line) for line in groups_path.read_text(encoding="utf-8").splitlines()]
    if inputs_result.get("status") != "PASS_FULL_DECODE_AND_NORMALIZED_CONTENT_SPLIT_GATES":
        raise ValueError("full decoded input receipt required")
    if inputs_result.get("artifact_hashes", {}).get("groups.jsonl") != sha256_file(groups_path):
        raise ValueError("group manifest hash mismatch")
    if len(groups) != 220 or len({row["source_group"] for row in groups}) != 220:
        raise ValueError("220 unique source groups are required")
    expected_splits = {"train": 126, "calibration": 38, "validation": 56}
    if {s: sum(row["split"] == s for row in groups) for s in expected_splits} != expected_splits:
        raise ValueError("frozen split counts differ")

    preflight_rows = {
        split: [
            json.loads(line)
            for line in (preflight_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for split in expected_splits
    }
    expected_groups = {
        row[field]
        for split, rows in preflight_rows.items()
        for row in rows
        for field in ("audio_source_group", "video_source_group")
    }
    if expected_groups != {row["source_group"] for row in groups}:
        raise ValueError("prepared groups do not cover frozen train/calibration/validation IDs")
    owners: dict[tuple[str, str], set[str]] = {}
    for modality in ("audio", "video"):
        for row in groups:
            digest = row[modality]["normalized_content_sha256"]
            owners.setdefault((modality, digest), set()).add(row["split"])
    normalized_cross_split_overlap = sum(len(splits) > 1 for splits in owners.values())
    normalized_unique = {
        modality: len({row[modality]["normalized_content_sha256"] for row in groups})
        for modality in ("audio", "video")
    }
    if normalized_cross_split_overlap or normalized_unique != {"audio": 220, "video": 220}:
        raise ValueError("normalized audio/video contents must be unique and split-disjoint")

    checked_arrays = 0
    total_audio_samples = 0
    for row in groups:
        path = inputs_dir / row["input_file"]
        if sha256_file(path) != row["input_sha256"]:
            raise ValueError("prepared NPZ hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            audio, video = data["audio"], data["video"]
            if (
                audio.ndim != 2
                or audio.shape[1] != 1
                or audio.dtype != np.float32
                or not np.isfinite(audio).all()
                or not np.any(audio)
            ):
                raise ValueError("audio array is not finite mono float32 24 kHz")
            if video.shape != (8, 288, 288, 3) or video.dtype != np.uint8:
                raise ValueError("video array shape/dtype differs from fixed encoder input")
            checked_arrays += 1
            total_audio_samples += len(audio)

    acquisition = json.loads((media_dir / "result.json").read_text(encoding="utf-8"))
    media_members_path = media_dir / "members.json"
    if acquisition.get("status") != "PASS_MEDIA_ACQUISITION_RAW_HASH_DISJOINT" or sha256_file(
        media_members_path
    ) != acquisition.get("members_manifest_sha256"):
        raise ValueError("verified raw-media acquisition manifest required")
    raw_members = json.loads(media_members_path.read_text(encoding="utf-8"))
    raw_hash_overlap = cross_split_overlap_count(list(raw_members.values()), "sha256")
    if raw_hash_overlap:
        raise ValueError("raw media content hash overlaps across splits")
    preflight_result = json.loads((preflight_dir / "result.json").read_text(encoding="utf-8"))
    for split in expected_splits:
        expected_hash = preflight_result.get("output_sha256", {}).get(f"{split}.jsonl")
        if not expected_hash or expected_hash != sha256_file(preflight_dir / f"{split}.jsonl"):
            raise ValueError(f"frozen {split} Decision manifest hash mismatch")

    output_dir.mkdir(parents=True)
    result = {
        "status": "PASS_INPUT_ARRAY_RELOAD_AND_CONTENT_DISJOINTNESS",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "inputs_result_sha256": sha256_file(inputs_result_path),
        "groups_manifest_sha256": sha256_file(groups_path),
        "media_manifest_sha256": sha256_file(media_members_path),
        "preflight_sha256": {
            split: sha256_file(preflight_dir / f"{split}.jsonl") for split in expected_splits
        },
        "verified_groups": len(groups),
        "verified_arrays": checked_arrays,
        "groups_by_split": expected_splits,
        "normalized_unique_content_by_modality": normalized_unique,
        "normalized_cross_split_overlap": normalized_cross_split_overlap,
        "raw_cross_split_overlap": raw_hash_overlap,
        "total_audio_samples_24khz": total_audio_samples,
        "sealed_audit_opened": False,
        "weights_loaded": False,
        "optimizer_updates": 0,
        "gpu_used": False,
        "training": False,
        "evaluation": False,
    }
    result_path = output_dir / "audit.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["audit_sha256"] = sha256_file(result_path)
    (output_dir / "audit.sha256").write_text(result["audit_sha256"] + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inputs_dir, args.media_dir, args.preflight_dir, args.output_dir)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "verified_groups",
                    "verified_arrays",
                    "normalized_cross_split_overlap",
                    "raw_cross_split_overlap",
                    "audit_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
