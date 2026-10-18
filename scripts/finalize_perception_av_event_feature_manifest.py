"""Create a split-correct derivative of frozen AV event features."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_media_pts import PLAN_SHA256
from scripts.perception_av_event_protocol import model_selection_video_ids
from scripts.train_av_event_source import validate_video_splits


def assign_model_split(
    plan_split: str,
    video_id: str,
    train_video_ids: set[str],
    selection_video_ids: set[str],
) -> str:
    """Map pinned acquisition TRAIN groups into fit/selection, failing closed."""
    if plan_split != "train":
        if plan_split not in {"calibration", "development_validation"}:
            raise ValueError(f"unrecognized pinned model split: {plan_split}")
        return plan_split
    if not video_id or video_id not in train_video_ids:
        raise ValueError("training video is not in the pinned training video set")
    return "selection" if video_id in selection_video_ids else "fit"


def finalize_rows(rows: list[dict], train_video_ids: set[str], selection_video_ids: set[str]):
    """Relabel rows only; preserve every feature path, hash, and target field."""
    if not train_video_ids or not selection_video_ids <= train_video_ids:
        raise ValueError("selection videos must be a nonempty subset of pinned train videos")
    finalized = []
    for source in rows:
        row = dict(source)
        row["split"] = assign_model_split(
            row.get("split"), row.get("video_id"), train_video_ids, selection_video_ids
        )
        finalized.append(row)
    validate_video_splits(finalized)
    if {row["video_id"] for row in finalized if row["split"] == "fit"} & selection_video_ids:
        raise ValueError("selection video leaked into fit")
    return finalized


def finalize_manifest(plan_path: Path, features_path: Path, output_path: Path, receipt_path: Path):
    if output_path.exists() or receipt_path.exists():
        raise ValueError("finalized manifest and receipt paths must be new")
    plan_bytes = plan_path.read_bytes()
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    if plan_hash != PLAN_SHA256:
        raise ValueError("plan differs from the pinned acquisition plan")
    plan = json.loads(plan_bytes)
    train_video_ids = {
        row["video_id"]
        for row in plan["splits"]["train"]["rows"]
        if row["model_split"] == "train"
    }
    if len(train_video_ids) != 300:
        raise ValueError("pinned plan must contain exactly 300 model-training videos")
    calibration_video_ids = {
        row["video_id"]
        for row in plan["splits"]["train"]["rows"]
        if row["model_split"] == "calibration"
    }
    valid_video_ids = {row["video_id"] for row in plan["splits"]["valid"]["rows"]}
    selection_video_ids = model_selection_video_ids(train_video_ids, selection_count=60)
    rows = [
        json.loads(line)
        for line in features_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_hash = sha256_file(features_path)
    finalized = finalize_rows(rows, train_video_ids, selection_video_ids)
    original_by_id = {row["id"]: row for row in rows}
    if len(original_by_id) != len(rows) or set(original_by_id) != {row["id"] for row in finalized}:
        raise ValueError("source and finalized event inventories differ")
    for row in finalized:
        restored = dict(row)
        restored["split"] = original_by_id[row["id"]]["split"]
        if restored != original_by_id[row["id"]]:
            raise ValueError("finalization changed a field other than split")
    for row in finalized:
        if row["id"].startswith("train:"):
            if row["video_id"] not in train_video_ids | calibration_video_ids:
                raise ValueError("feature row is absent from pinned TRAIN groups")
            expected_split = (
                "calibration"
                if row["video_id"] in calibration_video_ids
                else "selection"
                if row["video_id"] in selection_video_ids
                else "fit"
            )
        elif row["id"].startswith("valid:"):
            if row["video_id"] not in valid_video_ids:
                raise ValueError("feature row is absent from pinned VALID groups")
            expected_split = "development_validation"
        else:
            raise ValueError("feature row ID has an unknown official source split")
        if row["split"] != expected_split:
            raise ValueError("feature row split does not match pinned video group")
        for path_key, hash_key in (
            ("audio_feature_path", "audio_feature_sha256"),
            ("visual_feature_path", "visual_feature_sha256"),
        ):
            path = Path(row[path_key])
            if not path.is_file() or sha256_file(path) != row[hash_key]:
                raise ValueError(f"frozen feature file missing or hash-mismatched: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in finalized:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    counts = Counter(row["split"] for row in finalized)
    receipt = {
        "status": "PASS_SPLIT_ONLY_MANIFEST_FINALIZATION",
        "plan_sha256": plan_hash,
        "source_feature_manifest": str(features_path.resolve()),
        "source_feature_manifest_sha256": source_hash,
        "output_feature_manifest": str(output_path.resolve()),
        "output_feature_manifest_sha256": sha256_file(output_path),
        "rows": len(finalized),
        "rows_by_split": dict(sorted(counts.items())),
        "videos_by_split": {
            split: len({row["video_id"] for row in finalized if row["split"] == split})
            for split in sorted(counts)
        },
        "training_video_count": len(train_video_ids),
        "fit_video_count": len(train_video_ids - selection_video_ids),
        "selection_video_count": len(selection_video_ids),
        "selection_video_ids_sha256": hashlib.sha256(
            "\n".join(sorted(selection_video_ids)).encode()
        ).hexdigest(),
        "feature_arrays_recomputed": False,
        "only_row_split_fields_changed": True,
        "test_opened": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_manifest(args.plan, args.features, args.output, args.receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    _main()
