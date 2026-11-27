"""Combine frozen BSD10k train features while preserving validation features exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_feature_set(directory: Path, split: str) -> tuple[dict, dict, np.ndarray]:
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    index = json.loads((directory / "sample-index.json").read_text(encoding="utf-8"))
    features_path = directory / "features.npy"
    values = np.load(features_path, allow_pickle=False)
    ids = [str(item) for item in index.get("sound_ids", [])]
    durations = index.get("durations_seconds", [])
    if (
        report.get("status") != "PASS_FROZEN_FEATURES"
        or report.get("partial_smoke") is not False
        or report.get("split") != split
        or report.get("feature_sha256") != sha256_file(features_path)
        or report.get("sample_index_sha256") != sha256_file(directory / "sample-index.json")
        or len(ids) != len(set(ids))
        or values.shape != (len(ids), 384)
        or not np.isfinite(values).all()
        or len(durations) != len(ids)
        or not np.isfinite(np.asarray(durations, dtype=np.float64)).all()
    ):
        raise ValueError(f"invalid frozen {split} feature set: {directory}")
    return report, index, values


def _write_set(directory: Path, report: dict, index: dict, values: np.ndarray) -> None:
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite feature output: {directory}")
    directory.mkdir(parents=True)
    feature_path = directory / "features.npy"
    index_path = directory / "sample-index.json"
    np.save(feature_path, values.astype(np.float32, copy=False), allow_pickle=False)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    report["feature_shape"] = list(values.shape)
    report["records"] = len(index["sound_ids"])
    report["feature_sha256"] = sha256_file(feature_path)
    report["sample_index_sha256"] = sha256_file(index_path)
    (directory / "run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def prepare_feature_sets(
    combined_manifest_path: Path,
    additions_manifest_path: Path,
    base_train_dir: Path,
    additions_dir: Path,
    base_validation_dir: Path,
    output_train_dir: Path,
    output_validation_dir: Path,
) -> dict:
    if output_train_dir.exists() or output_validation_dir.exists():
        raise FileExistsError("refusing to overwrite one or both frozen-feature outputs")
    manifest = json.loads(combined_manifest_path.read_text(encoding="utf-8"))
    additions_manifest = json.loads(additions_manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(combined_manifest_path)
    base_hash = manifest.get("training_extension", {}).get("base_split_manifest_sha256")
    if not base_hash or additions_manifest.get("metadata_zip_sha256") != manifest.get(
        "metadata_zip_sha256"
    ):
        raise ValueError("combined and additions manifests do not share the pinned source")

    base_train_report, base_train_index, base_train_values = _read_feature_set(
        base_train_dir, "train"
    )
    added_report, added_index, added_values = _read_feature_set(additions_dir, "train")
    validation_report, validation_index, validation_values = _read_feature_set(
        base_validation_dir, "validation"
    )
    if (
        base_train_report.get("split_manifest_sha256") != base_hash
        or added_report.get("split_manifest_sha256") != sha256_file(additions_manifest_path)
        or validation_report.get("split_manifest_sha256") != base_hash
        or not (
            base_train_report.get("weights_sha256")
            == added_report.get("weights_sha256")
            == validation_report.get("weights_sha256")
        )
        or not (
            base_train_report.get("source_commit")
            == added_report.get("source_commit")
            == validation_report.get("source_commit")
        )
        or not (
            base_train_report.get("model_parameters")
            == added_report.get("model_parameters")
            == validation_report.get("model_parameters")
        )
    ):
        raise ValueError("feature inputs are not from the same pinned encoder and split lineage")

    expected_train_ids = {
        str(row["sound_id"]) for row in manifest["examples"] if row["split"] == "train"
    }
    base_ids = [str(item) for item in base_train_index["sound_ids"]]
    added_ids = [str(item) for item in added_index["sound_ids"]]
    if (
        set(base_ids) & set(added_ids)
        or set(base_ids) | set(added_ids) != expected_train_ids
        or set(base_ids)
        != {
            str(row["sound_id"])
            for row in json.loads(combined_manifest_path.read_text(encoding="utf-8"))["examples"]
            if row["split"] == "train" and row["license"] != "CC-BY-3.0"
        }
        or set(added_ids)
        != {
            str(row["sound_id"])
            for row in additions_manifest["examples"]
            if row["split"] == "train"
        }
    ):
        raise ValueError("feature IDs do not exactly match the frozen train manifests")

    merged_ids = sorted(base_ids + added_ids, key=int)
    duration_by_id = dict(zip(base_ids, base_train_index["durations_seconds"], strict=True))
    duration_by_id.update(dict(zip(added_ids, added_index["durations_seconds"], strict=True)))
    values_by_id = {sound_id: base_train_values[index] for index, sound_id in enumerate(base_ids)}
    values_by_id.update({sound_id: added_values[index] for index, sound_id in enumerate(added_ids)})
    merged_values = np.stack([values_by_id[sound_id] for sound_id in merged_ids])
    merged_index = {
        "split": "train",
        "sound_ids": merged_ids,
        "durations_seconds": [duration_by_id[sound_id] for sound_id in merged_ids],
        "window_count": int(base_train_index["window_count"] + added_index["window_count"]),
    }
    merged_report = dict(base_train_report)
    merged_report.update(
        {
            "split_manifest_sha256": manifest_hash,
            "records": len(merged_ids),
            "combined_from_frozen_base_train": sha256_file(base_train_dir / "run.json"),
            "combined_from_ccby3_addition": sha256_file(additions_dir / "run.json"),
            "labels_loaded": False,
            "metrics_computed": False,
        }
    )

    expected_validation_ids = {
        str(row["sound_id"]) for row in manifest["examples"] if row["split"] == "validation"
    }
    if set(validation_index["sound_ids"]) != expected_validation_ids:
        raise ValueError("the frozen validation feature IDs changed")
    validation_copy_report = dict(validation_report)
    validation_copy_report.update(
        {
            "split_manifest_sha256": manifest_hash,
            "reattached_to_combined_manifest": True,
            "source_validation_feature_sha256": validation_report["feature_sha256"],
            "labels_loaded": False,
            "metrics_computed": False,
        }
    )
    _write_set(output_train_dir, merged_report, merged_index, merged_values)
    _write_set(
        output_validation_dir, validation_copy_report, validation_index, validation_values.copy()
    )
    return {
        "status": "PASS_FEATURE_EXTENSION",
        "combined_manifest_sha256": manifest_hash,
        "train_examples": len(merged_ids),
        "added_train_examples": len(added_ids),
        "validation_examples": len(validation_index["sound_ids"]),
        "validation_feature_values_unchanged": True,
        "feature_encoder_sha256": base_train_report["weights_sha256"],
        "sealed_audit_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-manifest", type=Path, required=True)
    parser.add_argument("--additions-manifest", type=Path, required=True)
    parser.add_argument("--base-train-features", type=Path, required=True)
    parser.add_argument("--addition-features", type=Path, required=True)
    parser.add_argument("--base-validation-features", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_feature_sets(
        args.combined_manifest.resolve(strict=True),
        args.additions_manifest.resolve(strict=True),
        args.base_train_features.resolve(strict=True),
        args.addition_features.resolve(strict=True),
        args.base_validation_features.resolve(strict=True),
        args.output_train.resolve(),
        args.output_validation.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
