"""Fit and select a small readout over pinned, frozen CLEVR/ViLT features."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from scripts.extract_clevr_vqa_features import read_rows, sha256
from tiny_omni_decision.clevr_vqa_probe import fit_readout


def _load_features(feature_path: Path, metadata_path: Path, manifest_path: Path):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError(f"feature metadata does not match {manifest_path}")
    if metadata.get("feature_sha256") != sha256(feature_path):
        raise ValueError(f"feature cache hash mismatch: {feature_path}")
    values = np.load(feature_path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 768 or not np.isfinite(values).all():
        raise ValueError(f"invalid feature matrix: {feature_path}")
    return torch.from_numpy(values.astype(np.float32, copy=False)), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-manifest",
        "train-features",
        "train-feature-metadata",
        "validation-manifest",
        "validation-features",
        "validation-feature-metadata",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-control-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train_rows = read_rows(args.train_manifest)
    validation_rows = read_rows(args.validation_manifest)
    train_features, train_metadata = _load_features(
        args.train_features, args.train_feature_metadata, args.train_manifest
    )
    validation_features, validation_metadata = _load_features(
        args.validation_features,
        args.validation_feature_metadata,
        args.validation_manifest,
    )
    if len(train_rows) != len(train_features) or len(validation_rows) != len(validation_features):
        raise ValueError("feature and manifest row counts differ")
    train_images = {str(row["image_filename"]) for row in train_rows}
    validation_images = {str(row["image_filename"]) for row in validation_rows}
    if train_images & validation_images:
        raise ValueError("train and validation share CLEVR image identities")

    control_features = {}
    control_summary = None
    if args.validation_control_root is not None:
        control_summary = json.loads(
            (args.validation_control_root / "metadata.json").read_text(encoding="utf-8")
        )
        if control_summary.get("manifest_sha256") != sha256(args.validation_manifest):
            raise ValueError("validation control metadata does not match validation manifest")
        hashes = control_summary.get("controls", {}).get("control_sha256", {})
        for name, filename in (
            ("blank_image", "blank-image-features.npy"),
            ("shuffled_image", "shuffled-image-features.npy"),
        ):
            path = args.validation_control_root / filename
            if hashes.get(filename) != sha256(path):
                raise ValueError(f"validation control feature hash mismatch: {path}")
            values = np.load(path, allow_pickle=False)
            if values.shape != validation_features.shape or not np.isfinite(values).all():
                raise ValueError(f"invalid validation control feature matrix: {path}")
            control_features[name] = torch.from_numpy(values.astype(np.float32, copy=False))

    args.output.mkdir(parents=True)
    started = datetime.now(UTC)
    metadata = {
        "started_utc": started.isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "seed": 17,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW",
        "selector": "earliest minimum macro answer-kind validation NLL, then Brier",
        "manifest_sha256": {
            "train": sha256(args.train_manifest),
            "validation": sha256(args.validation_manifest),
        },
        "feature_sha256": {
            "train": sha256(args.train_features),
            "validation": sha256(args.validation_features),
        },
        "feature_extraction": {
            "train": train_metadata,
            "validation": validation_metadata,
        },
        "validation_control_features": (
            control_summary.get("controls") if control_summary is not None else None
        ),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "unique_train_images": len(train_images),
        "unique_validation_images": len(validation_images),
        "readout_trainable_parameters": 28 * 768 + 28,
        "device": "cpu",
        "torch_version": importlib.metadata.version("torch"),
    }
    (args.output / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = fit_readout(
        train_features,
        train_rows,
        validation_features,
        validation_rows,
        output_dir=args.output / "fit",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=17,
        device="cpu",
        validation_controls=control_features,
    )
    result["completed_utc"] = datetime.now(UTC).isoformat()
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
