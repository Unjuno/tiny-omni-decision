"""Run the frozen CLEVR-CoGenT valB audit once, without candidate reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or len({row.get("sample_id") for row in rows}) != len(rows):
        raise ValueError(f"manifest must contain unique non-empty sample IDs: {path}")
    return rows


def paired_bootstrap(
    predictions: list[dict[str, object]],
    control_predictions: list[dict[str, object]],
    *,
    replicates: int = 10_000,
    seed: int = 17,
) -> dict[str, object]:
    """Paired per-image percentile intervals; audit has one row per image."""
    if len(predictions) != len(control_predictions) or not predictions:
        raise ValueError("paired predictions must have equal non-zero row counts")
    ids = [row.get("sample_id") for row in predictions]
    if ids != [row.get("sample_id") for row in control_predictions]:
        raise ValueError("paired audit rows must have identical sample ID order")
    if [row.get("target_position") for row in predictions] != [
        row.get("target_position") for row in control_predictions
    ]:
        raise ValueError("paired audit rows must have identical targets")
    if [row.get("options") for row in predictions] != [
        row.get("options") for row in control_predictions
    ]:
        raise ValueError("paired audit rows must have identical candidate options")
    target = np.asarray([row["target_position"] for row in predictions], dtype=np.int64)

    def arrays(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        accuracy = []
        nll = []
        brier = []
        for row, expected in zip(rows, target, strict=True):
            probabilities = np.asarray(row["option_probabilities"], dtype=np.float64)
            if not 2 <= len(probabilities) <= 4 or not np.isfinite(probabilities).all():
                raise ValueError("invalid option probabilities in paired predictions")
            if not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
                raise ValueError("option probabilities must sum to one")
            if not 0 <= expected < len(probabilities):
                raise ValueError("target position is outside the candidate options")
            accuracy.append(float(int(probabilities.argmax()) == expected))
            nll.append(float(-np.log(np.clip(probabilities[expected], 1e-12, 1.0))))
            one_hot = np.zeros(len(probabilities), dtype=np.float64)
            one_hot[expected] = 1.0
            brier.append(float(np.square(probabilities - one_hot).sum()))
        return np.asarray(accuracy), np.asarray(nll), np.asarray(brier)

    real = arrays(predictions)
    control = arrays(control_predictions)
    deltas = np.stack([left - right for left, right in zip(real, control, strict=True)])
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty((replicates, 3), dtype=np.float64)
    # Chunking bounds temporary memory while keeping the bootstrap reproducible.
    chunk = 100
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, len(predictions), size=(stop - start, len(predictions)))
        samples[start:stop] = deltas[:, :].T[indices].mean(axis=1)
    names = ("accuracy_delta", "nll_delta", "brier_delta")
    return {
        "method": "paired per-image percentile bootstrap; one selected question per image",
        "replicates": replicates,
        "seed": seed,
        "unit_count": len(predictions),
        "delta_direction": "real image minus control",
        "interval_95_percent": {
            name: [float(value) for value in np.quantile(samples[:, index], [0.025, 0.975])]
            for index, name in enumerate(names)
        },
    }


def _load_feature(path: Path, metadata_path: Path, manifest_path: Path) -> np.ndarray:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "FROZEN_FEATURES_READY":
        raise ValueError(f"feature cache is not frozen/ready: {path}")
    if metadata.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError(f"feature metadata does not match manifest: {path}")
    if metadata.get("feature_sha256") != sha256(path):
        raise ValueError(f"feature cache hash mismatch: {path}")
    features = np.load(path, allow_pickle=False)
    if features.shape != (15_000, 768) or not np.isfinite(features).all():
        raise ValueError("audit features must be finite [15000, 768]")
    return features


def main() -> None:
    import torch

    from tiny_omni_decision.clevr_vqa_probe import evaluate_readout
    from tiny_omni_decision.clevr_vqa_readout import ClevrAnswerReadout

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "audit-manifest",
        "audit-metadata",
        "audit-features",
        "audit-feature-metadata",
        "candidate-freeze",
        "readout-checkpoint",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--control-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"one-time audit output already exists: {args.output}")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")

    freeze = json.loads(args.candidate_freeze.read_text(encoding="utf-8"))
    if not freeze.get("selection_frozen_before_cogent_b_access") or not freeze.get(
        "no_further_validation_tuning_after_freeze"
    ):
        raise ValueError("candidate selection was not frozen before CoGenT-B access")
    freeze_sha = sha256(args.candidate_freeze)
    artifacts = freeze["artifacts"]
    if sha256(args.readout_checkpoint) != artifacts["best_checkpoint"]["sha256"]:
        raise ValueError("readout checkpoint differs from the frozen selected checkpoint")
    base_path = Path(artifacts["base_checkpoint"]["path"])
    if sha256(base_path) != artifacts["base_checkpoint"]["sha256"]:
        raise ValueError("pinned ViLT base checkpoint hash mismatch")

    audit_meta = json.loads(args.audit_metadata.read_text(encoding="utf-8"))
    manifest_sha = sha256(args.audit_manifest)
    if (
        audit_meta.get("status") != "FROZEN_COGENT_B_AUDIT_MANIFEST"
        or audit_meta.get("manifest_sha256") != manifest_sha
        or audit_meta.get("candidate_selection_freeze_sha256") != freeze_sha
        or audit_meta.get("audit_examples") != 15_000
        or audit_meta.get("train_image_overlap") != 0
        or audit_meta.get("validation_image_overlap") != 0
        or audit_meta.get("normalized_question_overlap_with_development") != 0
    ):
        raise ValueError("audit manifest integrity/freeze metadata mismatch")
    rows = read_rows(args.audit_manifest)
    if len(rows) != 15_000 or len({row["image_filename"] for row in rows}) != 15_000:
        raise ValueError("audit must contain exactly one unique row per each valB image")
    if any(row["split"] != "cogent_b_audit" for row in rows):
        raise ValueError("audit manifest contains a non-audit split row")

    features = torch.from_numpy(
        _load_feature(args.audit_features, args.audit_feature_metadata, args.audit_manifest)
    )
    feature_metadata = json.loads(args.audit_feature_metadata.read_text(encoding="utf-8"))
    control_hashes = feature_metadata.get("controls", {}).get("control_sha256", {})
    control_features = {}
    for name, filename in (
        ("blank_image", "blank-image-features.npy"),
        ("shuffled_image", "shuffled-image-features.npy"),
    ):
        path = args.control_root / filename
        if control_hashes.get(filename) != sha256(path):
            raise ValueError(f"audit control cache hash mismatch: {path}")
        values = np.load(path, allow_pickle=False)
        if values.shape != (15_000, 768) or not np.isfinite(values).all():
            raise ValueError(f"invalid audit control feature matrix: {path}")
        control_features[name] = torch.from_numpy(values.astype(np.float32, copy=False))
    model = ClevrAnswerReadout()
    state = torch.load(args.readout_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    metrics, predictions = evaluate_readout(model, features, rows)
    control_results = {
        name: evaluate_readout(model, values, rows)
        for name, values in control_features.items()
    }

    args.output.mkdir(parents=True)
    (args.output / "audit-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    for name, (_control_metrics, control_predictions) in control_results.items():
        (args.output / f"audit-{name}-predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in control_predictions),
            encoding="utf-8",
        )
    result = {
        "status": "ONE_TIME_COGENT_B_AUDIT_COMPLETED",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "candidate_id": freeze["candidate_id"],
        "candidate_selection_freeze_sha256": freeze_sha,
        "selected_epoch": freeze["selected_epoch"],
        "base_checkpoint_sha256": artifacts["base_checkpoint"]["sha256"],
        "readout_checkpoint_sha256": sha256(args.readout_checkpoint),
        "audit_manifest_sha256": manifest_sha,
        "audit_feature_sha256": sha256(args.audit_features),
        "audit_examples": len(rows),
        "unique_images": len({row["image_filename"] for row in rows}),
        "metrics": metrics,
        "controls": {
            name: {
                "metrics": control_metrics,
                "paired_bootstrap_real_minus_control": paired_bootstrap(
                    predictions,
                    control_predictions,
                    replicates=args.bootstrap_replicates,
                    seed=17,
                ),
            }
            for name, (control_metrics, control_predictions) in control_results.items()
        },
        "limitations": [
            "single frozen readout candidate; no post-audit reselection or tuning",
            "CoGenT-B valB is an OOD audit for this image+question task, "
            "not a blind sealed product evaluation",
            "research-only: VQAv2-pretrained base provenance is not cleared for product rights",
        ],
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
