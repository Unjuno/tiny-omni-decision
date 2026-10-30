"""Evaluate frozen AV visibility readouts once on the selected official-valid cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.perception_visibility_evaluation import score_binary_logits
from tiny_omni_decision.perception_visibility_validation import paired_metric_bootstrap

EXPECTED_READOUT_RESULT_SHA256 = "95ee830089d9211f2b7d68929eda2020e02be958b77331ebd18c29636986a224"
EXPECTED_HEAD_SHA256 = {
    "audio": "3ac71d79327f6f992022950e3e2f65791e3a0ac6c4c9f18e2c3d1754feb81bf1",
    "video": "0f5895627d54143dffc482da96699609fc24164663d65a6af78ef44af1231df5",
    "joint": "8ae155133858d1b768613a3782d2ce2e5c28f54c2179377ce50ae48588719ee2",
}
EXPECTED_NORMALIZATION_SHA256 = "ddcad9c498cecb43d3faa65c55bd3305e75fc19cf99e7e19c80344ef15ac16f6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> Any:
    import torch

    safe_globals: list[Any] = []
    try:
        import numpy._core.multiarray as numpy_multiarray
    except ImportError:
        import numpy.core.multiarray as numpy_multiarray
    safe_globals.extend([numpy_multiarray._reconstruct, np.ndarray, np.dtype])
    if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float64DType"):
        safe_globals.append(np.dtypes.Float64DType)
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def feature_map(
    path: Path, manifest_sha256: str, expected_status: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != expected_status:
        raise ValueError(f"feature extraction is incomplete or failed: {path}")
    if result.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"feature receipt belongs to another manifest: {path}")
    records = result.get("records")
    if not isinstance(records, list):
        raise ValueError(f"missing feature records: {path}")
    by_video = {row["video_id"]: row for row in records}
    if len(by_video) != len(records):
        raise ValueError(f"repeated video records: {path}")
    return by_video, result


def run(
    *,
    selection_path: Path,
    media_validation_path: Path,
    manifest_path: Path,
    fit_cohort_path: Path,
    audio_result_path: Path,
    video_result_path: Path,
    readout_dir: Path,
) -> dict[str, Any]:
    selection_sha = sha256_file(selection_path)
    media_sha = sha256_file(media_validation_path)
    manifest_sha = sha256_file(manifest_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    media_validation = json.loads(media_validation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = selection.get("events", [])
    rows = manifest.get("validation", [])
    metadata = manifest.get("metadata", {})
    if (
        selection.get("source_split") != "official validation only"
        or selection.get("official_test_split_opened") is not False
        or len(examples) != 200
        or len({row["video_id"] for row in examples}) != 200
        or sum(row["target"] == 0 for row in examples) != 100
        or sum(row["target"] == 1 for row in examples) != 100
    ):
        raise ValueError("frozen official-validation cohort integrity check failed")
    if (
        len(rows) != 200
        or manifest.get("train") != []
        or metadata.get("selection_sha256") != selection_sha
        or metadata.get("media_validation_sha256") != media_sha
        or metadata.get("validation_is_blind") is not False
    ):
        raise ValueError("validation inference manifest mismatch")
    media_rows = media_validation.get("examples_manifest", [])
    media_by_video = {row["video_id"]: row for row in media_rows}
    if len(media_by_video) != 200:
        raise ValueError("verified media receipt does not cover all 200 videos")
    fit = json.loads(fit_cohort_path.read_text(encoding="utf-8"))
    fit_ids = {row["video_id"] for split in fit["splits"].values() for row in split}
    if fit_ids & {row["video_id"] for row in examples}:
        raise ValueError("official-validation cohort overlaps frozen readout fit videos")

    audio_rows, audio_receipt = feature_map(
        audio_result_path, manifest_sha, "PASS_TEMPORAL_MATCH_EFFICIENTAT_FEATURES"
    )
    video_rows, video_receipt = feature_map(
        video_result_path, manifest_sha, "PASS_TEMPORAL_MATCH_VIDEOPRISM_FEATURES"
    )
    ids = [row["video_id"] for row in examples]
    if set(audio_rows) != set(ids) or set(video_rows) != set(ids):
        raise ValueError("extracted audio/video IDs do not exactly match the cohort")

    readout_result_path = readout_dir / "result.json"
    normalization_path = readout_dir / "normalization.pt"
    if sha256_file(readout_result_path) != EXPECTED_READOUT_RESULT_SHA256:
        raise ValueError("frozen readout result hash mismatch")
    if sha256_file(normalization_path) != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError("frozen normalization hash mismatch")
    readout = json.loads(readout_result_path.read_text(encoding="utf-8"))
    if readout.get("status") != "PASS_MATCHED_FROZEN_AV_READOUT":
        raise ValueError("unexpected frozen readout status")
    normalization = safe_torch_load(normalization_path)
    heads = {}
    head_hashes = {}
    for name in ("audio", "video", "joint"):
        path = readout_dir / f"{name}_head.pt"
        head_hashes[name] = sha256_file(path)
        if head_hashes[name] != EXPECTED_HEAD_SHA256[name]:
            raise ValueError(f"frozen {name} head hash mismatch")
        heads[name] = safe_torch_load(path)

    means = {
        name: as_numpy(normalization[name]["mean"]).astype(np.float64)
        for name in ("audio", "video")
    }
    stds = {
        name: as_numpy(normalization[name]["std"]).astype(np.float64) for name in ("audio", "video")
    }
    if any(not np.isfinite(value).all() or np.any(value < 1e-6) for value in stds.values()):
        raise ValueError("invalid frozen train-only normalization")

    audio_vectors = []
    video_vectors = []
    for event, row in zip(examples, rows, strict=True):
        if row["video_id"] != event["video_id"] or row["video_event_id"] != event["event_id"]:
            raise ValueError("manifest order or event identity changed")
        audio_record = audio_rows[event["video_id"]]
        video_record = video_rows[event["video_id"]]
        media_record = media_by_video[event["video_id"]]
        window = row["video_window_s"]
        if (
            audio_record.get("event_id") != event["event_id"]
            or video_record.get("event_id") != event["event_id"]
            or audio_record.get("media_audio_sha256") != media_record.get("audio_sha256")
            or video_record.get("media_video_sha256") != media_record.get("video_sha256")
            or not np.allclose(audio_record.get("audio_window_s"), window, atol=1e-6, rtol=0)
            or not np.allclose(video_record.get("video_window_s"), window, atol=1e-6, rtol=0)
            or not np.allclose(
                video_record.get("video_frame_times_s"),
                row["video_frame_times_s"],
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError(f"feature event/window mismatch: {event['video_id']}")
        av = np.load(audio_record["feature_path"], allow_pickle=False)
        vv = np.load(video_record["feature_path"], allow_pickle=False)
        if (
            av.shape != (384,)
            or vv.shape != (768,)
            or not np.isfinite(av).all()
            or not np.isfinite(vv).all()
        ):
            raise ValueError(f"invalid features: {event['video_id']}")
        if (
            hashlib.sha256(np.ascontiguousarray(av).tobytes()).hexdigest()
            != audio_record["feature_sha256"]
        ):
            raise ValueError(f"audio feature hash mismatch: {event['video_id']}")
        if (
            hashlib.sha256(np.ascontiguousarray(vv).tobytes()).hexdigest()
            != video_record["feature_sha256"]
        ):
            raise ValueError(f"video feature hash mismatch: {event['video_id']}")
        audio_vectors.append((av.astype(np.float64) - means["audio"]) / stds["audio"])
        video_vectors.append((vv.astype(np.float64) - means["video"]) / stds["video"])

    audio = np.stack(audio_vectors)
    video = np.stack(video_vectors)
    targets = np.asarray([row["target"] for row in examples], dtype=np.int64)
    weights = {
        name: as_numpy(heads[name]["weight"]).astype(np.float64).reshape(-1)
        for name in ("audio", "video", "joint")
    }
    biases = {
        name: as_numpy(heads[name]["bias"]).astype(np.float64).reshape(-1)[0]
        for name in ("audio", "video", "joint")
    }
    if (
        weights["audio"].shape != (384,)
        or weights["video"].shape != (768,)
        or weights["joint"].shape != (1152,)
    ):
        raise ValueError("frozen readout shape mismatch")
    joined = np.concatenate([audio, video], axis=1)
    # Donor assignment is a deterministic derangement independent of labels.
    donor_order = sorted(
        range(len(ids)), key=lambda i: hashlib.sha256(f"17:{ids[i]}".encode()).hexdigest()
    )
    shuffled_audio = audio.copy()
    for index, source_index in enumerate(donor_order):
        shuffled_audio[source_index] = audio[donor_order[(index + 1) % len(donor_order)]]
    if any(np.array_equal(shuffled_audio[i], audio[i]) for i in range(len(ids))):
        raise ValueError("audio shuffle unexpectedly retained a self-pair")

    logits = {
        "audio": audio @ weights["audio"] + biases["audio"],
        "video": video @ weights["video"] + biases["video"],
        "joint": joined @ weights["joint"] + biases["joint"],
        "audio_zero": np.zeros_like(audio) @ weights["audio"] + biases["audio"],
        "video_zero": np.zeros_like(video) @ weights["video"] + biases["video"],
        "joint_audio_zero": np.concatenate([np.zeros_like(audio), video], axis=1) @ weights["joint"]
        + biases["joint"],
        "joint_video_zero": np.concatenate([audio, np.zeros_like(video)], axis=1) @ weights["joint"]
        + biases["joint"],
        "joint_audio_shuffled": np.concatenate([shuffled_audio, video], axis=1) @ weights["joint"]
        + biases["joint"],
    }
    metrics = {name: score_binary_logits(value, targets) for name, value in logits.items()}
    paired = {
        "joint_minus_audio": paired_metric_bootstrap(logits["audio"], logits["joint"], targets),
        "joint_minus_video": paired_metric_bootstrap(logits["video"], logits["joint"], targets),
        "audio_minus_audio_zero": paired_metric_bootstrap(
            logits["audio_zero"], logits["audio"], targets
        ),
        "video_minus_video_zero": paired_metric_bootstrap(
            logits["video_zero"], logits["video"], targets
        ),
        "joint_minus_audio_zero": paired_metric_bootstrap(
            logits["joint_audio_zero"], logits["joint"], targets
        ),
        "joint_minus_video_zero": paired_metric_bootstrap(
            logits["joint_video_zero"], logits["joint"], targets
        ),
        "joint_minus_audio_shuffle": paired_metric_bootstrap(
            logits["joint_audio_shuffled"], logits["joint"], targets
        ),
    }
    predictions = {}
    for name, scored in metrics.items():
        probabilities = scored["probability_visible"]
        guesses = scored["predictions"]
        predictions[name] = [
            {
                "video_id": event["video_id"],
                "event_id": event["event_id"],
                "target": event["target"],
                "prediction": guess,
                "probabilities": {"not_visible": 1.0 - probability, "visible": probability},
            }
            for event, guess, probability in zip(examples, guesses, probabilities, strict=True)
        ]
    return {
        "schema_version": 1,
        "status": "PASS_ONE_TIME_FIXED_READOUT_OFFICIAL_VALIDATION_DIAGNOSTIC",
        "scope": (
            "Nonblind, label-balanced exploratory transfer to a frozen official-validation cohort. "
            "Source-level feasibility/selection already inspected; this is not a sealed audit, "
            "not a product result, and does not close the joint-quality gate."
        ),
        "selection_sha256": selection_sha,
        "media_validation_sha256": media_sha,
        "manifest_sha256": manifest_sha,
        "fit_cohort_sha256": sha256_file(fit_cohort_path),
        "readout_result_sha256": EXPECTED_READOUT_RESULT_SHA256,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "head_sha256": head_hashes,
        "audio_feature_result_sha256": sha256_file(audio_result_path),
        "video_feature_result_sha256": sha256_file(video_result_path),
        "audio_feature_device": audio_receipt.get("device"),
        "video_feature_device": video_receipt.get("device"),
        "count": len(ids),
        "class_counts": {
            "not_visible": int((targets == 0).sum()),
            "visible": int((targets == 1).sum()),
        },
        "boundary_window_shift_count": metadata.get("boundary_shift_count"),
        "label_stratified_selection": True,
        "metrics": metrics,
        "paired_video_bootstrap": paired,
        "per_example_predictions": predictions,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "calibration_performed": False,
        "official_test_or_sealed_audit_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "selection",
        "media-validation",
        "manifest",
        "fit-cohort",
        "audio-result",
        "video-result",
        "readout-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite one-time result: {output}")
    report = run(
        selection_path=args.selection.resolve(strict=True),
        media_validation_path=args.media_validation.resolve(strict=True),
        manifest_path=args.manifest.resolve(strict=True),
        fit_cohort_path=args.fit_cohort.resolve(strict=True),
        audio_result_path=args.audio_result.resolve(strict=True),
        video_result_path=args.video_result.resolve(strict=True),
        readout_dir=args.readout_dir.resolve(strict=True),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": report["metrics"],
                "output": str(output),
                "sha256": sha256_file(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
