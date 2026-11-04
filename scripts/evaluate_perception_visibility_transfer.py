"""Apply the frozen AV readout once to the unseen train-video transfer cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import numpy._core.multiarray as numpy_multiarray
except ImportError:
    import numpy.core.multiarray as numpy_multiarray

from tiny_omni_decision.perception_visibility_evaluation import (
    paired_bootstrap_accuracy_delta,
    score_binary_logits,
)

EXPECTED_READOUT_RESULT_SHA256 = "95ee830089d9211f2b7d68929eda2020e02be958b77331ebd18c29636986a224"
EXPECTED_HEAD_SHA256 = {
    "audio": "3ac71d79327f6f992022950e3e2f65791e3a0ac6c4c9f18e2c3d1754feb81bf1",
    "video": "0f5895627d54143dffc482da96699609fc24164663d65a6af78ef44af1231df5",
    "joint": "8ae155133858d1b768613a3782d2ce2e5c28f54c2179377ce50ae48588719ee2",
}
EXPECTED_NORMALIZATION_SHA256 = "ddcad9c498cecb43d3faa65c55bd3305e75fc19cf99e7e19c80344ef15ac16f6"
MODELS = ("audio", "video", "joint")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> Any:
    import torch

    safe_globals: list[Any] = [numpy_multiarray._reconstruct, np.ndarray, np.dtype]
    if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float64DType"):
        safe_globals.append(np.dtypes.Float64DType)
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_feature_rows(
    result_path: Path, manifest_sha: str
) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    result_sha = sha256_file(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"feature result belongs to another cohort: {result_path}")
    records = result.get("records")
    if not isinstance(records, list):
        raise ValueError(f"feature result has no records: {result_path}")
    by_video = {row["video_id"]: row for row in records}
    if len(by_video) != len(records):
        raise ValueError(f"feature result repeats video groups: {result_path}")
    return by_video, result_sha, result


def run(
    *,
    cohort_path: Path,
    fit_cohort_path: Path,
    inference_manifest_path: Path,
    audio_result_path: Path,
    video_result_path: Path,
    readout_dir: Path,
) -> dict[str, Any]:
    cohort_sha = sha256_file(cohort_path)
    manifest_sha = sha256_file(inference_manifest_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    fit_cohort = json.loads(fit_cohort_path.read_text(encoding="utf-8"))
    fit_ids = {row["video_id"] for row in fit_cohort["splits"]["train"]}
    examples = cohort.get("examples", [])
    if (
        cohort.get("source_split") != "train"
        or cohort.get("fit_video_overlap") != 0
        or len(examples) != 256
        or len({row["video_id"] for row in examples}) != 256
        or any(row["video_id"] in fit_ids for row in examples)
        or len(fit_ids) != 256
    ):
        raise ValueError("transfer cohort split/group integrity check failed")
    if len(manifest.get("train", [])) != 256 or manifest.get("validation") != []:
        raise ValueError("inference manifest must contain exactly the train-only transfer cohort")
    if manifest.get("metadata", {}).get("cohort_sha256") != cohort_sha:
        raise ValueError("inference manifest cohort hash mismatch")
    if manifest.get("metadata", {}).get("labels_used_for_selection") is not True:
        raise ValueError("selection-label disclosure missing")
    manifest_by_video = {row["video_id"]: row for row in manifest["train"]}
    if set(manifest_by_video) != {row["video_id"] for row in examples}:
        raise ValueError("inference manifest video groups differ from frozen cohort")
    if any(
        int(manifest_by_video[row["video_id"]]["target"]) != int(row["target"])
        or int(manifest_by_video[row["video_id"]]["video_event_id"])
        != int(row["event_id"])
        for row in examples
    ):
        raise ValueError("inference manifest labels/events differ from frozen cohort")

    audio_rows, audio_features_sha, audio_feature_receipt = load_feature_rows(
        audio_result_path, manifest_sha
    )
    video_rows, video_features_sha, video_feature_receipt = load_feature_rows(
        video_result_path, manifest_sha
    )
    if set(audio_rows) != {row["video_id"] for row in examples}:
        raise ValueError("audio features do not exactly match transfer cohort")
    if set(video_rows) != set(audio_rows):
        raise ValueError("audio/video feature group identities differ")

    result_path = readout_dir / "result.json"
    if sha256_file(result_path) != EXPECTED_READOUT_RESULT_SHA256:
        raise ValueError("frozen readout result hash mismatch")
    readout = json.loads(result_path.read_text(encoding="utf-8"))
    if readout.get("status") != "PASS_MATCHED_FROZEN_AV_READOUT":
        raise ValueError("frozen readout status is not accepted")
    normalization_path = readout_dir / "normalization.pt"
    if sha256_file(normalization_path) != EXPECTED_NORMALIZATION_SHA256:
        raise ValueError("frozen normalization hash mismatch")
    normalization = safe_torch_load(normalization_path)
    heads = {}
    head_hashes = {}
    for model in MODELS:
        path = readout_dir / f"{model}_head.pt"
        digest = sha256_file(path)
        if digest != EXPECTED_HEAD_SHA256[model]:
            raise ValueError(f"frozen {model} head hash mismatch")
        head_hashes[model] = digest
        heads[model] = safe_torch_load(path)

    means = {
        key: as_numpy(normalization[key]["mean"]).astype(np.float64) for key in ("audio", "video")
    }
    stds = {
        key: as_numpy(normalization[key]["std"]).astype(np.float64) for key in ("audio", "video")
    }
    if any(not np.isfinite(std).all() or np.any(std < 1e-6) for std in stds.values()):
        raise ValueError("frozen train-only normalization contains invalid standard deviations")

    targets = np.asarray([int(row["target"]) for row in examples], dtype=np.int64)
    video_ids = [str(row["video_id"]) for row in examples]
    features_audio = []
    features_video = []
    predictions_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    for event in examples:
        video_id = event["video_id"]
        audio_record = audio_rows[video_id]
        video_record = video_rows[video_id]
        if int(audio_record["event_id"]) != int(event["event_id"]):
            raise ValueError(f"audio event identity mismatch: {video_id}")
        if int(video_record["event_id"]) != int(event["event_id"]):
            raise ValueError(f"video event identity mismatch: {video_id}")
        audio_vector = np.load(audio_record["feature_path"], allow_pickle=False)
        video_vector = np.load(video_record["feature_path"], allow_pickle=False)
        if audio_vector.shape != (384,) or video_vector.shape != (768,):
            raise ValueError(f"unexpected feature shape: {video_id}")
        if not np.isfinite(audio_vector).all() or not np.isfinite(video_vector).all():
            raise ValueError(f"nonfinite feature vector: {video_id}")
        if (
            hashlib.sha256(np.ascontiguousarray(audio_vector).tobytes()).hexdigest()
            != audio_record["feature_sha256"]
        ):
            raise ValueError(f"audio feature hash mismatch: {video_id}")
        if (
            hashlib.sha256(np.ascontiguousarray(video_vector).tobytes()).hexdigest()
            != video_record["feature_sha256"]
        ):
            raise ValueError(f"video feature hash mismatch: {video_id}")
        audio_window = audio_record.get("audio_window_s")
        video_window = video_record.get("video_window_s")
        expected_audio_window = manifest_by_video[video_id]["options"][0]["audio_window_s"]
        expected_video_window = manifest_by_video[video_id]["video_window_s"]
        if (
            audio_window is None
            or video_window is None
            or not np.allclose(audio_window, video_window, atol=1e-6, rtol=0)
            or not np.allclose(audio_window, expected_audio_window, atol=1e-6, rtol=0)
            or not np.allclose(video_window, expected_video_window, atol=1e-6, rtol=0)
        ):
            raise ValueError(f"audio/video feature windows are not aligned: {video_id}")
        features_audio.append((audio_vector.astype(np.float64) - means["audio"]) / stds["audio"])
        features_video.append((video_vector.astype(np.float64) - means["video"]) / stds["video"])

    audio_matrix = np.stack(features_audio)
    video_matrix = np.stack(features_video)
    feature_matrices = {
        "audio": audio_matrix,
        "video": video_matrix,
        "joint": np.concatenate([audio_matrix, video_matrix], axis=1),
    }
    metrics: dict[str, Any] = {}
    for model in MODELS:
        weight = as_numpy(heads[model]["weight"]).astype(np.float64).reshape(-1)
        bias = as_numpy(heads[model]["bias"]).astype(np.float64).reshape(-1)
        if weight.shape != (feature_matrices[model].shape[1],) or bias.shape != (1,):
            raise ValueError(f"frozen {model} head shape mismatch")
        logits = feature_matrices[model] @ weight + bias[0]
        scored = score_binary_logits(logits, targets)
        predictions = scored.pop("predictions")
        probabilities = scored.pop("probability_visible")
        metrics[model] = scored
        for index, event in enumerate(examples):
            prediction = int(predictions[index])
            probability_visible = float(probabilities[index])
            predictions_by_model[model].append(
                {
                    "video_id": event["video_id"],
                    "event_id": int(event["event_id"]),
                    "target": int(event["target"]),
                    "prediction": prediction,
                    "probabilities": {
                        "not_visible": 1.0 - probability_visible,
                        "visible": probability_visible,
                    },
                }
            )

    correct = {
        model: np.asarray(
            [row["target"] == row["prediction"] for row in predictions_by_model[model]],
            dtype=np.int64,
        )
        for model in MODELS
    }
    paired = {
        "joint_minus_audio_accuracy": paired_bootstrap_accuracy_delta(
            correct["audio"], correct["joint"]
        ),
        "joint_minus_video_accuracy": paired_bootstrap_accuracy_delta(
            correct["video"], correct["joint"]
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS_FIXED_READOUT_TRAIN_TRANSFER_DIAGNOSTIC",
        "scope": (
            "Exploratory transfer within official TRAIN to video groups excluded from the fixed "
            "readout fit. Annotation labels were read for cohort selection. This is nonblind, "
            "not independent validation, and not a product result."
        ),
        "cohort_sha256": cohort_sha,
        "inference_manifest_sha256": manifest_sha,
        "fit_cohort_sha256": sha256_file(fit_cohort_path),
        "readout_result_sha256": EXPECTED_READOUT_RESULT_SHA256,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "head_sha256": head_hashes,
        "audio_feature_result_sha256": audio_features_sha,
        "video_feature_result_sha256": video_features_sha,
        "audio_feature_device": audio_feature_receipt.get("device"),
        "video_feature_device": video_feature_receipt.get("device"),
        "audio_feature_elapsed_seconds": audio_feature_receipt.get("elapsed_seconds"),
        "video_feature_elapsed_seconds": video_feature_receipt.get("elapsed_seconds"),
        "count": len(video_ids),
        "class_counts": {
            "not_visible": int((targets == 0).sum()),
            "visible": int((targets == 1).sum()),
        },
        "metrics": metrics,
        "paired_accuracy_bootstrap": paired,
        "per_example_predictions": predictions_by_model,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "validation_or_test_data_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--fit-cohort", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--audio-result", type=Path, required=True)
    parser.add_argument("--video-result", type=Path, required=True)
    parser.add_argument("--readout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    report = run(
        cohort_path=args.cohort.resolve(strict=True),
        fit_cohort_path=args.fit_cohort.resolve(strict=True),
        inference_manifest_path=args.inference_manifest.resolve(strict=True),
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
                "paired_accuracy_bootstrap": report["paired_accuracy_bootstrap"],
                "output": str(output),
                "sha256": sha256_file(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
