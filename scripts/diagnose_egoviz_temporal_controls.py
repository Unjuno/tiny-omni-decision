"""Inference-only order/missing-input controls on a selected EgoViz CPU readout.

These transform cached feature positions, not raw video or IMU inputs. They do
not test the upstream encoders' response to reversed raw media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CONTROLS = (
    "unaltered",
    "reverse_video_features",
    "reverse_sensor_features",
    "reverse_both_features",
    "shift_sensor_features_one_frame",
    "missing_video_features",
    "missing_sensor_features",
)


def control_spec(name: str) -> dict[str, Any]:
    """Fixed mappings: two adjacent sensor patches correspond to one video slot."""
    if name not in CONTROLS:
        raise ValueError(f"unknown temporal control: {name}")
    video = list(range(8))
    sensor = list(range(16))
    if name in ("reverse_video_features", "reverse_both_features"):
        video.reverse()
    if name in ("reverse_sensor_features", "reverse_both_features"):
        sensor.reverse()
    if name == "shift_sensor_features_one_frame":
        sensor = sensor[-2:] + sensor[:-2]
    return {
        "video_frame_indices": video,
        "sensor_patch_indices": sensor,
        "zero_video": name == "missing_video_features",
        "zero_sensor": name == "missing_sensor_features",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnose(
    corpus: Path, video_cache: Path, sensor_cache: Path, result_dir: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    import torch

    from scripts.train_egoviz_vs_sync_readout import (
        JointTemporalScorer,
        _group_bootstrap_delta,
        _load_samples,
        _tensor_data,
        metrics,
        uniform_metrics,
    )

    report_path = result_dir / "readout-results.json"
    selected = json.loads(report_path.read_text(encoding="utf-8"))
    if selected.get("sealed_audit_accessed") is not False:
        raise ValueError("selected run does not attest to audit isolation")
    observed_hashes = {
        "train_samples": _sha256(corpus / "train-samples.jsonl"),
        "development_samples": _sha256(corpus / "development-samples.jsonl"),
        "video_features": _sha256(video_cache),
        "sensor_features": _sha256(sensor_cache),
    }
    if observed_hashes != selected["input_sha256"]:
        raise ValueError("diagnostic inputs differ from selected run")
    checkpoint = result_dir / "joint-temporal-state.pt"
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != selected["joint_temporal"]["state_sha256"]:
        raise ValueError("selected checkpoint hash mismatch")
    rows, video_by_id, sensor_features = _load_samples(corpus, video_cache, sensor_cache)
    video, candidates, targets, groups, sample_ids, _ = _tensor_data(
        rows, video_by_id, sensor_features, "development"
    )
    prediction_path = result_dir / "development-predictions.jsonl"
    saved_rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    saved = {row["sample_id"]: row for row in saved_rows}
    if len(saved_rows) != len(sample_ids) or set(saved) != set(sample_ids) or any(
        saved[key]["target"] != int(targets[index]) for index, key in enumerate(sample_ids)
    ):
        raise ValueError("saved development IDs/targets differ")
    expected_probabilities = np.asarray([saved[key]["joint_probabilities"] for key in sample_ids])
    model = JointTemporalScorer().cpu()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    logits_by_control: dict[str, np.ndarray] = {}
    for name in CONTROLS:
        spec = control_spec(name)
        video_indices = torch.tensor(spec["video_frame_indices"])
        sensor_indices = torch.tensor(spec["sensor_patch_indices"])
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(targets), 32):
                v = video[start : start + 32].index_select(1, video_indices)
                s = candidates[start : start + 32].index_select(3, sensor_indices)
                if spec["zero_video"]:
                    v = torch.zeros_like(v)
                if spec["zero_sensor"]:
                    s = torch.zeros_like(s)
                predictions.append(model(v, s).numpy())
        logits = np.concatenate(predictions)
        if not np.isfinite(logits).all():
            raise ValueError(f"non-finite logits for {name}")
        logits_by_control[name] = logits

    def probabilities(logits: np.ndarray) -> np.ndarray:
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    baseline = logits_by_control["unaltered"]
    baseline_difference = float(
        np.max(np.abs(probabilities(baseline) - expected_probabilities))
    )
    if baseline_difference > 1e-6:
        raise ValueError("unaltered diagnostic inference differs from saved probabilities")
    np.testing.assert_allclose(
        logits_by_control["reverse_both_features"], baseline, rtol=1e-5, atol=1e-6
    )
    controls = {}
    predictions = []
    y = targets.numpy()
    for name, logits in logits_by_control.items():
        probs = probabilities(logits)
        tied = np.ptp(logits, axis=1) <= 1e-8
        item = {
            "transform": control_spec(name),
            "metrics_argmax_first_tie_break": metrics(logits, y),
            "all_options_tied_fraction": float(tied.mean()),
            "max_probability_difference_from_unaltered": float(
                np.max(np.abs(probs - probabilities(baseline)))
            ),
            "prediction_changed_fraction": float(
                (logits.argmax(axis=1) != baseline.argmax(axis=1)).mean()
            ),
            "unaltered_minus_control_group_bootstrap": _group_bootstrap_delta(
                baseline, logits, y, groups, seed=17, repetitions=2000
            ),
        }
        if name.startswith("missing_"):
            np.testing.assert_allclose(logits, 0, atol=1e-7, rtol=0)
            item["uniform_expected_random_tie_metrics"] = uniform_metrics(len(y))
        controls[name] = item
        for index, key in enumerate(sample_ids):
            predictions.append(
                {
                    "control": name,
                    "sample_id": key,
                    "group_id": str(groups[index]),
                    "target": int(y[index]),
                    "prediction_argmax_first_tie_break": int(probs[index].argmax()),
                    "probabilities": probs[index].tolist(),
                }
            )
    if _sha256(checkpoint) != checkpoint_hash:
        raise ValueError("selected checkpoint changed during inference")
    result = {
        "status": "FIXED_CHECKPOINT_FEATURE_ORDER_DIAGNOSTIC_NOT_RAW_MEDIA_EVALUATION",
        "sample_count": len(sample_ids),
        "development_groups": len(np.unique(groups)),
        "input_sha256": observed_hashes,
        "checkpoint_sha256": checkpoint_hash,
        "selected_report_sha256": _sha256(report_path),
        "saved_prediction_sha256": _sha256(prediction_path),
        "runner_sha256": _sha256(Path(__file__)),
        "device": "cpu",
        "batch_size": 32,
        "training": False,
        "calibration_fit": False,
        "checkpoint_selection": False,
        "sealed_audit_accessed": False,
        "baseline_probability_max_abs_difference": baseline_difference,
        "scorer_logit_range_theoretical": [-1, 1],
        "maximum_four_choice_confidence_theoretical": float(1 / (1 + 3 * np.exp(-2))),
        "controls": controls,
        "limitations": [
            "Feature-index transforms do not reverse or re-encode the original media.",
            "Development data already selected the checkpoint; this is not a blind audit.",
            "Simultaneous reversal invariance is a scorer property, not learned robustness.",
            "Uniform missing-input metrics distinguish random ties from argmax tie-breaking.",
            "Single seed; session bootstrap is diagnostic and participant independence is unknown.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostic-results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in predictions),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--video-cache", type=Path, required=True)
    parser.add_argument("--sensor-cache", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(
        args.corpus, args.video_cache, args.sensor_cache, args.result_dir, args.output
    ), indent=2))


if __name__ == "__main__":
    main()
