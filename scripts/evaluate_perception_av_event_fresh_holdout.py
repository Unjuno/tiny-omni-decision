"""Evaluate fixed, already-selected AV readouts once on fresh TRAIN videos."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.analyze_av_event_readout_comparison import paired_video_bootstrap, score_rows
from scripts.audit_perception_av_correspondence import sha256_file

JOINT_DIR = Path("C:/CodexArtifacts/pretrained-reuse-r2/perception-av-event-readout-joint-02")
VIDEO_DIR = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-av-event-readout-video-only-matched-01"
)
SOURCE_FEATURE_HASH = "d92e4c91c236c69548f8fa0b58826e8afc853372ebf7a59b1e64567034f8df57"


def _load_features(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(row.get("split") != "fresh_holdout" for row in rows):
        raise ValueError("feature manifest must contain only fresh_holdout rows")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("fresh holdout event IDs must be unique")
    video_splits = {row["video_id"]: row["split"] for row in rows}
    if len(video_splits) < 2:
        raise ValueError("holdout must contain multiple video clusters")

    def load_array(path_value: str, expected_sha256: str, name: str) -> np.ndarray:
        feature_path = Path(path_value)
        if not feature_path.is_file() or sha256_file(feature_path) != expected_sha256:
            raise ValueError(f"{name} feature file is missing or hash-mismatched")
        values = np.load(feature_path, allow_pickle=False)
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{name} features must be finite numeric arrays")
        return np.asarray(values, dtype=np.float32)

    loaded = []
    for row in rows:
        audio = load_array(row["audio_feature_path"], row["audio_feature_sha256"], "audio")
        vision_tokens = load_array(
            row["visual_feature_path"], row["visual_feature_sha256"], "visual"
        )
        if vision_tokens.ndim != 3 or vision_tokens.shape[1:] != (257, 384):
            raise ValueError("DINOv2 features must have shape [candidates, 257, 384]")
        visual = vision_tokens[:, 0, :].copy()
        if (
            audio.shape != (512,)
            or visual.shape != (row["candidate_count"], 384)
            or visual.shape[0] < 2
        ):
            raise ValueError("feature shapes do not match the pinned readout architecture")
        if not 0 <= row["target_index"] < row["candidate_count"]:
            raise ValueError("target is outside supplied candidates")
        loaded.append({**row, "audio": audio, "visual": visual})
    return sorted(loaded, key=lambda row: row["id"])


def _load_checkpoint(run_dir: Path, expected_condition: str, feature_hash: str):
    import torch

    from tiny_omni_decision.av_event import AVEventSourceScorer

    result_path = run_dir / "result.json"
    checkpoint_path = run_dir / "best.pt"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    if (
        result.get("status") != "PASS_AV_EVENT_READOUT_FIT"
        or result.get("condition") != expected_condition
        or result.get("feature_manifest_sha256") != feature_hash
        or result.get("checkpoint_reload_verified") is not True
        or result.get("artifacts_sha256", {}).get("best.pt") != checkpoint_sha
    ):
        raise ValueError(f"{expected_condition} selected checkpoint/result integrity failed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("condition") != expected_condition or checkpoint.get("epoch") != result.get(
        "best_epoch"
    ):
        raise ValueError("selected checkpoint metadata differs from training receipt")
    model = AVEventSourceScorer(hidden_size=result["config"]["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    active_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if expected_condition == "joint"
        or name.startswith(("visual_projection.", "video_head."))
    )
    if active_parameters != result.get("trainable_parameters"):
        raise ValueError("readout parameter count differs from the frozen training receipt")
    model.eval().requires_grad_(False)
    return (
        model,
        result_path,
        checkpoint_path,
        result,
        {
            "total_model_parameters": actual_parameters,
            "active_condition_parameters": active_parameters,
        },
    )


def _predict(model, rows: list[dict], condition: str) -> list[dict]:
    import torch

    predictions = []
    with torch.inference_mode():
        for row in rows:
            inputs = {"visual_features": torch.from_numpy(row["visual"])}
            if condition == "joint":
                inputs["audio_features"] = torch.from_numpy(row["audio"])
            logits = model(condition=condition, **inputs).float()
            probabilities = logits.softmax(-1).cpu().numpy().astype(np.float64)
            predictions.append(
                {
                    "id": row["id"],
                    "video_id": row["video_id"],
                    "sound_id": row["sound_id"],
                    "target_index": row["target_index"],
                    "prediction_index": int(probabilities.argmax()),
                    "candidate_count": row["candidate_count"],
                    "candidate_order_sha256": row["candidate_order_sha256"],
                    "probabilities": probabilities.tolist(),
                }
            )
    return predictions


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate(features_path: Path, expected_feature_hash: str, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("fresh evaluation output directory required")
    feature_hash = sha256_file(features_path)
    if feature_hash != expected_feature_hash:
        raise ValueError("fresh holdout feature manifest hash mismatch")
    rows = _load_features(features_path)
    results = {}
    for name, run_dir, condition in (
        ("joint", JOINT_DIR, "joint"),
        ("video_only_matched", VIDEO_DIR, "video_only"),
    ):
        model, result_path, checkpoint_path, fit_result, parameter_counts = _load_checkpoint(
            run_dir, condition, SOURCE_FEATURE_HASH
        )
        # Reload the same immutable checkpoint and require exact inference replay.
        first = _predict(model, rows, condition)
        replay_model, _, _, _, _ = _load_checkpoint(
            run_dir,
            condition,
            fit_result["feature_manifest_sha256"],
        )
        replay = _predict(replay_model, rows, condition)
        if first != replay:
            raise ValueError(f"{condition} selected checkpoint replay is not exact")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / f"{name}-predictions.jsonl", first)
        results[name] = {
            "metrics": score_rows(first),
            "metrics_by_candidate_count": {
                str(count): score_rows([row for row in first if row["candidate_count"] == count])
                for count in sorted({row["candidate_count"] for row in first})
            },
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_result_sha256": sha256_file(result_path),
            "best_epoch": fit_result["best_epoch"],
            "trainable_parameters": fit_result["trainable_parameters"],
            **parameter_counts,
            "predictions_sha256": sha256_file(output_dir / f"{name}-predictions.jsonl"),
        }
        del model, replay_model

    joint_predictions = [
        json.loads(line)
        for line in (output_dir / "joint-predictions.jsonl").read_text().splitlines()
    ]
    video_predictions = [
        json.loads(line)
        for line in (output_dir / "video_only_matched-predictions.jsonl").read_text().splitlines()
    ]
    results["joint_minus_video_only_matched_video_bootstrap"] = paired_video_bootstrap(
        joint_predictions, video_predictions, draws=10_000, seed=17
    )
    uniform_rows = [
        {
            **row,
            "probabilities": [1.0 / row["candidate_count"]] * row["candidate_count"],
        }
        for row in joint_predictions
    ]
    results["uniform"] = score_rows(uniform_rows)
    by_video = defaultdict(int)
    for row in rows:
        by_video[row["video_id"]] += 1
    receipt = {
        "status": "PASS_FRESH_FIXED_CHECKPOINT_EVALUATION",
        "task": "Perception Test annotated audible event to visible object track selection",
        "interpretation": (
            "Nonblind TRAIN-subset holdout from IDs/assets unused by prior runs; "
            "source annotation archives were previously parsed for feasibility. "
            "Participant-level disjointness is unknown."
        ),
        "feature_manifest_sha256": feature_hash,
        "events": len(rows),
        "videos": len(by_video),
        "models": results,
        "optimizer_updates": 0,
        "checkpoint_selection": False,
        "training": False,
        "gpu_used": False,
        "sealed_or_test_data_opened": False,
    }
    path = output_dir / "result.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.features, args.features_sha256, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "events": result["events"],
                "videos": result["videos"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
