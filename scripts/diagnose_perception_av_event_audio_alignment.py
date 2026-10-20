"""Run fixed-checkpoint, CPU-only audio alignment interventions on one holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.analyze_av_event_readout_comparison import paired_video_bootstrap, score_rows
from scripts.audit_perception_av_correspondence import sha256_file
from scripts.evaluate_perception_av_event_fresh_holdout import (
    JOINT_DIR,
    _load_checkpoint,
    _load_features,
)

FEATURES_SHA256 = "5a4fed0a0419fe1091364b5b240b371e47c64ffff8516a2fe179eaac998fd1a4"
JOINT_CHECKPOINT_SHA256 = "be9a09f1c57c679b176d69c9a14825d4d38d053bee4b2ea5818ee108b7da8a36"
VIDEO_ONLY_PREDICTIONS_SHA256 = (
    "4acac4d853302a369aca1aabbe3368bf923a0a0ee41fcf50a4a8a74f3a745a71"
)
BASELINE_EVAL_DIR = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-av-event-fresh-holdout-eval-01"
)


def assign_cross_video_audio_donors(rows: list[dict]) -> dict[str, str]:
    """Map each event to an audio donor event from a different video group."""
    events_by_video: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        event_id, video_id = row.get("id"), row.get("video_id")
        if not isinstance(event_id, str) or not event_id or event_id in seen_ids:
            raise ValueError("rows need unique nonempty event IDs")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("rows need nonempty video IDs")
        seen_ids.add(event_id)
        events_by_video[video_id].append(event_id)
    if len(events_by_video) < 2:
        raise ValueError("audio shuffling needs at least two video groups")

    def order_key(video_id: str) -> bytes:
        return hashlib.sha256(
            f"perception-av-event-v1:cross-video-audio-donor:17:{video_id}".encode()
        ).digest()

    videos = sorted(events_by_video, key=order_key)
    for event_ids in events_by_video.values():
        event_ids.sort()
    donor_for_video = {
        video_id: videos[(index + 1) % len(videos)]
        for index, video_id in enumerate(videos)
    }
    assignment: dict[str, str] = {}
    for video_id, event_ids in events_by_video.items():
        donor_ids = events_by_video[donor_for_video[video_id]]
        for index, event_id in enumerate(event_ids):
            assignment[event_id] = donor_ids[index % len(donor_ids)]
    return assignment


def _predict(model, rows: list[dict], audio_by_id: dict[str, np.ndarray]) -> list[dict]:
    import torch

    predictions = []
    with torch.inference_mode():
        for row in rows:
            logits = model(
                condition="joint",
                audio_features=torch.from_numpy(audio_by_id[row["id"]]),
                visual_features=torch.from_numpy(row["visual"]),
            ).float()
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


def evaluate(features_path: Path, expected_features_sha256: str, output_dir: Path) -> dict:
    if expected_features_sha256 != FEATURES_SHA256:
        raise ValueError("feature hash argument differs from the fixed holdout hash")
    if sha256_file(features_path) != FEATURES_SHA256:
        raise ValueError("fresh holdout feature manifest hash mismatch")
    if output_dir.exists():
        raise ValueError("new output directory required; existing artifacts are immutable")

    rows = _load_features(features_path)
    donor_ids = assign_cross_video_audio_donors(rows)
    audio_by_id = {row["id"]: row["audio"] for row in rows}
    shuffled_audio = {event_id: audio_by_id[donor_id] for event_id, donor_id in donor_ids.items()}
    zero_audio = {row["id"]: np.zeros_like(row["audio"]) for row in rows}

    fit_result_path = JOINT_DIR / "result.json"
    fit_result_hash = json.loads(fit_result_path.read_text(encoding="utf-8"))[
        "feature_manifest_sha256"
    ]
    model, _, checkpoint_path, fit_result, _ = _load_checkpoint(
        JOINT_DIR, "joint", fit_result_hash
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != JOINT_CHECKPOINT_SHA256:
        raise ValueError("fixed joint checkpoint hash changed")
    if fit_result["feature_manifest_sha256"] != fit_result_hash:
        raise ValueError("joint fit feature source changed")

    started = time.perf_counter()
    predictions = {
        "joint_aligned": _predict(model, rows, audio_by_id),
        "joint_zero_audio": _predict(model, rows, zero_audio),
        "joint_cross_video_audio": _predict(model, rows, shuffled_audio),
    }
    elapsed_seconds = time.perf_counter() - started

    baseline_path = BASELINE_EVAL_DIR / "video_only_matched-predictions.jsonl"
    if sha256_file(baseline_path) != VIDEO_ONLY_PREDICTIONS_SHA256:
        raise ValueError("fixed video-only baseline prediction hash changed")
    video_only = [json.loads(line) for line in baseline_path.read_text().splitlines() if line]
    if {row["id"] for row in video_only} != {row["id"] for row in rows}:
        raise ValueError("video-only and intervention event IDs differ")

    output_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    metrics = {}
    for name, prediction_rows in predictions.items():
        path = output_dir / f"{name}-predictions.jsonl"
        _write_jsonl(path, prediction_rows)
        files[path.name] = sha256_file(path)
        metrics[name] = score_rows(prediction_rows)
    metrics["video_only_matched_reference"] = score_rows(video_only)

    receipt = {
        "status": "PASS_FIXED_CHECKPOINT_AUDIO_ALIGNMENT_DIAGNOSTIC",
        "scope": "exploratory fixed-checkpoint intervention on previously unused TRAIN videos",
        "feature_manifest_sha256": FEATURES_SHA256,
        "joint_checkpoint_sha256": checkpoint_sha,
        "video_only_prediction_sha256": VIDEO_ONLY_PREDICTIONS_SHA256,
        "events": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "interventions": {
            "joint_aligned": "original audio event and candidate crops",
            "joint_zero_audio": (
                "replace audio feature with zeros; OOD ablation, not a missing-modality token"
            ),
            "joint_cross_video_audio": (
                "deterministic SHA-256 ordered cyclic donor video; each event receives audio "
                "from a different video group"
            ),
        },
        "metrics": metrics,
        "paired_video_bootstrap": {
            "aligned_minus_zero_audio": paired_video_bootstrap(
                predictions["joint_aligned"], predictions["joint_zero_audio"], draws=10_000, seed=17
            ),
            "aligned_minus_cross_video_audio": paired_video_bootstrap(
                predictions["joint_aligned"],
                predictions["joint_cross_video_audio"],
                draws=10_000,
                seed=17,
            ),
            "aligned_minus_video_only": paired_video_bootstrap(
                predictions["joint_aligned"], video_only, draws=10_000, seed=17
            ),
        },
        "training": False,
        "optimizer_updates": 0,
        "checkpoint_selection": False,
        "gpu_used": False,
        "sealed_or_test_data_opened": False,
        "elapsed_inference_seconds": elapsed_seconds,
        "files_sha256": files,
        "warning": (
            "Input sensitivity is not proof of task correctness or causal necessity. "
            "This post-hoc diagnostic does not tune or select a model and must not be "
            "treated as an independent blind evaluation."
        ),
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = evaluate(args.features, args.features_sha256, args.output)
    print(json.dumps({"status": receipt["status"], "metrics": receipt["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
