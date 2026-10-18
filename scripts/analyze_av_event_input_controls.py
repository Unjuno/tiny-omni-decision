"""Run fixed-checkpoint AV input corruption diagnostics on development features."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.analyze_av_event_readout_comparison import (
    paired_video_bootstrap,
    score_rows,
)
from scripts.audit_perception_av_correspondence import sha256_file


def cross_video_audio_donors(rows: list[dict], seed: int = 17) -> dict[str, str]:
    """Map each event to audio from a deterministically chosen different video."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    ids = set()
    for row in rows:
        if row.get("id") in ids or not isinstance(row.get("id"), str):
            raise ValueError("event IDs must be nonempty and unique")
        if not isinstance(row.get("video_id"), str) or not row["video_id"]:
            raise ValueError("every event needs a video ID")
        ids.add(row["id"])
        by_video[row["video_id"]].append(row)
    if len(by_video) < 2:
        raise ValueError("cross-video audio shuffle requires multiple videos")
    ordered_videos = sorted(
        by_video,
        key=lambda video_id: hashlib.sha256(
            f"perception-av-event-cross-video-audio:{seed}:{video_id}".encode()
        ).digest(),
    )
    donors = {}
    for index, recipient_video in enumerate(ordered_videos):
        donor_video = ordered_videos[(index + 1) % len(ordered_videos)]
        donor_events = sorted(by_video[donor_video], key=lambda row: row["id"])
        for recipient in by_video[recipient_video]:
            choice_hash = hashlib.sha256(
                f"perception-av-event-cross-video-audio:{seed}:{recipient['id']}".encode()
            ).digest()
            donors[recipient["id"]] = donor_events[
                int.from_bytes(choice_hash[:8], "big") % len(donor_events)
            ]["id"]
    return donors


def previous_event_audio_donors(rows: list[dict]) -> dict[str, str]:
    """Use the immediately preceding event's audio, ordered by decoded sample start."""
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        sound_id = row.get("sound_id")
        if type(sound_id) is not int or sound_id < 0:
            raise ValueError("stale-audio controls require nonnegative integer sound IDs")
        by_video[row["video_id"]].append(row)
    donors = {}
    for events in by_video.values():
        ordered = sorted(
            events,
            key=lambda row: (
                row.get("audio_sample_start", row["sound_id"]),
                row["sound_id"],
                row["id"],
            ),
        )
        if len({row["sound_id"] for row in ordered}) != len(ordered):
            raise ValueError("duplicate sound IDs within a video")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            donors[current["id"]] = previous["id"]
    return donors


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def run_diagnostic(
    *,
    features_path: Path,
    checkpoint_path: Path,
    training_result_path: Path,
    natural_predictions_path: Path,
    output_dir: Path,
    seed: int = 17,
):
    if output_dir.exists():
        raise ValueError("fresh output directory required")
    # Imports are delayed so donor-schedule tests work in CPU-only dev CI.
    import torch

    from scripts.train_av_event_source import _score_rows, load_feature_rows
    from tiny_omni_decision.av_event import AVEventSourceScorer

    features_hash = sha256_file(features_path)
    training_result_hash = sha256_file(training_result_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    run_result = json.loads(training_result_path.read_text(encoding="utf-8"))
    if (
        run_result.get("status") != "PASS_AV_EVENT_READOUT_FIT"
        or run_result.get("condition") != "joint"
        or run_result.get("feature_manifest_sha256") != features_hash
    ):
        raise ValueError("training receipt does not identify this successful joint-feature run")
    expected_checkpoint_hash = run_result.get("artifacts_sha256", {}).get("best.pt")
    if expected_checkpoint_hash != checkpoint_hash:
        raise ValueError("best checkpoint hash differs from the training receipt")
    if run_result.get("checkpoint_reload_verified") is not True:
        raise ValueError("training receipt lacks selected-checkpoint reload verification")

    loaded = load_feature_rows(features_path)
    rows = [row for row in loaded if row["split"] == "development_validation"]
    rows.sort(key=lambda row: row["id"])
    row_by_id = {row["id"]: row for row in rows}
    saved_natural = [
        json.loads(line)
        for line in natural_predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(saved_natural) != len(rows) or {row["id"] for row in saved_natural} != set(row_by_id):
        raise ValueError("natural predictions do not match development event IDs")

    hidden_size = run_result["config"]["hidden_size"]
    model = AVEventSourceScorer(hidden_size=hidden_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("condition") != "joint"
        or checkpoint.get("epoch") != run_result["best_epoch"]
    ):
        raise ValueError("checkpoint metadata differs from selected joint run")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)

    _, natural = _score_rows(model, rows, "joint")
    saved_by_id = {row["id"]: row for row in saved_natural}
    if natural != [saved_by_id[row["id"]] for row in rows]:
        raise ValueError("reloaded best checkpoint does not reproduce saved natural predictions")

    zero_rows = [{**row, "audio": np.zeros_like(row["audio"])} for row in rows]
    _, zero_predictions = _score_rows(model, zero_rows, "joint")

    cross_donors = cross_video_audio_donors(rows, seed=seed)
    shuffled_rows = [
        {**row, "audio": row_by_id[cross_donors[row["id"]]]["audio"]} for row in rows
    ]
    _, shuffled_predictions = _score_rows(model, shuffled_rows, "joint")

    stale_donors = previous_event_audio_donors(rows)
    stale_ids = sorted(stale_donors)
    stale_rows = [
        {**row_by_id[event_id], "audio": row_by_id[stale_donors[event_id]]["audio"]}
        for event_id in stale_ids
    ]
    _, stale_predictions = _score_rows(model, stale_rows, "joint")

    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_sets = {
        "natural": natural,
        "zero_audio": zero_predictions,
        "cross_video_shuffled_audio": shuffled_predictions,
        "previous_event_audio": stale_predictions,
    }
    for name, predictions in prediction_sets.items():
        _write_jsonl(output_dir / f"{name}-predictions.jsonl", predictions)
    donor_rows = [
        {"event_id": event_id, "donor_event_id": donor_id, "control": "cross_video_shuffle"}
        for event_id, donor_id in sorted(cross_donors.items())
    ] + [
        {"event_id": event_id, "donor_event_id": donor_id, "control": "previous_event_stale"}
        for event_id, donor_id in sorted(stale_donors.items())
    ]
    _write_jsonl(output_dir / "audio-donor-map.jsonl", donor_rows)

    metrics = {name: score_rows(predictions) for name, predictions in prediction_sets.items()}
    metrics["natural_minus_corruption_video_bootstrap"] = {
        name: paired_video_bootstrap(natural, prediction_sets[name], draws=10_000, seed=seed)
        for name in ("zero_audio", "cross_video_shuffled_audio")
    }
    natural_stale_subset = [saved_by_id[event_id] for event_id in stale_ids]
    metrics["stale_subset_comparison"] = {
        "natural": score_rows(natural_stale_subset),
        "previous_event_audio": score_rows(stale_predictions),
        "event_ids": len(stale_ids),
        "videos": len({row_by_id[event_id]["video_id"] for event_id in stale_ids}),
        "natural_minus_previous_event_audio_video_bootstrap": paired_video_bootstrap(
            natural_stale_subset, stale_predictions, draws=10_000, seed=seed
        ),
    }
    receipt = {
        "status": "PASS_FIXED_CHECKPOINT_AV_INPUT_SENSITIVITY",
        "condition": "joint",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "training_result_sha256": training_result_hash,
        "features_sha256": features_hash,
        "natural_predictions_sha256": sha256_file(natural_predictions_path),
        "seed": seed,
        "development_events": len(rows),
        "development_videos": len({row["video_id"] for row in rows}),
        "previous_event_audio_events": len(stale_rows),
        "previous_event_audio_videos": len(
            {row["video_id"] for row in stale_rows}
        ),
        "metrics": metrics,
        "interpretation": (
            "Input sensitivity under deliberately corrupted alignment only; "
            "not evidence of correct "
            "semantic use, an independent quality estimate, or a tuning signal."
        ),
        "checkpoint_selection": False,
        "optimizer_updates": 0,
        "encoder_updates": 0,
        "gpu_used": False,
        "test_opened": False,
        "development_validation_previously_observed": True,
    }
    _write_json(output_dir / "result.json", receipt)
    receipt["artifacts_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "result.json"
    }
    _write_json(output_dir / "result.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--natural-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = run_diagnostic(
        features_path=args.features,
        checkpoint_path=args.checkpoint,
        training_result_path=args.training_result,
        natural_predictions_path=args.natural_predictions,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
