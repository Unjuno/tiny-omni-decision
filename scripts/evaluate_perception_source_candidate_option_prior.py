"""Score the fixed validation cohort with training-label option-slot priors only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_option_slot_prior(training_rows, *, smoothing=1.0):
    if not training_rows or smoothing <= 0:
        raise ValueError("option prior requires training labels and positive smoothing")
    max_candidates = 0
    candidate_counts = np.full(1, smoothing, dtype=np.float64)
    none_count = smoothing
    train_videos = set()
    for row in training_rows:
        candidates = row["request"]["candidates"]
        candidate_ids = [candidate["candidate_id"] for candidate in candidates]
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"invalid candidate IDs in training row {row['record_id']}")
        if not row.get("supervision"):
            raise ValueError("option prior requires training labels")
        targets = set(row["supervision"]["target_candidate_ids"])
        none_visible = bool(row["supervision"]["none_visible"])
        if none_visible and targets:
            raise ValueError(f"none-visible training row has target candidates: {row['record_id']}")
        if not none_visible and (not targets or not targets.issubset(candidate_ids)):
            raise ValueError(f"invalid target candidates in training row {row['record_id']}")
        if len(candidate_counts) < len(candidate_ids):
            candidate_counts = np.pad(
                candidate_counts,
                (0, len(candidate_ids) - len(candidate_counts)),
                constant_values=smoothing,
            )
        max_candidates = max(max_candidates, len(candidate_ids))
        if none_visible:
            none_count += 1.0
        else:
            share = 1.0 / len(targets)
            for target in targets:
                candidate_counts[candidate_ids.index(target)] += share
        train_videos.add(row.get("group_id"))
    return {
        "candidate_position_counts": candidate_counts,
        "none_visible_count": none_count,
        "max_training_candidate_count": max_candidates,
        "training_video_ids": train_videos,
        "smoothing": smoothing,
    }


def score_option_slot_prior(training_rows, validation_rows, *, smoothing=1.0):
    prior = build_option_slot_prior(training_rows, smoothing=smoothing)
    validation_videos = [row.get("group_id") for row in validation_rows]
    if len(validation_videos) != len(set(validation_videos)):
        raise ValueError("validation option-prior evaluation requires unique video groups")
    if prior["training_video_ids"] & set(validation_videos):
        raise ValueError("training and validation video groups overlap")
    predictions = []
    max_slots = len(prior["candidate_position_counts"])
    for row in validation_rows:
        if row.get("status") != "candidate" or row.get("split") != "validation":
            raise ValueError(f"unexpected validation record: {row.get('record_id')}")
        candidates = row["request"]["candidates"]
        candidate_ids = [candidate["candidate_id"] for candidate in candidates]
        targets = set(row["supervision"]["target_candidate_ids"])
        none_visible = bool(row["supervision"]["none_visible"])
        if none_visible and targets:
            raise ValueError(
                f"none-visible validation row has target candidates: {row['record_id']}"
            )
        if not none_visible and (not targets or not targets.issubset(candidate_ids)):
            raise ValueError(f"invalid target candidates in validation row {row['record_id']}")
        position_scores = np.full(len(candidate_ids), smoothing, dtype=np.float64)
        known = min(len(candidate_ids), max_slots)
        position_scores[:known] = prior["candidate_position_counts"][:known]
        scores = np.concatenate((position_scores, [prior["none_visible_count"]]))
        probabilities = scores / scores.sum()
        target_positions = [
            index for index, candidate_id in enumerate(candidate_ids) if candidate_id in targets
        ]
        positives = target_positions + ([len(candidate_ids)] if none_visible else [])
        if not positives:
            raise ValueError(f"validation row has no positive target: {row['record_id']}")
        top = int(np.argmax(probabilities))
        predictions.append(
            {
                "record_id": row["record_id"],
                "video_id": row["group_id"],
                "candidate_ids": candidate_ids,
                "target_candidate_ids": sorted(targets),
                "none_visible": none_visible,
                "predicted": "none_visible" if top == len(candidate_ids) else candidate_ids[top],
                "top1_correct": top in positives,
                "top1_confidence": float(probabilities[top]),
                "target_probability_mass": float(probabilities[positives].sum()),
                "nll": float(-np.log(probabilities[positives].sum())),
                "candidate_probabilities": {
                    candidate_id: float(probabilities[index])
                    for index, candidate_id in enumerate(candidate_ids)
                },
                "none_visible_probability": float(probabilities[-1]),
            }
        )
    return predictions, prior


def run(args):
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite option-prior output: {args.output}")
    training_rows = read_jsonl(args.train_labels)
    corpus_rows = read_jsonl(args.validation_corpus)
    validation_rows = [row for row in corpus_rows if row.get("status") == "candidate"]
    if (
        sha256_file(args.train_labels)
        != "c5a8396a6b47a8635033e445c67233c76a6ecb17d27f4e6943a153fd3598600d"
        or sha256_file(args.validation_corpus)
        != "4a15269531ef9a2a1bccdab731fa23e30cd3754850fc72fd35028b9986bd0844"
        or len(training_rows) != 2593
        or len(validation_rows) != 138
    ):
        raise ValueError("option-prior inputs differ from the fixed training/validation artifacts")
    predictions, prior = score_option_slot_prior(training_rows, validation_rows)
    from scripts.train_perception_source_candidate_readout import metrics_from_predictions

    metrics = metrics_from_predictions(predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions_path = args.output / "predictions.jsonl"
    with predictions_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "status": "PASS_TRAIN_FREQUENCY_OPTION_ONLY_BASELINE",
        "scope": "training-label position/count prior; no media features or fitted model",
        "train_labels_sha256": sha256_file(args.train_labels),
        "validation_corpus_sha256": sha256_file(args.validation_corpus),
        "training_records": len(training_rows),
        "training_video_groups": len(prior["training_video_ids"]),
        "validation_records": len(validation_rows),
        "validation_video_groups": len({row["group_id"] for row in validation_rows}),
        "max_training_candidate_count": prior["max_training_candidate_count"],
        "validation_candidate_count_distribution": dict(
            sorted(Counter(len(row["request"]["candidates"]) for row in validation_rows).items())
        ),
        "smoothing": prior["smoothing"],
        "training_candidate_position_counts": prior["candidate_position_counts"].tolist(),
        "training_none_visible_count_with_smoothing": prior["none_visible_count"],
        "metrics": metrics,
        "predictions_sha256": sha256_file(predictions_path),
        "media_features_loaded": False,
        "readout_checkpoint_loaded": False,
        "training_performed": False,
        "validation_labels_used_to_fit_prior": False,
        "checkpoint_selection_performed": False,
        "sealed_audit_opened": False,
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--validation-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
