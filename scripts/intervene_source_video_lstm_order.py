"""Reverse actual visual frame sequences in fixed train-only CV LSTM heads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.replay_source_unimodal_cv import compare_prediction
from scripts.summarize_source_unimodal_cv import paired_interval
from scripts.train_source_unimodal_cv import (
    AUDIO,
    LABELS,
    PINNED,
    PLAN,
    SOURCE,
    SOURCE_RUNNER,
    VIDEO,
    digest,
    fold_assignment,
)
from scripts.train_source_video_lstm_cv import (
    TemporalCandidateModel,
    load_sequences,
    sequence_batch,
)

RUN_SHA = "298ff62ffbd7a667680c583916c9c3847f1a4d3888b732bb7c7166f79c68accf"


def reverse_valid(sequences, lengths):
    if sequences.ndim != 4 or lengths.shape != sequences.shape[:2]:
        raise ValueError("sequence/length shape mismatch")
    result = sequences.clone()
    for i in range(len(lengths)):
        for j in range(lengths.shape[1]):
            length = int(lengths[i, j])
            if length < 0 or length > sequences.shape[2]:
                raise ValueError("invalid sequence length")
            if length > 1:
                result[i, j, :length] = sequences[i, j, :length].flip(0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("output exists")
    if digest(args.run_dir / "run.json") != RUN_SHA:
        raise ValueError("temporal run hash mismatch")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    run = json.loads((args.run_dir / "run.json").read_bytes())
    if run["status"] != "PASS_FIXED_TEMPORAL_TRAIN_ONLY_CV":
        raise ValueError("incomplete source run")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("pinned_source_readout", SOURCE_RUNNER)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    torch.set_num_threads(4)
    examples, _ = source.load_examples(PLAN, LABELS, VIDEO, AUDIO)
    assignments = fold_assignment([example.video_id for example in examples])
    assignment_sha = hashlib.sha256(json.dumps(assignments, sort_keys=True).encode()).hexdigest()
    if len(examples) != 2593 or assignment_sha != run["assignment_sha256"]:
        raise ValueError("source examples or fold assignment mismatch")
    sequences = load_sequences(examples)
    args.output_dir.mkdir(parents=True)
    rows = []
    for fold in range(3):
        entry = next(entry for entry in run["results"] if entry["fold"] == fold)
        checkpoint = args.run_dir / f"fold-{fold}-temporal.pt"
        predictions = args.run_dir / f"fold-{fold}-predictions.jsonl"
        if (
            digest(checkpoint) != entry["checkpoint_sha256"]
            or digest(predictions) != entry["prediction_sha256"]
        ):
            raise ValueError("checkpoint or prediction hash mismatch")
        saved = [json.loads(line) for line in predictions.read_bytes().splitlines()]
        valid = [example for example in examples if assignments[example.video_id] == fold]
        if len(valid) != len(saved):
            raise ValueError("prediction count mismatch")
        model = TemporalCandidateModel()
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(valid), 32):
                indices = list(range(start, min(start + 32, len(valid))))
                video, audio, targets, mask, batch = source.collate_examples(valid, indices)
                temporal, lengths = sequence_batch(batch, sequences)
                outputs = {}
                for condition, candidate_frames in [
                    ("original", temporal),
                    ("reverse", reverse_valid(temporal, lengths)),
                ]:
                    logits = model(video, audio, mask, candidate_frames, lengths)
                    probability = torch.softmax(logits, dim=1)
                    outputs[condition] = (
                        probability,
                        source.grouped_nll(logits, targets),
                        probability.argmax(dim=1),
                    )
                for index, example in enumerate(batch):
                    original, original_nll, original_chosen = outputs["original"]
                    dist = targets[index].float() / targets[index].sum()
                    probs = original[index, : len(example.candidate_ids)].tolist() + [
                        float(original[index, -1])
                    ]
                    compare_prediction(
                        saved[start + index],
                        example,
                        probs,
                        bool(targets[index, original_chosen[index]]),
                        float(original_nll[index]),
                        float((original[index] - dist).square().sum()),
                    )
                    positive_multi = any(
                        len(sequences[(example.record_id, candidate)]) > 1
                        for candidate in example.target_candidate_ids
                    )
                    row = {
                        "record_id": example.record_id,
                        "video_id": example.video_id,
                        "none_visible": example.none_visible,
                        "any_multi_frame": bool((lengths[index] > 1).any()),
                        "positive_multi_frame": positive_multi,
                    }
                    for condition, (_prob, nll, chosen) in outputs.items():
                        row[condition] = {
                            "correct": bool(targets[index, chosen[index]]),
                            "nll": float(nll[index]),
                            "choice": int(chosen[index]),
                        }
                    rows.append(row)
    if len(rows) != 2593 or len({row["record_id"] for row in rows}) != 2593:
        raise ValueError("incomplete held-out records")
    path = args.output_dir / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row) + "\n")
    comparisons = {}
    for field in ("correct", "nll"):
        comparisons[field] = paired_interval(
            [
                {
                    "video_id": row["video_id"],
                    "delta": float(row["original"][field]) - float(row["reverse"][field]),
                }
                for row in rows
            ]
        )
    strata = {}
    for name, selected in [
        ("all", rows),
        ("any_multi_frame", [r for r in rows if r["any_multi_frame"]]),
        ("visible_positive_multi_frame", [r for r in rows if r["positive_multi_frame"]]),
    ]:
        strata[name] = {
            "records": len(selected),
            "original_accuracy": float(np.mean([r["original"]["correct"] for r in selected])),
            "reverse_accuracy": float(np.mean([r["reverse"]["correct"] for r in selected])),
            "changed_choices": sum(
                r["original"]["choice"] != r["reverse"]["choice"] for r in selected
            ),
        }
    result = {
        "status": "PASS_TRAIN_ONLY_OOF_ORDER_INTERVENTION",
        "source_run_sha256": RUN_SHA,
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256_lf": hashlib.sha256(
            Path("docs/pretrained_reuse/R2_SOURCE_VIDEO_LSTM_ORDER_PROTOCOL_20261027.md")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        ).hexdigest(),
        "predictions_sha256": digest(path),
        "records": len(rows),
        "strata": strata,
        "paired_original_minus_reverse": comparisons,
        "bootstrap_repetitions": 10000,
        "seed": 17,
        "training": False,
        "old_development_or_sealed_audit_opened": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "limitations": "only residual frame order reversed; base visual path stays forward",
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
