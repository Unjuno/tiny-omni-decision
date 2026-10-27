"""Apply fixed audio interventions to the saved train-only CV joint models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.replay_source_unimodal_cv import RUN_SHA, compare_prediction
from scripts.summarize_source_unimodal_cv import paired_interval
from scripts.train_source_unimodal_cv import (
    AUDIO,
    LABELS,
    PINNED,
    PLAN,
    SOURCE,
    SOURCE_RUNNER,
    VIDEO,
    CandidateModel,
    digest,
    fold_assignment,
)


def donor_mapping(examples):
    groups = {}
    for example in examples:
        groups.setdefault(example.video_id, []).append(example)
    videos = sorted(groups)
    if len(videos) < 2:
        raise ValueError("at least two held-out video groups required")
    first = {video: min(groups[video], key=lambda e: e.record_id) for video in videos}
    return {
        example.record_id: first[videos[(index + 1) % len(videos)]].record_id
        for index, video in enumerate(videos)
        for example in groups[video]
    }


def intervention_metrics(records, condition):
    values = [row[condition] for row in records]
    result = {
        "accuracy": float(np.mean([x["correct"] for x in values])),
        "nll": float(np.mean([x["nll"] for x in values])),
    }
    for name, flag in [("visible", False), ("none_visible", True)]:
        subset = [row[condition]["correct"] for row in records if row["none_visible"] == flag]
        result[name] = {"records": len(subset), "accuracy": float(np.mean(subset))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("output exists")
    if digest(args.run_dir / "run.json") != RUN_SHA:
        raise ValueError("run receipt hash mismatch")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    run = json.loads((args.run_dir / "run.json").read_bytes())
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("pinned_source_readout", SOURCE_RUNNER)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    torch.set_num_threads(4)
    examples, _ = source.load_examples(PLAN, LABELS, VIDEO, AUDIO)
    assignments = fold_assignment([e.video_id for e in examples])
    sha = hashlib.sha256(json.dumps(assignments, sort_keys=True).encode()).hexdigest()
    if len(examples) != 2593 or sha != run["assignments_sha256"]:
        raise ValueError("source examples or fold assignment mismatch")
    args.output_dir.mkdir(parents=True)
    records, mappings = [], {}
    for fold in range(3):
        entry = next(e for e in run["results"] if e["fold"] == fold and e["mode"] == "joint")
        checkpoint = args.run_dir / f"fold-{fold}-joint.pt"
        predictions = args.run_dir / f"fold-{fold}-joint-predictions.jsonl"
        if (
            digest(checkpoint) != entry["checkpoint_sha256"]
            or digest(predictions) != entry["prediction_sha256"]
        ):
            raise ValueError("checkpoint or prediction hash mismatch")
        saved = [json.loads(line) for line in predictions.read_bytes().splitlines()]
        valid = [e for e in examples if assignments[e.video_id] == fold]
        if len(valid) != len(saved):
            raise ValueError("prediction count mismatch")
        by_id = {e.record_id: e for e in valid}
        donors = donor_mapping(valid)
        mappings.update(donors)
        model = CandidateModel()
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(valid), 32):
                indices = list(range(start, min(start + 32, len(valid))))
                video, audio, targets, mask, batch = source.collate_examples(valid, indices)
                mismatch = torch.from_numpy(
                    np.stack([by_id[donors[e.record_id]].audio for e in batch])
                )
                outputs = {}
                for condition, candidate_audio in [
                    ("original", audio),
                    ("zero_audio", torch.zeros_like(audio)),
                    ("mismatched_audio", mismatch),
                ]:
                    logits = model(video, candidate_audio, mask, "joint")
                    probabilities = torch.softmax(logits, dim=1)
                    nll = source.grouped_nll(logits, targets)
                    chosen = probabilities.argmax(dim=1)
                    outputs[condition] = (probabilities, nll, chosen)
                for index, example in enumerate(batch):
                    row = {
                        "record_id": example.record_id,
                        "video_id": example.video_id,
                        "none_visible": example.none_visible,
                    }
                    for condition, (_probabilities, nll, chosen) in outputs.items():
                        row[condition] = {
                            "correct": bool(targets[index, chosen[index]]),
                            "nll": float(nll[index]),
                        }
                    base, base_nll, base_chosen = outputs["original"]
                    distribution = targets[index].float() / targets[index].sum()
                    brier = float((base[index] - distribution).square().sum())
                    probs = base[index, : len(example.candidate_ids)].tolist() + [
                        float(base[index, -1])
                    ]
                    compare_prediction(
                        saved[start + index],
                        example,
                        probs,
                        bool(targets[index, base_chosen[index]]),
                        float(base_nll[index]),
                        brier,
                    )
                    records.append(row)
    if len(records) != 2593 or len(mappings) != 2593:
        raise ValueError("incomplete records or duplicate IDs")
    prediction_path = args.output_dir / "interventions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as out:
        for row in records:
            out.write(json.dumps(row) + "\n")
    comparisons = {}
    for condition in ("zero_audio", "mismatched_audio"):
        comparisons[condition] = {}
        for metric in ("correct", "nll"):
            pairs = [
                {
                    "video_id": row["video_id"],
                    "delta": float(row["original"][metric]) - float(row[condition][metric]),
                }
                for row in records
            ]
            comparisons[condition][metric] = paired_interval(pairs)
    result = {
        "status": "PASS_TRAIN_ONLY_OOF_AUDIO_INTERVENTIONS",
        "source_run_sha256": RUN_SHA,
        "protocol_sha256_lf": hashlib.sha256(
            Path("docs/pretrained_reuse/R2_SOURCE_UNIMODAL_CV_INTERVENTION_PROTOCOL_20261027.md")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        ).hexdigest(),
        "script_sha256": digest(Path(__file__)),
        "donor_mapping_sha256": hashlib.sha256(
            json.dumps(mappings, sort_keys=True).encode()
        ).hexdigest(),
        "predictions_sha256": digest(prediction_path),
        "records": len(records),
        "metrics": {
            mode: intervention_metrics(records, mode)
            for mode in ("original", "zero_audio", "mismatched_audio")
        },
        "paired_original_minus_control": comparisons,
        "seed": 17,
        "bootstrap_repetitions": 10000,
        "training": False,
        "old_development_or_sealed_audit_opened": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "limitations": (
            "internal nonblind OOF diagnostic; interventions do not establish causal necessity"
        ),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
