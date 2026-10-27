"""Reload train-only CV checkpoints and reproduce their saved predictions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

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

RUN_SHA = "cadda73e4f396a50b376a2e8f5e59c7321531560464e6ed3d6c7e082757cc8a5"


def compare_prediction(saved, example, probabilities, hit, nll, brier):
    if (
        saved["record_id"] != example.record_id
        or saved["video_id"] != example.video_id
        or saved["none_visible"] != example.none_visible
        or saved["candidate_ids"] != list(example.candidate_ids)
        or saved["target_candidate_ids"] != sorted(example.target_candidate_ids)
    ):
        raise ValueError("prediction identity mismatch")
    if saved["correct"] != hit:
        raise ValueError("prediction decision mismatch")
    expected = np.asarray(saved["probabilities"], dtype=np.float64)
    actual = np.asarray(probabilities, dtype=np.float64)
    if expected.shape != actual.shape or not np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
        raise ValueError("prediction probabilities mismatch")
    if not np.isclose(saved["nll"], nll, rtol=1e-6, atol=1e-6):
        raise ValueError("NLL mismatch")
    if not np.isclose(saved["brier"], brier, rtol=1e-6, atol=1e-6):
        raise ValueError("Brier mismatch")
    return float(np.max(np.abs(expected - actual)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    run_path = args.run_dir / "run.json"
    if digest(run_path) != RUN_SHA:
        raise ValueError("run receipt hash mismatch")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    run = json.loads(run_path.read_bytes())
    if run["status"] != "PASS_TRAIN_ONLY_GROUP_CV" or len(run["results"]) != 9:
        raise ValueError("incomplete source run")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("pinned_source_readout", SOURCE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    examples, _ = module.load_examples(PLAN, LABELS, VIDEO, AUDIO)
    assignment = fold_assignment([e.video_id for e in examples])
    assignment_sha = hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()
    if assignment_sha != run["assignments_sha256"] or len(examples) != 2593:
        raise ValueError("fold assignment mismatch")
    details = []
    for entry in run["results"]:
        fold, mode = entry["fold"], entry["mode"]
        stem = f"fold-{fold}-{mode}"
        checkpoint = args.run_dir / f"{stem}.pt"
        prediction_file = args.run_dir / f"{stem}-predictions.jsonl"
        if (
            digest(checkpoint) != entry["checkpoint_sha256"]
            or digest(prediction_file) != entry["prediction_sha256"]
        ):
            raise ValueError("checkpoint or prediction hash mismatch")
        saved = [json.loads(line) for line in prediction_file.read_bytes().splitlines()]
        valid = [example for example in examples if assignment[example.video_id] == fold]
        if len(valid) != len(saved):
            raise ValueError("prediction count mismatch")
        model = CandidateModel()
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.eval()
        maximum = 0.0
        with torch.inference_mode():
            for start in range(0, len(valid), 32):
                indices = list(range(start, min(start + 32, len(valid))))
                video, audio, targets, mask, batch = module.collate_examples(valid, indices)
                logits = model(video, audio, mask, mode)
                probabilities = torch.softmax(logits, dim=1)
                nll = module.grouped_nll(logits, targets)
                chosen = probabilities.argmax(dim=1)
                for index, example in enumerate(batch):
                    row = probabilities[index]
                    brier = float(
                        (row - targets[index].float() / targets[index].sum()).square().sum()
                    )
                    current = row[: len(example.candidate_ids)].tolist() + [float(row[-1])]
                    maximum = max(
                        maximum,
                        compare_prediction(
                            saved[start + index],
                            example,
                            current,
                            bool(targets[index, chosen[index]]),
                            float(nll[index]),
                            brier,
                        ),
                    )
        details.append(
            {"fold": fold, "mode": mode, "rows": len(valid), "max_probability_delta": maximum}
        )
    result = {
        "status": "PASS_ALL_NINE_PREDICTION_REPLAYS",
        "source_run_sha256": RUN_SHA,
        "script_sha256": digest(Path(__file__)),
        "details": details,
        "total_predictions": sum(d["rows"] for d in details),
        "training": False,
        "development_features_or_labels_read": False,
        "sealed_audit_opened": False,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
