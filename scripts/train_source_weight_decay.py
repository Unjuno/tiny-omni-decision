"""Run a fixed-budget stronger weight-decay comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.train_source_unimodal_cv import (
    AUDIO,
    LABELS,
    PINNED,
    PLAN,
    ROOT,
    SOURCE,
    SOURCE_RUNNER,
    VIDEO,
    CandidateModel,
    digest,
    fold_assignment,
)
from scripts.train_source_video_lstm_cv import (
    TemporalCandidateModel,
    sequence_batch,
)

RUN_SHA = "619f9ea6c704477c8c2ddcb6ea3030e52aac470e5205dd550cb0ae0dba14d906"

EVALUATIONS = (1, 2, 4, 7, 8, 16, 24, 32)
CHECKPOINTS = (7, 16, 32)


def score(model, mode, video, audio, mask, batch, sequences):
    if mode == "baseline":
        return model(video, audio, mask, "joint")
    if mode == "temporal":
        temporal, lengths = sequence_batch(batch, sequences)
        return model(video, audio, mask, temporal, lengths)
    raise ValueError("unknown mode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("output exists")
    if shutil.disk_usage(ROOT).free < 2**30:
        raise OSError("less than 1 GiB free disk")
    if digest(args.baseline_run_dir / "run.json") != RUN_SHA:
        raise ValueError("baseline receipt differs")
    protocol_path = Path("docs/pretrained_reuse/R2_SOURCE_WEIGHT_DECAY_PROTOCOL_20261027.md")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if protocol_sha != "291c4c8558efc25f0d9c43a3d4a808f5166f129acdf0682707484d151376beef":
        raise ValueError("protocol hash mismatch")
    for path, expected in (
        (
            Path("scripts/train_source_unimodal_cv.py"),
            "3f018f47f48cb0062521edf3d970041799e83a2de2f5836f1273eee07c583592",
        ),
        (
            Path("scripts/train_source_video_lstm_cv.py"),
            "e4dc331ebd9b1740d755f45d620f5fd861bfc927f49e3e2dbab52c2f23000e11",
        ),
    ):
        if digest(path) != expected:
            raise ValueError(f"readout source hash mismatch: {path}")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("pinned_source_readout", SOURCE_RUNNER)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    result = {
        "status": "RUNNING",
        "baseline_run_sha256": RUN_SHA,
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256_lf": protocol_sha,
        "seed": 17,
        "weight_decay": 0.01,
        "epochs": 32,
        "evaluations": EVALUATIONS,
        "checkpoints": CHECKPOINTS,
        "device": "cpu",
        "old_development_or_sealed_audit_opened": False,
        "completed": [],
    }
    (args.output_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    try:
        torch.set_num_threads(4)
        examples, _ = source.load_examples(PLAN, LABELS, VIDEO, AUDIO)
        assignments = fold_assignment([e.video_id for e in examples])
        baseline = json.loads((args.baseline_run_dir / "run.json").read_bytes())
        assignment_sha = hashlib.sha256(
            json.dumps(assignments, sort_keys=True).encode()
        ).hexdigest()
        if len(examples) != 2593 or assignment_sha != baseline["assignment_sha256"]:
            raise ValueError("source examples or fold assignments mismatch")
        sequences = None
        history_path = args.output_dir / "history.jsonl"
        for fold in range(3):
            train = [e for e in examples if assignments[e.video_id] != fold]
            valid = [e for e in examples if assignments[e.video_id] == fold]
            for mode in ("baseline",):
                torch.manual_seed(17)
                model = CandidateModel() if mode == "baseline" else TemporalCandidateModel()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                rng = np.random.default_rng(17)
                for epoch in range(1, 33):
                    if shutil.disk_usage(ROOT).free < 2**30:
                        raise OSError("less than 1 GiB free disk")
                    model.train()
                    order = rng.permutation(len(train))
                    loss_total = 0.0
                    for start in range(0, len(order), 32):
                        indices = order[start : start + 32].tolist()
                        video, audio, targets, mask, batch = source.collate_examples(train, indices)
                        loss = source.grouped_nll(
                            score(model, mode, video, audio, mask, batch, sequences), targets
                        ).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        if not torch.isfinite(norm):
                            raise FloatingPointError("nonfinite gradient")
                        optimizer.step()
                        loss_total += float(loss.detach()) * len(indices)
                    event = {
                        "fold": fold,
                        "mode": mode,
                        "epoch": epoch,
                        "train_online_nll": loss_total / len(train),
                        "train_rows": len(train),
                        "validation_rows": len(valid),
                    }
                    if epoch in EVALUATIONS:
                        predictions = []
                        model.eval()
                        with torch.inference_mode():
                            for start in range(0, len(valid), 32):
                                indices = list(range(start, min(start + 32, len(valid))))
                                video, audio, targets, mask, batch = source.collate_examples(
                                    valid, indices
                                )
                                logits = score(model, mode, video, audio, mask, batch, sequences)
                                probs = torch.softmax(logits, dim=1)
                                nll, chosen = (
                                    source.grouped_nll(logits, targets),
                                    probs.argmax(dim=1),
                                )
                                for j, example in enumerate(batch):
                                    target = targets[j].float() / targets[j].sum()
                                    predictions.append(
                                        {
                                            "record_id": example.record_id,
                                            "video_id": example.video_id,
                                            "none_visible": example.none_visible,
                                            "correct": bool(targets[j, chosen[j]]),
                                            "candidate_ids": list(example.candidate_ids),
                                            "target_candidate_ids": sorted(
                                                example.target_candidate_ids
                                            ),
                                            "nll": float(nll[j]),
                                            "brier": float((probs[j] - target).square().sum()),
                                            "confidence": float(probs[j, chosen[j]]),
                                            "probabilities": probs[
                                                j, : len(example.candidate_ids)
                                            ].tolist()
                                            + [float(probs[j, -1])],
                                        }
                                    )
                        path = (
                            args.output_dir / f"fold-{fold}-{mode}-epoch-{epoch}-predictions.jsonl"
                        )
                        with path.open("x", encoding="utf-8") as stream:
                            for row in predictions:
                                stream.write(json.dumps(row) + "\n")
                        event.update(
                            {
                                "validation_accuracy": float(
                                    np.mean([p["correct"] for p in predictions])
                                ),
                                "validation_nll": float(np.mean([p["nll"] for p in predictions])),
                                "prediction_sha256": digest(path),
                            }
                        )
                    if epoch in CHECKPOINTS:
                        path = args.output_dir / f"fold-{fold}-{mode}-epoch-{epoch}.pt"
                        torch.save(model.state_dict(), path)
                        event["checkpoint_sha256"] = digest(path)
                    with history_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event) + "\n")
                    if epoch in EVALUATIONS:
                        print(json.dumps(event), flush=True)
                result["completed"].append(
                    {
                        "fold": fold,
                        "mode": mode,
                        "parameters": sum(p.numel() for p in model.parameters()),
                    }
                )
                (args.output_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        result.update(
            {
                "status": "PASS_FIXED_32_EPOCH_CURVES",
                "assignment_sha256": assignment_sha,
                "history_sha256": digest(history_path),
                "elapsed_seconds": time.perf_counter() - started,
                "cuda_initialized": torch.cuda.is_initialized(),
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "FAILED",
                "error": repr(error),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        raise
    finally:
        (args.output_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
