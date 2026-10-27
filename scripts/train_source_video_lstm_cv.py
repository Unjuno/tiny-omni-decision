"""Train a visual-LSTM residual on existing train-only source-candidate folds."""

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
from torch import nn
from torch.nn import functional as F

from scripts.replay_source_unimodal_cv import RUN_SHA
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


class TemporalCandidateModel(CandidateModel):
    def __init__(self):
        super().__init__()
        self.temporal = nn.LSTM(768, 32, batch_first=True)
        self.temporal_residual = nn.Linear(32, 128, bias=False)
        nn.init.zeros_(self.temporal_residual.weight)

    def forward(self, video, audio, mask, sequences, lengths):
        batch, candidates, time_count, _ = sequences.shape
        if lengths.shape != mask.shape or torch.any(lengths[mask] < 1):
            raise ValueError("candidate sequence lengths must be positive")
        if torch.any(lengths[~mask] != 0) or torch.any(lengths > time_count):
            raise ValueError("invalid candidate padding")
        flat_mask = mask.reshape(-1)
        valid = sequences.reshape(-1, time_count, 768)[flat_mask]
        valid_lengths = lengths.reshape(-1)[flat_mask]
        packed = nn.utils.rnn.pack_padded_sequence(
            F.normalize(valid, dim=-1), valid_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.temporal(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        delta = output[torch.arange(len(valid)), valid_lengths - 1] - output[:, 0]
        hidden = torch.zeros((batch * candidates, 32), dtype=video.dtype, device=video.device)
        hidden[flat_mask] = delta
        visual = torch.tanh(
            self.video(video) + self.temporal_residual(hidden.reshape(batch, candidates, 32))
        )
        sound = torch.tanh(self.audio(audio))
        repeated = sound[:, None, :].expand(-1, candidates, -1)
        fused = torch.cat([visual, repeated, visual * repeated, (visual - repeated).abs()], dim=-1)
        logits = self.candidate(fused).squeeze(-1).masked_fill(~mask, -1e9)
        pooled = (visual * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        none = self.none(torch.cat([pooled, sound], dim=-1))
        return torch.cat([logits, none], dim=1)


def sequence_batch(batch, sequences):
    maximum = max(len(example.candidate_ids) for example in batch)
    values = np.zeros((len(batch), maximum, 4, 768), dtype=np.float32)
    lengths = np.zeros((len(batch), maximum), dtype=np.int64)
    for i, example in enumerate(batch):
        for j, candidate in enumerate(example.candidate_ids):
            vector = sequences[(example.record_id, candidate)]
            if vector.ndim != 2 or not 1 <= len(vector) <= 4 or vector.shape[1] != 768:
                raise ValueError("invalid cached sequence")
            values[i, j, : len(vector)] = vector
            lengths[i, j] = len(vector)
    return torch.from_numpy(values), torch.from_numpy(lengths)


def load_sequences(examples):
    plan = json.loads(PLAN.read_bytes())["train"]["records"]
    index = [json.loads(line) for line in (VIDEO / "feature_index.jsonl").read_bytes().splitlines()]
    vectors = np.load(VIDEO / "candidate_frame_roi_features.npy", mmap_mode="r")
    lookup = {
        (r["record_id"], r["candidate_id"], int(r["frame_id"])): int(r["feature_index"])
        for r in index
    }
    if len(lookup) != len(index):
        raise ValueError("duplicate frame feature identity")
    sequences = {}
    for row in plan:
        for candidate in row["candidates"]:
            frames = sorted(int(frame["frame_id"]) for frame in candidate["frames"])
            if len(frames) != len(set(frames)):
                raise ValueError("duplicate frame")
            sequences[(row["record_id"], candidate["candidate_id"])] = np.stack(
                [
                    vectors[lookup[(row["record_id"], candidate["candidate_id"], frame)]]
                    for frame in frames
                ]
            )
    expected = {(e.record_id, c) for e in examples for c in e.candidate_ids}
    if set(sequences) != expected or any(not np.isfinite(x).all() for x in sequences.values()):
        raise ValueError("cached sequences do not match frozen candidate set")
    return sequences


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
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("pinned_source_readout", SOURCE_RUNNER)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    args.output_dir.mkdir(parents=True)
    start_time = time.perf_counter()
    run = {
        "status": "RUNNING",
        "script_sha256": digest(Path(__file__)),
        "baseline_run_sha256": RUN_SHA,
        "protocol_sha256_lf": hashlib.sha256(
            Path("docs/pretrained_reuse/R2_SOURCE_VIDEO_LSTM_PROTOCOL_20261027.md")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        ).hexdigest(),
        "seed": 17,
        "epochs": 7,
        "folds": 3,
        "device": "cpu",
        "sealed_audit_opened": False,
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    try:
        torch.set_num_threads(4)
        examples, _ = source.load_examples(PLAN, LABELS, VIDEO, AUDIO)
        assignment = fold_assignment([e.video_id for e in examples])
        baseline = json.loads((args.baseline_run_dir / "run.json").read_bytes())
        assignment_sha = hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()
        if len(examples) != 2593 or assignment_sha != baseline["assignments_sha256"]:
            raise ValueError("frozen train fold identity differs")
        sequences = load_sequences(examples)
        results = []
        for fold in range(3):
            train = [e for e in examples if assignment[e.video_id] != fold]
            valid = [e for e in examples if assignment[e.video_id] == fold]
            torch.manual_seed(17)
            model = TemporalCandidateModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            rng = np.random.default_rng(17)
            history = []
            for epoch in range(7):
                model.train()
                order = rng.permutation(len(train))
                loss_total = 0.0
                for start in range(0, len(order), 32):
                    indices = order[start : start + 32].tolist()
                    video, audio, targets, mask, batch = source.collate_examples(train, indices)
                    temporal, lengths = sequence_batch(batch, sequences)
                    loss = source.grouped_nll(
                        model(video, audio, mask, temporal, lengths), targets
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
                history.append(loss_total / len(train))
                print(
                    json.dumps({"fold": fold, "epoch": epoch + 1, "train_nll": history[-1]}),
                    flush=True,
                )
            model.eval()
            predictions = []
            with torch.inference_mode():
                for start in range(0, len(valid), 32):
                    indices = list(range(start, min(start + 32, len(valid))))
                    video, audio, targets, mask, batch = source.collate_examples(valid, indices)
                    temporal, lengths = sequence_batch(batch, sequences)
                    logits = model(video, audio, mask, temporal, lengths)
                    probs = torch.softmax(logits, dim=1)
                    nll, chosen = source.grouped_nll(logits, targets), probs.argmax(dim=1)
                    for j, example in enumerate(batch):
                        target = targets[j].float() / targets[j].sum()
                        predictions.append(
                            {
                                "record_id": example.record_id,
                                "video_id": example.video_id,
                                "none_visible": example.none_visible,
                                "correct": bool(targets[j, chosen[j]]),
                                "candidate_ids": list(example.candidate_ids),
                                "target_candidate_ids": sorted(example.target_candidate_ids),
                                "nll": float(nll[j]),
                                "brier": float((probs[j] - target).square().sum()),
                                "confidence": float(probs[j, chosen[j]]),
                                "probabilities": probs[j, : len(example.candidate_ids)].tolist()
                                + [float(probs[j, -1])],
                            }
                        )
            checkpoint = args.output_dir / f"fold-{fold}-temporal.pt"
            output = args.output_dir / f"fold-{fold}-predictions.jsonl"
            torch.save(model.state_dict(), checkpoint)
            with output.open("x", encoding="utf-8") as stream:
                for row in predictions:
                    stream.write(json.dumps(row) + "\n")
            result = {
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "accuracy": float(np.mean([r["correct"] for r in predictions])),
                "nll": float(np.mean([r["nll"] for r in predictions])),
                "parameters": sum(p.numel() for p in model.parameters()),
                "train_loss_history": history,
                "checkpoint_sha256": digest(checkpoint),
                "prediction_sha256": digest(output),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        run.update(
            {
                "status": "PASS_FIXED_TEMPORAL_TRAIN_ONLY_CV",
                "results": results,
                "assignment_sha256": assignment_sha,
                "elapsed_seconds": time.perf_counter() - start_time,
                "cuda_initialized": torch.cuda.is_initialized(),
            }
        )
    except Exception as error:
        run.update(
            {
                "status": "FAILED",
                "error": repr(error),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        raise
    finally:
        (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
