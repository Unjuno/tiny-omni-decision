"""Train fixed train-only group-CV multimodal and unimodal source baselines."""

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

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
SOURCE = Path("C:/Users/junny/.codex/worktrees/r2-perception-source-candidate/tiny-omni-decision")
PLAN = ROOT / "perception-source-candidate-feature-plan-20261025-02/feature-plan.json"
LABELS = ROOT / "perception-source-candidate-dev-split-20261025-03/train.jsonl"
VIDEO = ROOT / "perception-source-candidate-videoprism-train-attempt-01"
AUDIO = ROOT / "perception-source-candidate-audio-train-attempt-01"
SOURCE_RUNNER = SOURCE / "scripts/train_perception_source_candidate_readout.py"
SOURCE_HELPER = SOURCE / "src/tiny_omni_decision/perception_source_candidate_readout.py"
PINNED = {
    PLAN: "9cf2fae738270a14e632b745b8a0f78214440eb057de76f0ba825d4413824247",
    LABELS: "c5a8396a6b47a8635033e445c67233c76a6ecb17d27f4e6943a153fd3598600d",
    SOURCE_RUNNER: "f1bd23af35fec79b394505394c406ad77e6993c38377564c3aa1626a57f04827",
    SOURCE_HELPER: "2a58a0ffa4bc0d4a8460477cbad6dca863c712a702c404332a25b50fa8c5e403",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fold_assignment(video_ids):
    unique = sorted(
        set(video_ids), key=lambda v: (hashlib.sha256(f"source-cv-v1:17:{v}".encode()).digest(), v)
    )
    if len(unique) < 3:
        raise ValueError("at least three distinct videos required")
    return {video: index % 3 for index, video in enumerate(unique)}


class CandidateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video = nn.Linear(1536, 128)
        self.audio = nn.Linear(384, 128)
        self.candidate = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.none = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, video, audio, mask, mode):
        if mode == "audio":
            video = torch.zeros_like(video)
        elif mode == "video":
            audio = torch.zeros_like(audio)
        elif mode != "joint":
            raise ValueError("unknown mode")
        visual = torch.tanh(self.video(video))
        sound = torch.tanh(self.audio(audio))
        repeated = sound[:, None, :].expand(-1, video.shape[1], -1)
        fused = torch.cat([visual, repeated, visual * repeated, (visual - repeated).abs()], dim=-1)
        candidate = self.candidate(fused).squeeze(-1).masked_fill(~mask, -1e9)
        pooled = (visual * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        none = self.none(torch.cat([pooled, sound], dim=-1))
        return torch.cat([candidate, none], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    if shutil.disk_usage(ROOT).free < 2**30:
        raise OSError("less than 1 GiB free disk")
    protocol_path = Path("docs/pretrained_reuse/R2_SOURCE_UNIMODAL_CV_PROTOCOL_20261027.md")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if protocol_sha != "9a6713ebfbab7f3c570be12a7c36ac020a4d9cea8f7105dc129cd7d895e3d2ee":
        raise ValueError("protocol hash mismatch")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input differs: {path}")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location(
        "pinned_source_readout", SOURCE / "scripts/train_perception_source_candidate_readout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    collate_examples, grouped_nll, load_examples = (
        module.collate_examples,
        module.grouped_nll,
        module.load_examples,
    )

    args.output.mkdir(parents=True)
    start = time.perf_counter()
    metadata = {
        "status": "RUNNING",
        "seed": 17,
        "folds": 3,
        "epochs": 7,
        "model_modes": ["audio", "video", "joint"],
        "device": "cpu",
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256_lf": protocol_sha,
        "pinned_inputs": {str(k): v for k, v in PINNED.items()},
        "development_features_or_labels_read": False,
        "sealed_audit_opened": False,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        torch.set_num_threads(4)
        examples, _ = load_examples(PLAN, LABELS, VIDEO, AUDIO)
        if len(examples) != 2593:
            raise ValueError("unexpected train example count")
        assignment = fold_assignment([e.video_id for e in examples])
        if len(assignment) != 320:
            raise ValueError("unexpected video-group count")
        assignments_sha = hashlib.sha256(
            json.dumps(assignment, sort_keys=True).encode()
        ).hexdigest()
        results = []
        for fold in range(3):
            train = [e for e in examples if assignment[e.video_id] != fold]
            valid = [e for e in examples if assignment[e.video_id] == fold]
            for mode in ("audio", "video", "joint"):
                torch.manual_seed(17)
                model = CandidateModel()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                rng = np.random.default_rng(17)
                history = []
                for _epoch in range(7):
                    model.train()
                    order = rng.permutation(len(train))
                    total = 0.0
                    for start_index in range(0, len(order), 32):
                        indices = order[start_index : start_index + 32].tolist()
                        video, audio, targets, mask, _ = collate_examples(train, indices)
                        loss = grouped_nll(model(video, audio, mask, mode), targets).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        if not torch.isfinite(norm):
                            raise FloatingPointError("nonfinite gradient")
                        optimizer.step()
                        total += float(loss.detach()) * len(indices)
                    history.append(total / len(train))
                model.eval()
                correct, nlls, briers, confidences, predictions = [], [], [], [], []
                with torch.inference_mode():
                    for start_index in range(0, len(valid), 32):
                        indices = list(range(start_index, min(start_index + 32, len(valid))))
                        video, audio, targets, mask, batch = collate_examples(valid, indices)
                        logits = model(video, audio, mask, mode)
                        nll = grouped_nll(logits, targets)
                        prob = torch.softmax(logits, dim=1)
                        chosen = prob.argmax(dim=1)
                        for j, example in enumerate(batch):
                            hit = bool(targets[j, chosen[j]])
                            distribution = targets[j].float() / targets[j].sum()
                            brier = float((prob[j] - distribution).square().sum())
                            correct.append(hit)
                            nlls.append(float(nll[j]))
                            briers.append(brier)
                            confidences.append(float(prob[j, chosen[j]]))
                            predictions.append(
                                {
                                    "record_id": example.record_id,
                                    "video_id": example.video_id,
                                    "none_visible": example.none_visible,
                                    "correct": hit,
                                    "candidate_ids": list(example.candidate_ids),
                                    "target_candidate_ids": sorted(example.target_candidate_ids),
                                    "nll": float(nll[j]),
                                    "brier": brier,
                                    "confidence": float(prob[j, chosen[j]]),
                                    "probabilities": prob[j, : len(example.candidate_ids)].tolist()
                                    + [float(prob[j, -1])],
                                }
                            )
                stem = f"fold-{fold}-{mode}"
                torch.save(model.state_dict(), args.output / f"{stem}.pt")
                with (args.output / f"{stem}-predictions.jsonl").open("x", encoding="utf-8") as out:
                    for row in predictions:
                        out.write(json.dumps(row) + "\n")
                ece = 0.0
                confidence_array, correct_array = np.asarray(confidences), np.asarray(correct)
                for bin_index in range(10):
                    lower, upper = bin_index / 10, (bin_index + 1) / 10
                    selected = (confidence_array >= lower) & (
                        confidence_array <= upper if bin_index == 9 else confidence_array < upper
                    )
                    if selected.any():
                        ece += float(
                            selected.mean()
                            * abs(
                                confidence_array[selected].mean() - correct_array[selected].mean()
                            )
                        )
                result = {
                    "fold": fold,
                    "mode": mode,
                    "train_rows": len(train),
                    "validation_rows": len(valid),
                    "train_groups": len({e.video_id for e in train}),
                    "validation_groups": len({e.video_id for e in valid}),
                    "accuracy": float(np.mean(correct)),
                    "nll": float(np.mean(nlls)),
                    "brier": float(np.mean(briers)),
                    "ece_10": ece,
                    "train_loss_history": history,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "checkpoint_sha256": digest(args.output / f"{stem}.pt"),
                    "prediction_sha256": digest(args.output / f"{stem}-predictions.jsonl"),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
        metadata.update(
            {
                "status": "PASS_TRAIN_ONLY_GROUP_CV",
                "assignments_sha256": assignments_sha,
                "results": results,
                "elapsed_seconds": time.perf_counter() - start,
                "cuda_initialized": torch.cuda.is_initialized(),
            }
        )
    except Exception as error:
        metadata.update(
            {
                "status": "FAILED",
                "error": repr(error),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        raise
    finally:
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
