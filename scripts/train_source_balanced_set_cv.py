"""Fixed three-fold source set-decision comparison on frozen train features."""

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

from scripts.source_balanced_set import balanced_set_loss
from scripts.source_set_decision import decode_set
from scripts.source_set_metrics import score_record
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

PROTOCOL = Path("docs/pretrained_reuse/R2_SOURCE_BALANCED_SET_PROTOCOL_20261027.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("output must be under the designated artifact root")
    if shutil.disk_usage(ROOT).free < 2**30 + 64 * 2**20:
        raise OSError("need 64 MiB above the 1 GiB disk reserve before starting")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input mismatch: {path}")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("pinned_source", SOURCE_RUNNER)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if protocol_hash != "62170ce4321103795d0dcf336b545f41b8061e2de85fcb87332c620cce784b81":
        raise ValueError("protocol hash mismatch")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    receipt = {
        "status": "RUNNING",
        "seed": 17,
        "epochs": 7,
        "folds": 3,
        "modes": ["audio", "video", "joint"],
        "script_sha256": digest(Path(__file__)),
        "loss_script_sha256": digest(Path("scripts/source_balanced_set.py")),
        "decoder_script_sha256": digest(Path("scripts/source_set_decision.py")),
        "metric_script_sha256": digest(Path("scripts/source_set_metrics.py")),
        "protocol_sha256_lf": protocol_hash,
        "pinned_inputs": {str(k): v for k, v in PINNED.items()},
        "device": "cpu",
        "sealed_audit_opened": False,
        "results": [],
    }
    (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        torch.set_num_threads(4)
        examples, _ = source.load_examples(PLAN, LABELS, VIDEO, AUDIO)
        groups = fold_assignment([e.video_id for e in examples])
        if len(examples) != 2593 or len(groups) != 320:
            raise ValueError("unexpected dataset or video groups")
        receipt["assignment_sha256"] = hashlib.sha256(
            json.dumps(groups, sort_keys=True).encode()
        ).hexdigest()
        for fold in range(3):
            train = [e for e in examples if groups[e.video_id] != fold]
            valid = [e for e in examples if groups[e.video_id] == fold]
            if {e.video_id for e in train} & {e.video_id for e in valid}:
                raise ValueError("video group overlap")
            for mode in ("audio", "video", "joint"):
                torch.manual_seed(17)
                model = CandidateModel()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                rng = np.random.default_rng(17)
                history = []
                for _epoch in range(7):
                    if shutil.disk_usage(ROOT).free < 2**30:
                        raise OSError("less than 1 GiB free")
                    model.train()
                    total = 0.0
                    order = rng.permutation(len(train))
                    for start_index in range(0, len(order), 32):
                        indices = order[start_index : start_index + 32].tolist()
                        video, audio, targets, mask, _ = source.collate_examples(train, indices)
                        loss = balanced_set_loss(model(video, audio, mask, mode), mask, targets)
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
                predictions = []
                with torch.inference_mode():
                    for start_index in range(0, len(valid), 32):
                        indices = list(range(start_index, min(start_index + 32, len(valid))))
                        video, audio, targets, mask, batch = source.collate_examples(valid, indices)
                        logits = model(video, audio, mask, mode)
                        selected = decode_set(logits, mask)
                        conditional = torch.sigmoid(logits[:, :-1])
                        none = torch.sigmoid(logits[:, -1])
                        for j, example in enumerate(batch):
                            ids = list(example.candidate_ids)
                            n = len(ids)
                            choices = [ids[k] for k in range(n) if selected[j, k]]
                            if selected[j, -1]:
                                choices.append("none_visible")
                            target_ids = sorted(example.target_candidate_ids)
                            conditional_probs = conditional[j, :n].tolist()
                            none_prob = float(none[j])
                            scores = score_record(
                                ids, target_ids, choices, conditional_probs, none_prob
                            )
                            predictions.append(
                                {
                                    "record_id": example.record_id,
                                    "video_id": example.video_id,
                                    "candidate_ids": ids,
                                    "target_candidate_ids": target_ids,
                                    "selected": choices,
                                    "conditional_probabilities": conditional_probs,
                                    "none_probability": none_prob,
                                    **scores,
                                }
                            )
                if len(predictions) != len(valid) or len(
                    {r["record_id"] for r in predictions}
                ) != len(valid):
                    raise ValueError("validation record mismatch")
                stem = f"fold-{fold}-{mode}"
                checkpoint = args.output / f"{stem}.pt"
                torch.save(model.state_dict(), checkpoint)
                path = args.output / f"{stem}-predictions.jsonl"
                with path.open("x", encoding="utf-8") as output:
                    for row in predictions:
                        output.write(json.dumps(row) + "\n")
                result = {
                    "fold": fold,
                    "mode": mode,
                    "train_rows": len(train),
                    "validation_rows": len(valid),
                    "train_loss_history": history,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "exact": float(np.mean([r["exact"] for r in predictions])),
                    "checkpoint_sha256": digest(checkpoint),
                    "prediction_sha256": digest(path),
                }
                receipt["results"].append(result)
                (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        receipt.update(
            {
                "status": "PASS_FIXED_TRAIN_SOURCE_CV",
                "elapsed_seconds": time.perf_counter() - start,
                "cuda_initialized": torch.cuda.is_initialized(),
            }
        )
    except Exception as error:
        receipt.update(
            {
                "status": "FAILED",
                "error": repr(error),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        raise
    finally:
        (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
