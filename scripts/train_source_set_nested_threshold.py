"""Nested threshold selection using only outer-training source videos."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.source_set_decision import set_loss
from scripts.summarize_source_temporal_long_curve import paired_bootstrap
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

OLD_SHA = "5df64f799220dbe95c1234556266c92e4aff332c524bc79c048e1a485fe2218b"
MEMBERSHIP = (0.2, 0.35, 0.5, 0.65, 0.8)
NONE = (0.35, 0.5, 0.65)
PROTOCOL = Path("docs/pretrained_reuse/R2_SOURCE_SET_NESTED_THRESHOLD_PROTOCOL_20261027.md")


def inner_assignment(ids):
    groups = sorted(
        set(ids), key=lambda v: (hashlib.sha256(f"set-inner-v1:17:{v}".encode()).digest(), v)
    )
    if len(groups) < 2:
        raise ValueError("insufficient videos")
    return {video: i % 2 for i, video in enumerate(groups)}


def select(row, member_threshold, none_threshold):
    if row["none_probability"] >= none_threshold:
        return {"none_visible"}
    ids, scores = row["candidate_ids"], row["conditional_probabilities"]
    if (
        not ids
        or len(ids) != len(scores)
        or any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores)
    ):
        raise ValueError("invalid candidate probabilities")
    selected = {key for key, score in zip(ids, scores, strict=True) if score >= member_threshold}
    if not selected:

        def logit(p):
            p = min(max(p, 1e-7), 1 - 1e-7)
            return math.log(p / (1 - p))

        maxlogit = max(map(logit, scores))
        selected = {
            key for key, score in zip(ids, scores, strict=True) if maxlogit - logit(score) <= 1e-6
        }
    return selected


def stratum(row):
    n = len(row["target_candidate_ids"])
    return "none" if n == 0 else "one" if n == 1 else "multi"


def exact(row, member_threshold, none_threshold):
    target = set(row["target_candidate_ids"]) or {"none_visible"}
    return select(row, member_threshold, none_threshold) == target


def choose(rows):
    if not rows or {stratum(r) for r in rows} != {"none", "one", "multi"}:
        raise ValueError("insufficient validation strata")
    records = []
    for m in MEMBERSHIP:
        for n in NONE:
            stats = {
                s: np.mean([exact(r, m, n) for r in rows if stratum(r) == s])
                for s in ("none", "one", "multi")
            }
            macro = float(np.mean(list(stats.values())))
            records.append({"membership": m, "none": n, "macro_exact": macro, "strata": stats})
    best = min(
        records,
        key=lambda r: (
            -r["macro_exact"],
            abs(r["membership"] - 0.5) + abs(r["none"] - 0.5),
            r["membership"],
            r["none"],
        ),
    )
    return best, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("output exists or outside artifact root")
    if shutil.disk_usage(ROOT).free < 2**30 + 64 * 2**20:
        raise OSError("insufficient disk reserve")
    if digest(args.outer_run / "run.json") != OLD_SHA:
        raise ValueError("outer run mismatch")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"input mismatch {path}")
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("source", SOURCE_RUNNER)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if protocol_hash != "aa1858df528ce1e1ba6467b58b305faf50a7db189d05b79a345a96c640df5390":
        raise ValueError("protocol hash mismatch")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    receipt = {
        "status": "RUNNING",
        "outer_run_sha256": OLD_SHA,
        "protocol_sha256_lf": protocol_hash,
        "script_sha256": digest(Path(__file__)),
        "loss_sha256": digest(Path("scripts/source_set_decision.py")),
        "outer_folds": [],
        "seed": 17,
        "sealed_audit_opened": False,
    }
    (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        torch.set_num_threads(4)
        examples, _ = source.load_examples(PLAN, LABELS, VIDEO, AUDIO)
        outer = fold_assignment([e.video_id for e in examples])
        old_run = json.loads((args.outer_run / "run.json").read_bytes())
        if (
            len(examples) != 2593
            or len(outer) != 320
            or hashlib.sha256(json.dumps(outer, sort_keys=True).encode()).hexdigest()
            != old_run["assignment_sha256"]
        ):
            raise ValueError("source grouping mismatch")
        lookup = {(r["fold"], r["mode"]): r for r in old_run["results"]}
        pooled_pairs = []
        for fold in range(3):
            pool = [e for e in examples if outer[e.video_id] != fold]
            held = {e.video_id for e in examples if outer[e.video_id] == fold}
            split = inner_assignment([e.video_id for e in pool])
            inner_rows = []
            inner_receipts = []
            for inner in range(2):
                train = [e for e in pool if split[e.video_id] != inner]
                valid = [e for e in pool if split[e.video_id] == inner]
                if ({e.video_id for e in train} & {e.video_id for e in valid}) or held & {
                    e.video_id for e in train + valid
                }:
                    raise ValueError("nested group overlap")
                torch.manual_seed(17)
                model = CandidateModel()
                opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                rng = np.random.default_rng(17)
                history = []
                for _ in range(7):
                    if shutil.disk_usage(ROOT).free < 2**30:
                        raise OSError("disk reserve exhausted")
                    model.train()
                    order = rng.permutation(len(train))
                    total = 0.0
                    for pos in range(0, len(order), 32):
                        indices = order[pos : pos + 32].tolist()
                        video, audio, targets, mask, _ = source.collate_examples(train, indices)
                        loss = set_loss(model(video, audio, mask, "joint"), mask, targets)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite loss")
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        if not torch.isfinite(norm):
                            raise FloatingPointError("nonfinite gradients")
                        opt.step()
                        total += float(loss.detach()) * len(indices)
                    history.append(total / len(train))
                model.eval()
                predictions = []
                with torch.inference_mode():
                    for pos in range(0, len(valid), 32):
                        ix = list(range(pos, min(pos + 32, len(valid))))
                        video, audio, _, mask, batch = source.collate_examples(valid, ix)
                        logits = model(video, audio, mask, "joint")
                        candidate = torch.sigmoid(logits[:, :-1])
                        none = torch.sigmoid(logits[:, -1])
                        for j, e in enumerate(batch):
                            predictions.append(
                                {
                                    "record_id": e.record_id,
                                    "video_id": e.video_id,
                                    "candidate_ids": list(e.candidate_ids),
                                    "target_candidate_ids": sorted(e.target_candidate_ids),
                                    "conditional_probabilities": candidate[
                                        j, : len(e.candidate_ids)
                                    ].tolist(),
                                    "none_probability": float(none[j]),
                                }
                            )
                stem = f"outer-{fold}-inner-{inner}"
                checkpoint = args.output / f"{stem}.pt"
                torch.save(model.state_dict(), checkpoint)
                predpath = args.output / f"{stem}-predictions.jsonl"
                with predpath.open("x", encoding="utf-8") as out:
                    for row in predictions:
                        out.write(json.dumps(row) + "\n")
                inner_rows.extend(predictions)
                inner_receipts.append(
                    {
                        "inner": inner,
                        "train_records": len(train),
                        "validation_records": len(valid),
                        "train_groups": len({e.video_id for e in train}),
                        "validation_groups": len({e.video_id for e in valid}),
                        "history": history,
                        "checkpoint_sha256": digest(checkpoint),
                        "prediction_sha256": digest(predpath),
                    }
                )
            if len(inner_rows) != len(pool) or len({r["record_id"] for r in inner_rows}) != len(
                pool
            ):
                raise ValueError("inner coverage mismatch")
            best, grid = choose(inner_rows)
            choice_path = args.output / f"outer-{fold}-threshold.json"
            with choice_path.open("x", encoding="utf-8") as out:
                json.dump({"selected": best, "grid": grid}, out, indent=2)
                out.write("\n")
            # Outer predictions are read only after selection is persisted.
            predpath = args.outer_run / f"fold-{fold}-joint-predictions.jsonl"
            if digest(predpath) != lookup[(fold, "joint")]["prediction_sha256"]:
                raise ValueError("outer prediction mismatch")
            outer_rows = list(map(json.loads, predpath.read_bytes().splitlines()))
            if {r["video_id"] for r in outer_rows} != held:
                raise ValueError("outer group mismatch")
            for row in outer_rows:
                old = exact(row, 0.5, 0.5)
                if old != row["exact"]:
                    raise ValueError("fixed rule replay mismatch")
                tuned = exact(row, best["membership"], best["none"])
                pooled_pairs.append(
                    (
                        {"video_id": row["video_id"], "correct": old, "stratum": stratum(row)},
                        {"correct": tuned},
                    )
                )
            entry = {
                "fold": fold,
                "inner": inner_receipts,
                "choice_sha256": digest(choice_path),
                "selected": best,
                "outer_records": len(outer_rows),
            }
            receipt["outer_folds"].append(entry)
            (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
        if len(pooled_pairs) != 2593:
            raise ValueError("outer coverage mismatch")
        results = {}
        for name in ("all", "none", "one", "multi"):
            pairs = [p for p in pooled_pairs if name == "all" or p[0]["stratum"] == name]
            results[name] = {
                "n": len(pairs),
                "baseline_accuracy": float(np.mean([x["correct"] for x, _ in pairs])),
                "selected_accuracy": float(np.mean([y["correct"] for _, y in pairs])),
                "paired_delta": paired_bootstrap(pairs, "correct"),
            }
        receipt.update(
            {
                "status": "PASS_NESTED_THRESHOLDS",
                "results": results,
                "elapsed_seconds": time.perf_counter() - start,
                "cuda_initialized": torch.cuda.is_initialized(),
                "bootstrap_draws": 10000,
                "bootstrap_seed": 17,
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
    print(json.dumps(receipt["results"], indent=2))


if __name__ == "__main__":
    main()
