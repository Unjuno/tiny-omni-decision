"""Matched CPU TRAIN-only sampler experiment; no validation or selection."""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_av_training_signal import (
    FEATURE_SHA,
    QUERY_SHA,
    ROOT,
    make_model,
    probe,
    select_train_pairs,
    sha,
    validate_pair_tensors,
    write_json,
)
from scripts.run_conditional_clevrer_av_readout import _batch_blocks, _encode_options, _logits
from scripts.train_clevrer_synthetic_av_temporal_decision import _metrics


def pair_indexes(ids, scenes, targets):
    groups = defaultdict(list)
    if len(ids) != len(scenes) or len(ids) != len(targets):
        raise ValueError("aligned pair metadata required")
    for i, pair in enumerate(ids):
        groups[str(pair)].append(i)
    pairs = []
    for pair in sorted(groups):
        indexes = groups[pair]
        if (len(indexes) != 2 or len({str(scenes[i]) for i in indexes}) != 1
                or {int(targets[i]) for i in indexes} != {0, 1}):
            raise ValueError("pairs must share a scene and have targets 0 and 1")
        pairs.append(sorted(indexes, key=lambda i: int(targets[i])))
    return torch.tensor(pairs, dtype=torch.long)


def sampler_orders(pairs, *, paired, epochs, seed):
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
            or sorted(pairs.flatten().tolist()) != list(range(pairs.numel()))):
        raise ValueError("pairs must partition all row indexes exactly once")
    rng = torch.Generator().manual_seed(seed)
    return [pairs[torch.randperm(len(pairs), generator=rng)].flatten() if paired
            else torch.randperm(pairs.numel(), generator=rng) for _ in range(epochs)]


def evaluate(model, values, query):
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(values["targets"]), 64):
            sl = slice(start, start + 64)
            blocks = _batch_blocks(values["audio"][sl], values["text"][sl], values["video"][sl])
            logits.append(_logits(model, blocks, query.unsqueeze(0).expand(
                len(values["targets"][sl]), -1, -1)))
    return torch.cat(logits)


def run_arm(name, values, query, orders, small, output):
    model = make_model(False)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.0001)
    history, consumed, seen, recent, grad = [], 0, set(), [], []
    epoch_batches = math.ceil(len(values["targets"]) / 64)
    started = time.perf_counter()
    for update in range(2049):
        if update % 128 == 0:
            logits = evaluate(model, values, query)
            record = {"step": update, "train": _metrics(logits, values["targets"]),
                      "consumed": consumed, "unique_examples": len(seen),
                      "wall_seconds": time.perf_counter() - started,
                      "recent_training_ce": float(np.mean(recent)) if recent else None,
                      "recent_gradient_norm": float(np.mean(grad)) if grad else None,
                      "fixed_32_train_probe": probe(model, small, query)}
            history.append(record)
            write_json(output / f"history-{name}.json", history)
            print(f"{name} {update}: train={record['train']}", flush=True)
            recent, grad = [], []
        if update == 2048:
            break
        indexes = orders[update // epoch_batches][
            (update % epoch_batches) * 64:(update % epoch_batches + 1) * 64]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        blocks = _batch_blocks(values["audio"][indexes], values["text"][indexes],
                               values["video"][indexes])
        scores = _logits(model, blocks, query.unsqueeze(0).expand(len(indexes), -1, -1))
        loss = F.cross_entropy(scores, values["targets"][indexes])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient")
        optimizer.step()
        consumed += len(indexes)
        seen.update(indexes.tolist())
        recent.append(float(loss.detach()))
        grad.append(float(norm))
    checkpoint = output / f"diagnostic-final-{name}.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = make_model(False)
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    torch.testing.assert_close(evaluate(reloaded, values, query), logits, rtol=0, atol=0)
    np.save(output / f"train-final-logits-{name}.npy", logits.numpy(), allow_pickle=False)
    return {"final_train": history[-1]["train"], "optimizer_updates": 2048,
            "consumed_examples": consumed, "unique_examples": len(seen),
            "parameters": sum(p.numel() for p in params),
            "wall_seconds": time.perf_counter() - started,
            "checkpoint_sha256": sha(checkpoint), "reload_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--text-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new output directory required")
    if sha(args.features) != FEATURE_SHA:
        raise ValueError("frozen feature hash differs")
    torch.set_num_threads(14)
    torch.set_num_interop_threads(1)
    query, encoder = _encode_options(args.text_model_dir)
    query = query.clone()
    if hashlib.sha256(query.numpy().astype("<f4").tobytes()).hexdigest() != QUERY_SHA:
        raise ValueError("original query hash differs")
    torch.set_num_threads(2)
    with np.load(args.features, allow_pickle=False) as bank:
        indexes = np.flatnonzero(bank["splits"] == "train")
        metadata = {key: bank[key][indexes].copy() for key in
                    ["sample_ids", "scene_ids", "pair_ids", "targets", "splits"]}
        # Reject any training scene shared with another split using metadata only.
        if set(metadata["scene_ids"]) & set(bank["scene_ids"][bank["splits"] != "train"]):
            raise ValueError("train scene overlaps another split")
        values = {name: torch.from_numpy(bank[key][indexes].copy()) for name, key in
                  {"audio": "audios", "video": "videos", "text": "texts",
                   "targets": "targets"}.items()}
    pairs = pair_indexes(metadata["pair_ids"], metadata["scene_ids"], metadata["targets"])
    validate_pair_tensors({k: v[pairs.flatten()] for k, v in values.items()})
    small_indexes = select_train_pairs(metadata, pairs=16, seed=17)
    small = {k: v[small_indexes] for k, v in values.items()}
    epochs = math.ceil(2048 / math.ceil(len(indexes) / 64))
    orders = {name: sampler_orders(pairs, paired=name == "paired", epochs=epochs, seed=17)
              for name in ["original", "paired"]}
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "running", "utc_start": datetime.now(UTC).isoformat(),
        "scope": "TRAIN-only optimizer/sampler diagnostic, not model selection or quality",
        "hypothesis": "keeping opposite-target scene pairs together changes fitting collapse",
        "single_intervention": "row shuffle vs whole-pair shuffle; pair internals target0/1",
        "all_train_rows": len(indexes), "scenes": len(set(metadata["scene_ids"])),
        "pairs": len(pairs), "sample_ids": metadata["sample_ids"].tolist(),
        "feature_sha256": FEATURE_SHA, "candidate_query_sha256": QUERY_SHA,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                  cwd=ROOT, text=True).strip(),
        "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT / "scripts/audit_av_training_signal.py",
            ROOT / "scripts/run_conditional_clevrer_av_readout.py",
            ROOT / "src/tiny_omni_decision/native_av_interaction.py"]},
        "device": "cpu", "torch": torch.__version__, "python": platform.python_version(),
        "seed": 17, "encoder": encoder, "cpu_threads": 2, "option_encoder_threads": 14,
        "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "clip_norm": 1.,
        "batch_size": 64, "max_updates": 2048, "train_measurement_interval": 128,
        "performance_early_stopping": False, "checkpoint_selection": "none; final only",
        "input_container_includes_validation": True,
        "validation_used_for_fit_or_scoring": False, "sealed_audit_opened": False,
        "sampler_hashes": {name: hashlib.sha256(torch.cat(order).numpy().astype("<i8")
                                               .tobytes()).hexdigest()
                           for name, order in orders.items()},
        "command_arguments": {k: str(v.resolve()) for k, v in vars(args).items()},
    }
    write_json(args.output / "protocol.json", protocol)
    results = {name: run_arm(name, values, query, orders[name], small, args.output)
               for name in ["original", "paired"]}
    if sha(args.features) != FEATURE_SHA:
        raise RuntimeError("frozen feature cache changed")
    write_json(args.output / "results.json", results)
    protocol.update(status="complete", utc_end=datetime.now(UTC).isoformat(),
                    protected_feature_hash_unchanged=True)
    write_json(args.output / "run-metadata.json", protocol)


if __name__ == "__main__":
    main()
