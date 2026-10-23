"""CPU-only TRAIN fitting diagnostic for the frozen conditional AV experiment.

This deliberately fits a tiny cohort. Its accuracy is not generalization quality.
No validation selector, validation scoring, model promotion, or GPU allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.run_conditional_clevrer_av_readout import (
    _batch_blocks,
    _encode_options,
    _logits,
    _make_baseline,
)
from tiny_omni_decision.native_av_interaction import ConditionalAVReadout

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SHA = "0b146fa637049d84ad88f0a71638cf30f6e967bd66add13ead0ab062bcbebdff"
QUERY_SHA = "48e8d635f56f856bf9ab2d9d4d03b5da84a47cd08f301053fcb3a581ccf60fa1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_train_pairs(bank, *, pairs, seed):
    """One whole pair per scene, ranked by a label-independent fixed hash."""
    if pairs < 1:
        raise ValueError("positive pair count required")
    groups = defaultdict(list)
    for index, pair in enumerate(bank["pair_ids"]):
        groups[str(pair)].append(index)
    eligible = []
    for pair, indexes in groups.items():
        splits = {str(bank["splits"][i]) for i in indexes}
        if "train" not in splits:
            continue
        if splits != {"train"} or len(indexes) != 2:
            raise ValueError("train pair must be complete and split-disjoint")
        scenes = {str(bank["scene_ids"][i]) for i in indexes}
        if len(scenes) != 1 or {int(bank["targets"][i]) for i in indexes} != {0, 1}:
            raise ValueError("train pair must share one scene and contain both targets")
        rank = hashlib.sha256(f"{seed}:{pair}".encode()).hexdigest()
        eligible.append((rank, pair, scenes.pop(), indexes))
    selected, seen = [], set()
    for _, _, scene, indexes in sorted(eligible):
        if scene in seen:
            continue
        seen.add(scene)
        selected.extend(sorted(indexes, key=lambda i: int(bank["targets"][i])))
        if len(seen) == pairs:
            return np.asarray(selected, dtype=np.int64)
    raise ValueError(f"requested {pairs} distinct train scenes; found {len(seen)}")


def validate_pair_tensors(values):
    for name, tensor in values.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite {name}")
    for start in range(0, len(values["targets"]), 2):
        if values["targets"][start:start + 2].tolist() != [0, 1]:
            raise ValueError("pair target order differs")
        for name in ["video", "text"]:
            if not torch.equal(values[name][start], values[name][start + 1]):
                raise ValueError(f"paired {name} must be identical")
        if torch.equal(values["audio"][start], values["audio"][start + 1]):
            raise ValueError("paired audio must differ")


def make_model(freeze_query):
    baseline = _make_baseline(17)
    torch.manual_seed(17)
    model = ConditionalAVReadout(baseline, exchange=True, gated=True, scalar_event_audio=True)
    if freeze_query:
        model.baseline.query.requires_grad_(False)
    return model


def probe(model, values, query):
    """No optimizer update: fixed-cohort margins, input gradients and controls."""
    model.eval()
    model.zero_grad(set_to_none=True)
    audio = values["audio"].detach().clone().requires_grad_(True)
    video = values["video"].detach().clone().requires_grad_(True)
    blocks = _batch_blocks(audio, values["text"], video)
    queries = query.unsqueeze(0).expand(len(audio), -1, -1)
    logits = _logits(model, blocks, queries)
    loss = F.cross_entropy(logits, values["targets"])
    loss.backward()
    base = model.baseline if isinstance(model, ConditionalAVReadout) else model
    probability = logits.detach().softmax(-1)
    margins = logits.detach()[:, 1] - logits.detach()[:, 0]
    with torch.no_grad():
        projected = base.query(query)
        reverse = _logits(model, blocks, queries.flip(1)).flip(1)
        zero = _logits(model, _batch_blocks(
            torch.zeros_like(audio), values["text"], video), queries)
    grads = {name: float(p.grad.norm()) for name, p in model.named_parameters()
             if p.grad is not None}
    output = {
        "count": len(audio), "fitting_accuracy": float(
            (logits.argmax(-1) == values["targets"]).float().mean()),
        "fitting_nll": float(loss.detach()),
        "fitting_brier": float((probability - F.one_hot(values["targets"], 2)).square().sum(-1)
                               .mean()),
        "projected_query_l2": float(torch.dist(projected[0], projected[1])),
        "margin_std": float(margins.std()),
        "paired_margin_delta_mean_abs": float((margins[0::2] - margins[1::2]).abs().mean()),
        "audio_input_gradient_norm": float(audio.grad.norm()),
        "video_input_gradient_norm": float(video.grad.norm()),
        "parameter_gradient_norms": grads,
        "option_order_max_probability_delta": float((probability - reverse.softmax(-1))
                                                     .abs().max()),
        "zero_audio_max_probability_delta": float((probability - zero.softmax(-1)).abs().max()),
    }
    if not all(np.isfinite(value) for value in output.values() if isinstance(value, float)):
        raise FloatingPointError("nonfinite diagnostic")
    model.zero_grad(set_to_none=True)
    return output


def fit(model, values, query, output, name):
    """Predeclared 256-update full-batch fit; no checkpoint or budget selection."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0001)
    blocks = _batch_blocks(values["audio"], values["text"], values["video"])
    queries = query.unsqueeze(0).expand(len(values["targets"]), -1, -1)
    history = [{"step": 0, **probe(model, values, query)}]
    query_initial = {k: v.detach().clone() for k, v in model.baseline.query.state_dict().items()}
    started = time.perf_counter()
    for step in range(1, 257):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(_logits(model, blocks, queries), values["targets"])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite fitting loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite fitting gradient")
        optimizer.step()
        if step in [1, 16, 64, 128, 256]:
            record = {"step": step, "optimizer_gradient_norm": float(norm),
                      **probe(model, values, query)}
            history.append(record)
            print(json.dumps({"arm": name, "step": step,
                              "nll": record["fitting_nll"],
                              "accuracy": record["fitting_accuracy"]}), flush=True)
    checkpoint = output / f"diagnostic-final-{name}.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = make_model(name == "fixed_query")
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(_logits(model.eval(), blocks, queries),
                                   _logits(reloaded.eval(), blocks, queries), rtol=0, atol=0)
    query_delta = max(float((v - query_initial[k]).abs().max())
                      for k, v in model.baseline.query.state_dict().items())
    if name == "fixed_query" and query_delta != 0:
        raise RuntimeError("query parameters changed despite freezing")
    write_json(output / f"history-{name}.json", history)
    return {"updates": 256, "unique_examples": len(values["targets"]),
            "example_exposures": 256 * len(values["targets"]),
            "seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "query_parameter_max_change": query_delta,
            "checkpoint_sha256": sha(checkpoint), "reload_exact": True,
            "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--text-model-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new diagnostic output required")
    if sha(args.features) != FEATURE_SHA:
        raise ValueError("frozen feature bank hash differs")
    # Match the original CPU option encoder's reduction order before hashing.
    # The small fitting experiment uses two threads only after this boundary.
    torch.set_num_threads(14)
    torch.set_num_interop_threads(1)
    with np.load(args.features, allow_pickle=False) as bank:
        indexes = select_train_pairs(bank, pairs=16, seed=17)
        ids = bank["sample_ids"][indexes].tolist()
        scenes = bank["scene_ids"][indexes].tolist()
        values = {name: torch.from_numpy(bank[key][indexes].copy()) for name, key in
                  {"audio": "audios", "video": "videos", "text": "texts",
                   "targets": "targets"}.items()}
    validate_pair_tensors(values)
    query, encoder = _encode_options(args.text_model_dir)
    query = query.clone()  # convert inference tensor to an ordinary frozen feature
    if hashlib.sha256(query.numpy().astype("<f4").tobytes()).hexdigest() != QUERY_SHA:
        raise ValueError("pinned option embedding hash differs")
    torch.set_num_threads(2)
    protected = [args.features, args.reference_run / "effective-config.json"] + [
        args.reference_run / f"{stage}-{arm}.pt" for stage in ["best", "final"]
        for arm in ["c0_native_attention", "exchange_gate"]]
    hashes = {str(p): sha(p) for p in protected}
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "running", "scope": "TRAIN fitting diagnostic; not validation quality",
        "utc_start": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "device": "cpu", "torch": torch.__version__, "python": platform.python_version(),
        "cpu_threads": 2, "option_encoder_cpu_threads": 14,
        "candidate_query_sha256": QUERY_SHA, "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runner_sha256": sha(__file__), "readout_sha256": sha(
            ROOT / "src/tiny_omni_decision/native_av_interaction.py"),
        "encoder": encoder, "protected_file_hashes": hashes,
        "seed": 17, "selection": "16 TRAIN scene-disjoint pairs by sha256(seed:pair_id)",
        "sample_ids": ids, "scene_ids": scenes,
        "feature_container_includes_validation": True,
        "validation_rows_used_for_fit_or_scoring": False, "sealed_audit_opened": False,
        "updates": 256, "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
        "clip_gradient_norm": 1.0, "batch": "all 32 selected training rows, every update",
        "arms": ["original", "fixed_query"],
        "single_intervention": "freeze baseline.query at identical seed17 initial weights",
        "hypothesis": "query collapse is sufficient to explain tiny-cohort fitting failure",
        "interpretation": "failure to fit with fixed query refutes sufficiency, not all effects",
        "checkpoint_rule": "final only; no validation/early stopping/seed/budget selection",
        "command_arguments": {k: str(v.resolve()) for k, v in vars(args).items()},
    }
    write_json(args.output / "protocol.json", protocol)
    historical = {}
    for arm in ["c0_native_attention", "exchange_gate"]:
        for stage in ["init", "best", "final"]:
            model = _make_baseline(17) if arm == "c0_native_attention" else make_model(False)
            if stage != "init":
                checkpoint = torch.load(args.reference_run / f"{stage}-{arm}.pt",
                                        map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model_state"])
            historical[f"{arm}-{stage}"] = probe(model, values, query)
    write_json(args.output / "historical-train-probes.json", historical)
    results = {name: fit(make_model(name == "fixed_query"), values, query, args.output, name)
               for name in ["original", "fixed_query"]}
    if hashes != {str(p): sha(p) for p in protected}:
        raise RuntimeError("protected input changed")
    write_json(args.output / "results.json", results)
    protocol.update(status="complete", protected_hashes_unchanged=True,
                    utc_end=__import__("datetime").datetime.now(__import__("datetime").UTC).isoformat())
    write_json(args.output / "run-metadata.json", protocol)


if __name__ == "__main__":
    main()
