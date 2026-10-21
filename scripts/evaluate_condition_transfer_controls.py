"""Run the predeclared seven-arm fixed-checkpoint transfer diagnosis."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.build_condition_transfer_controls import VARIANTS
    from scripts.evaluate_text_condition_probe import checkpoint_contract
    from scripts.train_frozen_text_probe import _metrics

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    expected_delta, prefixes = checkpoint_contract("encoder_coverage")
    expected = {
        args.data
        / "inputs.jsonl": "a1bad72fff10e9a264b66557e9b2c1079954d508aa6f07972060804dd0c6221b",
        args.model
        / "model.safetensors": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
        args.checkpoint: expected_delta,
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f"hash mismatch: {path}")
    rows = [json.loads(line) for line in (args.data / "inputs.jsonl").read_text().splitlines()]
    if len(rows) != 7168 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("incomplete or duplicate inputs")
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    write(
        "protocol.json",
        {
            "paths": {str(p): h for p, h in expected.items()},
            "runner_sha256": sha(Path(__file__)),
            "training": False,
            "selection": False,
            "variants": VARIANTS,
        },
    )
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
        .to(device)
        .eval()
    )
    delta = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if set(delta) != {n for n, _ in model.named_parameters() if n.startswith(prefixes)}:
        raise ValueError("delta names mismatch")
    model.load_state_dict(delta, strict=False)
    model.requires_grad_(False)
    predictions = []
    metrics = {}
    paired = {}
    max_tokens = 0
    with torch.inference_mode():
        for variant in VARIANTS:
            selected = [r for r in rows if r["variant"] == variant]
            values = []
            for offset in range(0, len(selected), 8):
                batch = selected[offset : offset + 8]
                first = [
                    f"Question: {r['question']}\nCandidate answer: {o}"
                    for r in batch
                    for o in r["options"]
                ]
                second = [f"State: {r['state']}" for r in batch for _ in r["options"]]
                inputs = tokenizer(
                    first, second, padding=True, truncation=False, return_tensors="pt"
                )
                max_tokens = max(max_tokens, inputs["input_ids"].shape[1])
                if max_tokens > 512:
                    raise ValueError("input would truncate")
                scores = model(**inputs.to(device)).logits[:, 0].cpu()
                if not torch.isfinite(scores).all():
                    raise ValueError("nonfinite logits")
                values.extend(scores.split([len(r["options"]) for r in batch]))
            targets = [r["options"].index(r["target"]) for r in selected]
            metrics[variant] = {"overall": _metrics(values, targets), "tasks": {}}
            for task in sorted({r["task"] for r in selected}):
                ids = [i for i, r in enumerate(selected) if r["task"] == task]
                metrics[variant]["tasks"][task] = _metrics(
                    [values[i] for i in ids], [targets[i] for i in ids]
                )
            paired[variant] = np.array(
                [int(v.argmax()) == t for v, t in zip(values, targets, strict=True)]
            )
            predictions.extend(
                dict(
                    id=r["id"],
                    base_id=r["base_id"],
                    group=r["group"],
                    variant=variant,
                    target=r["target"],
                    options=r["options"],
                    probabilities=v.softmax(-1).tolist(),
                )
                for r, v in zip(selected, values, strict=True)
            )
            print(variant, metrics[variant]["overall"], flush=True)
    comparison = {}
    for variant in VARIANTS[1:]:
        diff = (
            (paired[variant].astype(float) - paired["control"].astype(float))
            .reshape(128, 8)
            .mean(1)
        )
        rng = np.random.default_rng(20261012)
        draws = diff[rng.integers(0, 128, (10000, 128))].mean(1)
        comparison[variant] = {
            "accuracy_difference": float(diff.mean()),
            "percentile_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    write("predictions.json", predictions)
    write(
        "result.json",
        {
            "status": "PASS_EXECUTION_ONLY",
            "metrics": metrics,
            "paired_state_bootstrap": comparison,
            "seed": 20261012,
            "resamples": 10000,
            "groups": 128,
            "max_tokens": max_tokens,
            "seconds": time.perf_counter() - start,
            "device": device,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "limitation": "Known development study; no multiplicity adjustment.",
        },
    )


if __name__ == "__main__":
    main()
