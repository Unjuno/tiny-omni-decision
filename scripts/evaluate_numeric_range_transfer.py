"""Post-selection comparison on the preregistered numeric transfer corpus."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def selected_step(history, result):
    if result["status"] != "PASS" or result["steps"] != 2048:
        raise ValueError("run must complete before transfer evaluation")
    if [r["step"] for r in history] != list(range(0, 2049, 128)):
        raise ValueError("incomplete history")
    if not all(math.isfinite(r["selector"]) for r in history):
        raise ValueError("nonfinite selection history")
    best = min(history, key=lambda r: r["selector"])["step"]
    if result["best_step"] != best:
        raise ValueError("selected checkpoint is inconsistent with frozen selector")
    return best


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.train_frozen_text_probe import _metrics

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "model", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    if sha(args.data) != "41df283d98a6e251482f9fedfed468e00231d179b32ae6891eadbe2bb820d16a":
        raise ValueError("transfer corpus hash mismatch")
    if sha(args.model / "model.safetensors") != (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    ):
        raise ValueError("base hash mismatch")
    roots = [args.baseline, args.candidate]
    configs = [load(p / "config.json") for p in roots]
    results = [load(p / "result.json") for p in roots]
    steps = [
        selected_step(load(p / "history.json"), r) for p, r in zip(roots, results, strict=True)
    ]
    for key in (
        "seed",
        "steps",
        "batch",
        "lr",
        "weight_decay",
        "clip",
        "dropout",
        "optimizer_reset",
        "initialization",
        "selector",
        "train_ids",
        "validation_ids",
        "trainable_policy",
    ):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"controlled field mismatch: {key}")
    if configs[1]["curriculum_training"] != "numeric_range":
        raise ValueError("wrong candidate experiment")
    if configs[0].get("curriculum_training", "original") != "original":
        raise ValueError("wrong baseline experiment")
    unchanged_paths = []
    for config, suffix, expected_hash in zip(
        configs,
        ("conditional-curriculum-01/train.jsonl", "numeric-range-curriculum-01/train.jsonl"),
        (
            "4d6110f4a401cfd6e50b9d0720a3a5fcc3aea146118ff7d1b0a05351764afbbe",
            "95f4edc33711a6ac334fff77cb08300a45bbe2bc8acefad1e8591b069c076101",
        ),
        strict=True,
    ):
        paths = {p.replace("\\", "/"): h for p, h in config["paths"].items()}
        training = [p for p in paths if p.endswith(suffix)]
        if len(training) != 1 or paths.pop(training[0]) != expected_hash:
            raise ValueError("unexpected training data")
        unchanged_paths.append(paths)
    if unchanged_paths[0] != unchanged_paths[1]:
        raise ValueError("additional source changes confound comparison")
    hashes = [sha(p / "best-delta.pt") for p in roots]
    if hashes[0] != "4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318":
        raise ValueError("baseline checkpoint mismatch")
    if any(h != r["best_sha256"] for h, r in zip(hashes, results, strict=True)):
        raise ValueError("checkpoint hash mismatch")
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    if len(rows) != 2048 or len({r["id"] for r in rows}) != 2048:
        raise ValueError("invalid transfer questions")
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    write(
        "protocol.json",
        {
            "roots": list(map(str, roots)),
            "selected_steps": steps,
            "checkpoint_sha256": hashes,
            "input_sha256": sha(args.data),
            "runner_sha256": sha(Path(__file__)),
            "selection_frozen_before_transfer": True,
        },
    )
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    metrics, correctness = [], []
    start = time.perf_counter()
    max_tokens = 0
    for label, root in zip(("baseline", "candidate"), roots, strict=True):
        model = (
            AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
            .to(device)
            .eval()
        )
        delta = torch.load(root / "best-delta.pt", weights_only=True, map_location="cpu")
        if set(delta) != {
            n
            for n, _ in model.named_parameters()
            if n.startswith(("bert.encoder.layer.", "classifier."))
        }:
            raise ValueError("delta scope mismatch")
        model.load_state_dict(delta, strict=False)
        model.requires_grad_(False)
        values = []
        with torch.inference_mode():
            for offset in range(0, len(rows), 8):
                batch = rows[offset : offset + 8]
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
                    raise ValueError("transfer input would truncate")
                logits = model(**inputs.to(device)).logits[:, 0].cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite logits")
                values.extend(logits.split([len(r["options"]) for r in batch]))
        targets = [r["options"].index(r["target"]) for r in rows]
        summary = {"overall": _metrics(values, targets), "groups": {}}
        for field in ("task", "scale"):
            keys = [r["task"] if field == "task" else r["numeric_transform"]["scale"] for r in rows]
            for key in sorted(set(keys)):
                ids = [i for i, v in enumerate(keys) if v == key]
                summary["groups"][f"{field}:{key}"] = _metrics(
                    [values[i] for i in ids], [targets[i] for i in ids]
                )
        metrics.append(summary)
        correctness.append(
            np.array([int(v.argmax()) == t for v, t in zip(values, targets, strict=True)])
        )
        write(
            label + "-predictions.json",
            [
                dict(
                    id=r["id"],
                    group=r["group"],
                    target=r["target"],
                    options=r["options"],
                    probabilities=v.softmax(-1).tolist(),
                )
                for r, v in zip(rows, values, strict=True)
            ],
        )
        print(label, summary["overall"], flush=True)
        del model, delta
        if device == "cuda":
            torch.cuda.empty_cache()
    diff = (correctness[1].astype(float) - correctness[0].astype(float)).reshape(256, 8).mean(1)
    rng = np.random.default_rng(20261012)
    draws = diff[rng.integers(0, 256, (10000, 256))].mean(1)
    write(
        "result.json",
        {
            "status": "PASS_EXECUTION_ONLY",
            "metrics": metrics,
            "accuracy_difference": float(diff.mean()),
            "paired_state_percentile_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
            "bootstrap_seed": 20261012,
            "resamples": 10000,
            "max_tokens": max_tokens,
            "seconds": time.perf_counter() - start,
            "device": device,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "limitation": "Development transfer test, not the sealed product audit.",
        },
    )


if __name__ == "__main__":
    main()
