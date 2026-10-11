"""Fixed stratified training-fit diagnosis; no training or model selection."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path


def choose_groups(rows, per_scale=32, seed=83):
    groups = {}
    for row in rows:
        scale = row["numeric_transform"]["scale"]
        groups.setdefault(scale, set()).add(row["group"])
    rng = random.Random(seed)
    selected = set()
    for scale in sorted(groups):
        selected.update(rng.sample(sorted(groups[scale]), per_scale))
    return selected


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.train_frozen_text_probe import _metrics

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    specs = [
        (
            "narrow",
            "conditional-curriculum-01",
            "conditional-repair-encoder-coverage-01",
            "4d6110f4a401cfd6e50b9d0720a3a5fcc3aea146118ff7d1b0a05351764afbbe",
            "4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318",
        ),
        (
            "numeric_range",
            "numeric-range-curriculum-01",
            "conditional-repair-numeric-range-01",
            "95f4edc33711a6ac334fff77cb08300a45bbe2bc8acefad1e8591b069c076101",
            "381b64c5482872936c1135700815448095dedfa02d7a3dc46212a2ae9aec7948",
        ),
    ]
    if sha(args.model / "model.safetensors") != (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    ):
        raise ValueError("base hash mismatch")
    corpora = []
    for _, corpus, run, corpus_hash, weight_hash in specs:
        if sha(args.root / corpus / "train.jsonl") != corpus_hash:
            raise ValueError("corpus hash mismatch")
        if sha(args.root / run / "best-delta.pt") != weight_hash:
            raise ValueError("checkpoint hash mismatch")
        corpora.append(
            [
                json.loads(line)
                for line in (args.root / corpus / "train.jsonl").read_text().splitlines()
            ]
        )
    groups = choose_groups(corpora[1])
    samples = [[r for r in rows if r["group"] in groups] for rows in corpora]
    if any(len(rows) != 1024 for rows in samples) or (
        [r["id"] for r in samples[0]] != [r["id"] for r in samples[1]]
    ):
        raise ValueError("training sample alignment mismatch")
    for spec, rows in zip(specs, samples, strict=True):
        config = json.loads((args.root / spec[2] / "config.json").read_text())
        result = json.loads((args.root / spec[2] / "result.json").read_text())
        consumed = set(config["train_ids"][: result["best_step"] * 8])
        if not {r["id"] for r in rows} <= consumed or result["status"] != "PASS":
            raise ValueError("sample was not consumed by selected checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    write(
        "protocol.json",
        {
            "seed": 83,
            "groups_per_scale": 32,
            "groups": sorted(groups),
            "specs": specs,
            "runner_sha256": sha(Path(__file__)),
            "selection": False,
            "limitation": "Each model sees its own training values; not same-input causal comparison.",
        },
    )
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    start = time.perf_counter()
    reports = {}
    for spec, rows in zip(specs, samples, strict=True):
        model = (
            AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
            .to(device)
            .eval()
        )
        delta = torch.load(
            args.root / spec[2] / "best-delta.pt", weights_only=True, map_location="cpu"
        )
        if set(delta) != {
            n
            for n, _ in model.named_parameters()
            if n.startswith(("bert.encoder.layer.", "classifier."))
        }:
            raise ValueError("checkpoint scope mismatch")
        model.load_state_dict(delta, strict=False)
        model.requires_grad_(False)
        values = []
        max_tokens = 0
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
                    raise ValueError("training slice would truncate")
                logits = model(**inputs.to(device)).logits[:, 0].cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite logits")
                values.extend(logits.split([len(r["options"]) for r in batch]))
        targets = [r["options"].index(r["target"]) for r in rows]
        summary = {"overall": _metrics(values, targets), "tasks": {}, "max_tokens": max_tokens}
        for task in sorted({r["task"] for r in rows}):
            ids = [i for i, r in enumerate(rows) if r["task"] == task]
            summary["tasks"][task] = _metrics([values[i] for i in ids], [targets[i] for i in ids])
        reports[spec[0]] = summary
        write(
            spec[0] + "-predictions.json",
            [
                dict(
                    id=r["id"],
                    target=r["target"],
                    options=r["options"],
                    probabilities=v.softmax(-1).tolist(),
                )
                for r, v in zip(rows, values, strict=True)
            ],
        )
        print(spec[0], summary["overall"], flush=True)
        del model, delta
        if device == "cuda":
            torch.cuda.empty_cache()
    write(
        "result.json",
        {
            "status": "PASS_EXECUTION_ONLY",
            "reports": reports,
            "seconds": time.perf_counter() - start,
            "device": device,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        },
    )


if __name__ == "__main__":
    main()
