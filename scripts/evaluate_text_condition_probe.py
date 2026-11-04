"""Approved seed-17 64-state conditional-ranking probe; inference only."""

import argparse
import json
import random
import time
from pathlib import Path

from tiny_omni_decision.dataset import sha256_file


def examples():
    rng = random.Random(17)
    rows = []
    for index in range(64):
        prices = rng.sample(range(10, 1000), 3)
        risks = rng.sample(range(1, 100), 3)
        state = [
            {"candidate": name, "price": p, "risk": r}
            for name, p, r in zip(("A", "B", "C"), prices, risks, strict=True)
        ]
        for field in ("price", "risk"):
            for direction in ("lowest", "highest"):
                values = [item[field] for item in state]
                value = min(values) if direction == "lowest" else max(values)
                rows.append(
                    {
                        "id": f"{index}:{field}:{direction}",
                        "state_id": index,
                        "state": json.dumps(state),
                        "question": f"Select the candidate with the {direction} {field}.",
                        "options": ["A", "B", "C"],
                        "target": values.index(value),
                        "field": field,
                        "direction": direction,
                    }
                )
    return rows


def candidate_text(option, style):
    if style == "bare":
        return option
    if style == "described":
        return f"{option}: Candidate {option}."
    raise ValueError("unknown candidate style")


def checkpoint_contract(reference):
    if reference == "original_epoch3":
        return (
            "5fa8323a47e72c3796d0da35618b603fa49784cacb670dee0e910e8d5757ed78",
            ("bert.encoder.layer.5.", "classifier."),
        )
    if reference == "encoder_coverage":
        return (
            "4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318",
            ("bert.encoder.layer.", "classifier."),
        )
    raise ValueError("unknown pinned checkpoint")


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.train_frozen_text_probe import _metrics

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "adaptation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--weights", choices=("base", "adapted"), default="adapted")
    parser.add_argument("--candidate-style", choices=("bare", "described"), default="bare")
    parser.add_argument(
        "--checkpoint-reference",
        choices=("original_epoch3", "encoder_coverage"),
        default="original_epoch3",
    )
    args = parser.parse_args()
    if (
        sha256_file(args.model / "model.safetensors")
        != "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    ):
        raise ValueError("base hash mismatch")
    delta_path = args.adaptation / "best-delta.pt"
    expected_sha, prefixes = checkpoint_contract(args.checkpoint_reference)
    if sha256_file(delta_path) != expected_sha:
        raise ValueError("selected checkpoint mismatch")
    rows = examples()
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    write("inputs.json", rows)
    write(
        "protocol.json",
        {
            "weights": args.weights,
            "checkpoint_reference": args.checkpoint_reference,
            "candidate_style": args.candidate_style,
            "seed": 17,
            "states": 64,
            "questions": 256,
            "options": 3,
            "training": False,
            "selection": False,
            "runner_sha256": sha256_file(Path(__file__)),
            "base_revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
            "delta_sha256": sha256_file(delta_path),
            "input_sha256": sha256_file(args.output / "inputs.json"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    first, second = [], []
    for row in rows:
        for option in row["options"]:
            rendered = candidate_text(option, args.candidate_style)
            first.append(f"Question: {row['question']}\nCandidate answer: {rendered}")
            second.append(f"State: {row['state']}")
    lengths = [len(v) for v in tokenizer(first, second, truncation=False)["input_ids"]]
    if max(lengths) > 512:
        raise ValueError("probe would truncate")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
        .to(device)
        .eval()
    )
    delta = torch.load(delta_path, map_location="cpu", weights_only=True)
    expected_names = {n for n, _ in model.named_parameters() if n.startswith(prefixes)}
    if set(delta) != expected_names:
        raise ValueError("delta keys mismatch")
    if args.weights == "adapted":
        model.load_state_dict(delta, strict=False)
    model.requires_grad_(False)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(first), 32):
            batch = tokenizer(
                first[start : start + 32],
                second[start : start + 32],
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(device)
            scores.extend(model(**batch).logits[:, 0].float().cpu().tolist())
    logits = torch.tensor(scores).reshape(-1, 3)
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite scores")
    predictions = logits.argmax(-1).tolist()
    targets = [r["target"] for r in rows]
    probabilities = logits.softmax(-1).tolist()
    write(
        "predictions.json",
        [
            dict(row, probabilities=p, prediction=g)
            for row, p, g in zip(rows, probabilities, predictions, strict=True)
        ],
    )
    groups = {}
    for field in ("price", "risk"):
        for direction in ("lowest", "highest"):
            ids = [
                i for i, r in enumerate(rows) if r["field"] == field and r["direction"] == direction
            ]
            groups[f"{field}:{direction}"] = _metrics(
                [logits[i] for i in ids], [targets[i] for i in ids]
            )
    correct_pairs = sum(
        predictions[i] == targets[i] and predictions[i + 1] == targets[i + 1]
        for i in range(0, 256, 2)
    )
    changed_pairs = sum(predictions[i] != predictions[i + 1] for i in range(0, 256, 2))
    result = {
        "status": "PASS_EXECUTION_ONLY",
        "weights": args.weights,
        "candidate_style": args.candidate_style,
        "metrics": _metrics(list(logits), targets),
        "groups": groups,
        "both_correct_condition_pairs": correct_pairs,
        "total_condition_pairs": 128,
        "changed_prediction_pairs": changed_pairs,
        "target_counts": {str(i): targets.count(i) for i in range(3)},
        "max_tokens": max(lengths),
        "seconds": time.perf_counter() - started,
        "device": device,
        "torch": torch.__version__,
        "dtype": str(next(model.parameters()).dtype),
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "limitation": "Constructed diagnostic; not blind product validation. No tuning follows.",
    }
    write("result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
