"""Approved fixed NLI condition diagnostic; no training or calibration."""

import argparse
import json
import time
from pathlib import Path

from scripts.evaluate_text_condition_probe import examples
from tiny_omni_decision.dataset import sha256_file


def checkpoint_contract(name):
    contracts = {
        "minilm2": (
            "ec9c5b6a08b60d7a01b465c7ed44b496244dfd1bed3274c95875a74cea89ab35",
            "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
            1,
        ),
        "deberta_xsmall": (
            "56b89067a4e60362f498f39b7067020911618fad37d64fa23a5f959a04fd4080",
            "262ae02f29173eec1c250f90804dc7edc677dcff",
            0,
        ),
    }
    if name not in contracts:
        raise ValueError("unknown checkpoint")
    return contracts[name]


def hypothesis(row, option):
    if row["field"] not in ("price", "risk") or row["direction"] not in ("lowest", "highest"):
        raise ValueError("unsupported condition")
    if option not in ("A", "B", "C"):
        raise ValueError("unsupported candidate")
    return f"Candidate {option} has the {row['direction']} {row['field']}."


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.train_frozen_text_probe import _metrics

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("minilm2", "deberta_xsmall"), default="minilm2")
    args = parser.parse_args()
    weight_hash, revision, entailment_index = checkpoint_contract(args.checkpoint)
    if sha256_file(args.model / "model.safetensors") != weight_hash:
        raise ValueError("base hash mismatch")
    rows = examples()
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    write("inputs.json", rows)
    if sha256_file(args.output / "inputs.json") != (
        "8737946632cf65dd85b6360e9ed1f7b7b70ed7b59df230a0a7f5c9698233fb74"
    ):
        raise ValueError("fixed diagnostic inputs changed")
    write(
        "protocol.json",
        {
            "seed": 17,
            "states": 64,
            "questions": 256,
            "options": 3,
            "training": False,
            "selection": False,
            "runner_sha256": sha256_file(Path(__file__)),
            "base_revision": revision,
            "weight_sha256": weight_hash,
            "entailment_index": entailment_index,
            "input_sha256": sha256_file(args.output / "inputs.json"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    first, second = [], []
    for row in rows:
        for option in row["options"]:
            first.append(row["state"])
            second.append(hypothesis(row, option))
    write("rendered-pairs.json", list(zip(first, second, strict=True)))
    lengths = [len(v) for v in tokenizer(first, second, truncation=False)["input_ids"]]
    if max(lengths) > 512:
        raise ValueError("probe would truncate")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            args.model, local_files_only=True, dtype=torch.float32
        )
        .to(device)
        .eval()
    )
    if model.config.label2id.get("entailment") != entailment_index:
        raise ValueError("unexpected entailment label")
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
            scores.extend(model(**batch).logits[:, entailment_index].float().cpu().tolist())
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
        "parameters": sum(p.numel() for p in model.parameters()),
        "scoring": "entailment logit softmax over options; temperature 1",
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
