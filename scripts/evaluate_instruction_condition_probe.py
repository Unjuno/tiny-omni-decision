"""Fixed instruction-model comparison; constrained option logits, no generation."""

import argparse
import json
import time
from pathlib import Path

from scripts.evaluate_text_condition_probe import examples
from tiny_omni_decision.dataset import sha256_file


def prompt(row):
    return (
        f"State: {row['state']}\nQuestion: {row['question']}\n"
        "Options: A, B, C.\nReturn only the option label A, B, or C."
    )


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.train_frozen_text_probe import _metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        sha256_file(args.model / "model.safetensors")
        != "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
    ):
        raise ValueError("checkpoint hash mismatch")
    args.output.mkdir(exist_ok=False, parents=True)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    rows = examples()
    write("inputs.json", rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = [tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B", "C")]
    if any(len(t) != 1 for t in tokens) or len({t[0] for t in tokens}) != 3:
        raise ValueError("option labels must be distinct single tokens")
    ids = [t[0] for t in tokens]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(row)}], tokenize=False, add_generation_prompt=True
        )
        for row in rows
    ]
    lengths = [len(t) for t in tokenizer(texts, truncation=False)["input_ids"]]
    if max(lengths) > 2048:
        raise ValueError("unexpected probe length")
    write(
        "protocol.json",
        {
            "model_revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
            "runner_sha256": sha256_file(Path(__file__)),
            "option_token_ids": ids,
            "input_sha256": sha256_file(args.output / "inputs.json"),
            "scoring": "next-token A/B/C logits, conditional softmax, temperature1",
            "training": False,
            "calibration": False,
            "selection": False,
        },
    )
    write("prompts.json", texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32)
        .to(device)
        .eval()
    )
    model.requires_grad_(False)
    scores, mass = [], []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            inputs = tokenizer(
                texts[start : start + 16], padding=True, truncation=False, return_tensors="pt"
            ).to(device)
            logits = model(**inputs, use_cache=False).logits[:, -1].float()
            scores.extend(logits[:, ids].cpu())
            mass.extend(logits.softmax(-1)[:, ids].sum(-1).cpu().tolist())
    values = torch.stack(scores)
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite scores")
    predicted = values.argmax(-1).tolist()
    targets = [r["target"] for r in rows]
    write(
        "predictions.json",
        [
            dict(r, prediction=p, probabilities=v, unconstrained_option_mass=m)
            for r, p, v, m in zip(rows, predicted, values.softmax(-1).tolist(), mass, strict=True)
        ],
    )
    result = {
        "status": "PASS_EXECUTION_ONLY",
        "metrics": _metrics(list(values), targets),
        "condition_pairs_both_correct": sum(
            predicted[i] == targets[i] and predicted[i + 1] == targets[i + 1]
            for i in range(0, 256, 2)
        ),
        "condition_pairs_changed": sum(predicted[i] != predicted[i + 1] for i in range(0, 256, 2)),
        "pairs": 128,
        "prediction_counts": {str(i): predicted.count(i) for i in range(3)},
        "mean_unconstrained_option_mass": sum(mass) / len(mass),
        "max_tokens": max(lengths),
        "seconds": time.perf_counter() - started,
        "device": device,
        "dtype": "float32",
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "scope": "Observed constructed diagnostic; no product qualification or tuning.",
    }
    write("result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
