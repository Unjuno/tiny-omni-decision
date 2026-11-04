"""One fixed zero-shot relevance-to-decision diagnostic; no calibration fitting."""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.train_frozen_text_probe import _load_examples, _metrics
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.schema import DatasetManifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "validation", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert sha256_file(args.model / "model.safetensors") == (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    )
    assert sha256_file(args.validation) == (
        "7be1d1624752938553e13e1eecef5363ca2c56dd264da2915b97ddae8d0b60a8"
    )
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    )
    examples = _load_examples(args.validation, manifest, preserve_noul_criteria=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    first, second, widths = [], [], []
    for example in examples:
        widths.append(len(example.options))
        for option in example.options:
            first.append(f"Question: {example.question}\nCandidate answer: {option}")
            second.append(f"State: {example.state}")
    lengths = tokenizer(first, add_special_tokens=False, truncation=False)["input_ids"]
    if any(len(ids) + tokenizer.num_special_tokens_to_add(pair=True) >= 512 for ids in lengths):
        raise ValueError("question and candidate exceed reserved budget")
    args.output.mkdir(parents=True, exist_ok=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
        .eval()
        .to(device)
    )
    values, truncated = [], 0
    with torch.inference_mode():
        for start in range(0, len(first), 32):
            a, b = first[start : start + 32], second[start : start + 32]
            full = tokenizer(a, b, truncation=False)["input_ids"]
            truncated += sum(len(ids) > 512 for ids in full)
            inputs = tokenizer(
                a, b, max_length=512, truncation="only_second", padding=True, return_tensors="pt"
            ).to(device)
            scores = model(**inputs).logits[:, 0].float().cpu()
            if not torch.isfinite(scores).all():
                raise ValueError("nonfinite scores")
            values.extend(scores.tolist())
    logits, predictions, offset = [], [], 0
    for example, width in zip(examples, widths, strict=True):
        scores = torch.tensor(values[offset : offset + width])
        logits.append(scores)
        predictions.append(
            {
                "id": example.id,
                "options": example.options,
                "target": example.target,
                "probabilities": scores.softmax(-1).tolist(),
            }
        )
        offset += width
    report = {
        "status": "PASS_EXECUTION_ONLY",
        "metrics": _metrics(logits, [e.options.index(e.target) for e in examples]),
        "device": device,
        "wall_seconds": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "pairs": len(first),
        "state_truncated_pairs": truncated,
        "temperature": 1.0,
        "training": False,
        "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "runner_sha256": sha256_file(Path(__file__)),
        "blind_audit": False,
        "limitation": (
            "Retrieval logits softmax is not validated calibration; different task "
            "training and serialization confound architecture comparison."
        ),
    }
    (args.output / "predictions.json").write_text(json.dumps(predictions) + "\n", encoding="utf-8")
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
