"""Fixed seed-17 final-block adaptation experiment, with a separate smoke mode."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.train_frozen_text_probe import _assert_disjoint, _load_examples, _metrics
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.schema import DatasetManifest


def configure_trainable(model):
    names = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(("bert.encoder.layer.5.", "classifier."))
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if not names:
        raise ValueError("no matching final-block parameters")
    return names


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def choice_loss(scores, widths, targets):
    parts = list(scores.split(widths))
    return torch.stack(
        [F.cross_entropy(p[None], t[None]) for p, t in zip(parts, targets, strict=True)]
    ).mean()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "train", "validation", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    expected = {
        args.model
        / "model.safetensors": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
        args.train: "1f3e0f751b6fdd1115cf6af4ea3ebc13545a1148614eda4f9f56ae8492c80675",
        args.validation: "7be1d1624752938553e13e1eecef5363ca2c56dd264da2915b97ddae8d0b60a8",
    }
    for path, sha in expected.items():
        if sha256_file(path) != sha:
            raise ValueError(f"hash mismatch: {path}")
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    )
    train = _load_examples(args.train, manifest.model_copy(update={"split": "train"}), True)
    validation = _load_examples(
        args.validation, manifest.model_copy(update={"split": "validation"}), True
    )
    _assert_disjoint(train, validation)
    if args.smoke:
        train, validation = train[:16], validation[:8]
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(17)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(17)
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = {}
    for split, examples in (("train", train), ("validation", validation)):
        rows, truncated = [], 0
        for example in examples:
            first = [
                f"Question: {example.question}\nCandidate answer: {o}" for o in example.options
            ]
            second = [f"State: {example.state}"] * len(first)
            lengths = tokenizer(first, add_special_tokens=False, truncation=False)["input_ids"]
            if any(len(ids) + 3 >= 512 for ids in lengths):
                raise ValueError("question/candidate exceeds budget")
            full = tokenizer(first, second, truncation=False)["input_ids"]
            truncated += sum(len(ids) > 512 for ids in full)
            batch = tokenizer(first, second, max_length=512, truncation="only_second")
            rows.append([{k: v[i] for k, v in batch.items()} for i in range(len(first))])
        encoded[split] = rows
        write(
            args.output / f"{split}-inputs.json",
            {"ids": [e.id for e in examples], "state_truncated_pairs": truncated},
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True
    ).to(device)
    names = configure_trainable(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 1774849:
        raise ValueError("unexpected trainable parameter count")
    before = frozen_hash(model)
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01)
    config = {
        "seed": 17,
        "epochs": 1 if args.smoke else 3,
        "batch_questions": 8,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "dropout": "disabled in all modules",
        "scheduler": "constant",
        "trainable_names": names,
        "trainable_parameters": 1774849,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "argv": sys.argv,
        "device": device,
        "smoke": args.smoke,
        "runner_sha256": sha256_file(Path(__file__)),
        "torch": torch.__version__,
        "hashes": {str(p): s for p, s in expected.items()},
    }
    write(args.output / "config.json", config)

    def forward(split, indices):
        groups = [encoded[split][i] for i in indices]
        batch = tokenizer.pad(
            [v for group in groups for v in group], padding=True, return_tensors="pt"
        ).to(device)
        scores = model(**batch).logits[:, 0]
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite scores")
        return scores, [len(group) for group in groups]

    def evaluate():
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(validation), 8):
                scores, widths = forward(
                    "validation", list(range(start, min(start + 8, len(validation))))
                )
                values.extend(p.cpu() for p in scores.split(widths))
        return values

    def delta():
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    started = time.perf_counter()
    history, best, steps = [], float("inf"), 0
    for epoch in range(1, config["epochs"] + 1):
        order = list(range(len(train)))
        random.shuffle(order)
        total, count = 0.0, 0
        model.eval()  # Gradients remain enabled; frozen representations have no dropout noise.
        for start in range(0, len(order), 8):
            indices = order[start : start + 8]
            scores, widths = forward("train", indices)
            targets = torch.tensor(
                [train[i].options.index(train[i].target) for i in indices], device=device
            )
            loss = choice_loss(scores, widths, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters):
                raise ValueError("no nonzero trainable gradients")
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError("frozen parameter received gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
            steps += 1
            if steps % 128 == 0:
                print(
                    json.dumps(
                        {"epoch": epoch, "step": steps, "seconds": time.perf_counter() - started}
                    ),
                    flush=True,
                )
        values = evaluate()
        metrics = _metrics(values, [e.options.index(e.target) for e in validation])
        if frozen_hash(model) != before:
            raise ValueError("frozen weights changed")
        history.append(
            {
                "epoch": epoch,
                "step": steps,
                "rolling_train_ce": total / count,
                "validation": metrics,
                "last_gradient_norm": float(norm),
            }
        )
        if metrics["nll"] < best:
            best_values = [v.clone() for v in values]
            best, best_epoch = metrics["nll"], epoch
            torch.save(delta(), args.output / "best-delta.pt")
        write(args.output / "history.json", history)
    torch.save(delta(), args.output / "final-delta.pt")
    model.load_state_dict(
        torch.load(args.output / "best-delta.pt", weights_only=True), strict=False
    )
    first, second = evaluate(), evaluate()
    if not all(torch.equal(a, b) for a, b in zip(first, best_values, strict=True)):
        raise ValueError("best checkpoint reload mismatch")
    if not all(torch.equal(a, b) for a, b in zip(first, second, strict=True)):
        raise ValueError("repeat prediction mismatch")
    write(
        args.output / "predictions.json",
        [
            {
                "id": e.id,
                "target": e.target,
                "options": e.options,
                "probabilities": p.softmax(-1).tolist(),
            }
            for e, p in zip(validation, first, strict=True)
        ],
    )
    write(
        args.output / "result.json",
        {
            "status": "PASS",
            "smoke": args.smoke,
            "best_epoch": best_epoch,
            "metrics": _metrics(first, [e.options.index(e.target) for e in validation]),
            "frozen_sha256": before,
            "steps": steps,
            "seconds": time.perf_counter() - started,
            "best_sha256": sha256_file(args.output / "best-delta.pt"),
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        },
    )


if __name__ == "__main__":
    main()
