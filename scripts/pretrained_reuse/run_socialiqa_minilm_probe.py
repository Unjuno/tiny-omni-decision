"""Run the frozen MiniLM SocialIQA probe with group-safe train validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

ANSWERS = ("answerA", "answerB", "answerC")


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def read_split(data_dir: Path, name: str) -> list[dict[str, Any]]:
    items = [
        json.loads(line)
        for line in (data_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = (data_dir / f"{name}-labels.lst").read_text(encoding="utf-8").splitlines()
    if len(items) != len(labels):
        raise ValueError(f"{name}: {len(items)} items but {len(labels)} labels")
    result = []
    for index, (item, label) in enumerate(zip(items, labels, strict=True)):
        if set(ANSWERS) - item.keys() or not {"context", "question"} <= item.keys():
            raise ValueError(f"{name}:{index}: missing required SocialIQA fields")
        if label not in {"1", "2", "3"}:
            raise ValueError(f"{name}:{index}: invalid 1-based label {label!r}")
        result.append(
            {
                **item,
                "id": f"{name}-{index:05d}",
                "target": int(label) - 1,
                "group": normalized(item["context"]),
            }
        )
    return result


def grouped_split(
    items: list[dict[str, Any]], seed: int, validation_fraction: float
) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        groups.setdefault(item["group"], []).append(index)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    target = max(1, round(len(items) * validation_fraction))
    validation_groups: set[str] = set()
    selected = 0
    for group in ordered:
        if selected >= target:
            break
        validation_groups.add(group)
        selected += len(groups[group])
    validation = [i for i, item in enumerate(items) if item["group"] in validation_groups]
    train = [i for i, item in enumerate(items) if item["group"] not in validation_groups]
    if not train or not validation:
        raise ValueError("group split produced an empty partition")
    if {items[i]["group"] for i in train} & {items[i]["group"] for i in validation}:
        raise AssertionError("context group leaked across train/validation")
    return train, validation


def encode(
    items: list[dict[str, Any]],
    indices: list[int],
    tokenizer: Any,
    model: nn.Module,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    pairs: list[tuple[str, str]] = []
    for index in indices:
        item = items[index]
        prompt = f"{item['context']}\nQuestion: {item['question']}"
        pairs.extend((prompt, item[key]) for key in ANSWERS)
    model.eval()
    chunks = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            full_lengths = tokenizer(
                [a for a, _ in batch],
                [b for _, b in batch],
                padding=False,
                truncation=False,
            )["input_ids"]
            truncated += sum(len(tokens) > max_length for tokens in full_lengths)
            encoded = tokenizer(
                [a for a, _ in batch],
                [b for _, b in batch],
                padding=True,
                truncation="longest_first",
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            chunks.append(output.last_hidden_state[:, 0].float().cpu())
    return torch.cat(chunks).reshape(len(indices), 3, -1), truncated


class OptionScorer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(hidden_size, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = logits.softmax(-1)
    accuracy = (logits.argmax(-1) == labels).float().mean().item()
    nll = nn.functional.cross_entropy(logits, labels).item()
    one_hot = nn.functional.one_hot(labels, num_classes=3).float()
    brier = ((probabilities - one_hot) ** 2).sum(-1).mean().item()
    confidence, prediction = probabilities.max(-1)
    correct = prediction.eq(labels).float()
    ece = 0.0
    for lower in torch.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            ece += mask.float().mean().item() * abs(
                confidence[mask].mean().item() - correct[mask].mean().item()
            )
    return {"accuracy": accuracy, "nll": nll, "brier": brier, "ece_10": ece}


def fit(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[OptionScorer, list[dict[str, Any]], int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scorer = OptionScorer(train_features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    data = TensorDataset(train_features, train_labels)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    best_nll = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        scorer.train()
        loss_total = 0.0
        seen = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(scorer(features), labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(labels)
            seen += len(labels)
        scorer.eval()
        with torch.inference_mode():
            val_logits = scorer(val_features.to(device)).cpu()
        observed = metrics(val_logits, val_labels)
        history.append({"epoch": epoch, "train_ce": loss_total / seen, **observed})
        if observed["nll"] < best_nll - 1e-4:
            best_nll = observed["nll"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint")
    scorer.load_state_dict(best_state)
    return scorer, history, best_epoch


def prediction_rows(
    items: list[dict[str, Any]], indices: list[int], logits: torch.Tensor
) -> list[dict[str, Any]]:
    probs = logits.softmax(-1).tolist()
    rows = []
    for local, index in enumerate(indices):
        item = items[index]
        rows.append(
            {
                "id": item["id"],
                "context_group_sha256": hashlib.sha256(item["group"].encode()).hexdigest(),
                "target": item["target"],
                "prediction": int(logits[local].argmax().item()),
                "probabilities": probs[local],
            }
        )
    return rows


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_candidate(
    name: str,
    model_id: str,
    revision: str,
    args: argparse.Namespace,
    train_items: list[dict[str, Any]],
    dev_items: list[dict[str, Any]],
    train_indices: list[int],
    val_indices: list[int],
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=args.hf_cache, local_files_only=args.local_files_only
    )
    encoder = AutoModel.from_pretrained(
        model_id, revision=revision, cache_dir=args.hf_cache, local_files_only=args.local_files_only
    ).to(device)
    encoder.requires_grad_(False)
    parameter_count = sum(p.numel() for p in encoder.parameters())
    train_x, train_truncated = encode(
        train_items,
        train_indices,
        tokenizer,
        encoder,
        device,
        args.max_length,
        args.encode_batch_size,
    )
    val_x, val_truncated = encode(
        train_items,
        val_indices,
        tokenizer,
        encoder,
        device,
        args.max_length,
        args.encode_batch_size,
    )
    val_y = torch.tensor([train_items[i]["target"] for i in val_indices])
    train_y = torch.tensor([train_items[i]["target"] for i in train_indices])
    scorer, history, best_epoch = fit(
        train_x,
        train_y,
        val_x,
        val_y,
        args.seed,
        device,
        args.max_epochs,
        args.patience,
        args.question_batch_size,
    )
    val_logits = scorer(val_x.to(device)).cpu()
    candidate_dir = output / name
    candidate_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {key: value.cpu() for key, value in scorer.state_dict().items()},
        candidate_dir / "best-readout.pt",
    )
    save_json(candidate_dir / "training-history.json", history)
    val_rows = prediction_rows(train_items, val_indices, val_logits)
    (candidate_dir / "internal-validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in val_rows), encoding="utf-8"
    )
    result: dict[str, Any] = {
        "candidate": name,
        "model_id": model_id,
        "revision": revision,
        "encoder_parameters": parameter_count,
        "readout_parameters": sum(p.numel() for p in scorer.parameters()),
        "max_length": args.max_length,
        "train_pairs": len(train_indices) * len(ANSWERS),
        "train_truncated_pairs": train_truncated,
        "internal_validation_pairs": len(val_indices) * len(ANSWERS),
        "internal_validation_truncated_pairs": val_truncated,
        "best_epoch": best_epoch,
        "internal_validation": metrics(val_logits, val_y),
        "history": history,
        "elapsed_feature_and_fit_seconds": time.perf_counter() - start,
        "validation_predictions_sha256": hashlib.sha256(
            (candidate_dir / "internal-validation-predictions.jsonl").read_bytes()
        ).hexdigest(),
        "readout_sha256": hashlib.sha256(
            (candidate_dir / "best-readout.pt").read_bytes()
        ).hexdigest(),
    }
    # Publisher development data is touched only after validation-based selection
    # of epoch/checkpoint for this candidate; it is never used for fitting.
    dev_indices = list(range(len(dev_items)))
    dev_x, dev_truncated = encode(
        dev_items, dev_indices, tokenizer, encoder, device, args.max_length, args.encode_batch_size
    )
    dev_y = torch.tensor([item["target"] for item in dev_items])
    with torch.inference_mode():
        dev_logits = scorer(dev_x.to(device)).cpu()
    result["publisher_dev"] = metrics(dev_logits, dev_y)
    result["publisher_dev_rows"] = len(dev_items)
    result["publisher_dev_pairs"] = len(dev_items) * len(ANSWERS)
    result["publisher_dev_truncated_pairs"] = dev_truncated
    dev_rows = prediction_rows(dev_items, dev_indices, dev_logits)
    dev_path = candidate_dir / "publisher-dev-predictions.jsonl"
    dev_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dev_rows), encoding="utf-8"
    )
    result["publisher_dev_predictions_sha256"] = hashlib.sha256(dev_path.read_bytes()).hexdigest()
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=96)
    parser.add_argument("--question-batch-size", type=int, default=96)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train_items = read_split(args.data_dir, "train")
    dev_items = read_split(args.data_dir, "dev")
    train_groups, val_groups = {x["group"] for x in train_items}, {x["group"] for x in dev_items}
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f"publisher train/dev context overlap: {len(overlap)}")
    train_indices, val_indices = grouped_split(train_items, args.seed, 0.10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_manifest = {
        "seed": args.seed,
        "group_key": "normalized context",
        "train_items": len(train_indices),
        "internal_validation_items": len(val_indices),
        "train_context_groups": len({train_items[i]["group"] for i in train_indices}),
        "internal_validation_context_groups": len({train_items[i]["group"] for i in val_indices}),
        "context_overlap_train_internal_validation": 0,
        "publisher_train_items": len(train_items),
        "publisher_dev_items": len(dev_items),
        "publisher_train_dev_context_overlap": 0,
        "train_indices_sha256": hashlib.sha256(json.dumps(train_indices).encode()).hexdigest(),
        "internal_validation_indices_sha256": hashlib.sha256(
            json.dumps(val_indices).encode()
        ).hexdigest(),
        "publisher_dev_id_order_sha256": hashlib.sha256(
            "\n".join(x["id"] for x in dev_items).encode()
        ).hexdigest(),
    }
    save_json(args.output_dir / "split-manifest.json", split_manifest)
    save_json(
        args.output_dir / "effective-config.json",
        {
            "seed": args.seed,
            "max_length": args.max_length,
            "encode_batch_size": args.encode_batch_size,
            "question_batch_size": args.question_batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "group_key": "normalized context",
            "validation_fraction_target": 0.10,
            "encoder_trainable": False,
            "readout": "shared option MLP: hidden->128 tanh->1",
            "readout_optimizer": "AdamW lr=0.001 weight_decay=0.0001",
            "early_stopping": "minimum internal validation NLL delta 0.0001; configured patience",
            "feature": "last-layer CLS for [context + question, answer] pair",
        },
    )
    models = [
        (
            "minilm_l3",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "4ca70771034acceecb2e72475f72050fcdde4ddc",
        ),
        (
            "minilm_l6",
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        ),
    ]
    results = []
    for name, model_id, revision in models:
        results.append(
            run_candidate(
                name,
                model_id,
                revision,
                args,
                train_items,
                dev_items,
                train_indices,
                val_indices,
                args.output_dir,
                device,
            )
        )
    selected = min(results, key=lambda item: item["internal_validation"]["nll"])
    save_json(
        args.output_dir / "result.json",
        {
            "status": "completed",
            "seed": args.seed,
            "device": str(device),
            "selection_rule": (
                "lowest grouped internal-validation NLL; ties use listed candidate order"
            ),
            "selected_candidate": selected["candidate"],
            "results": results,
            "label_counts_publisher_train": dict(
                Counter(str(x["target"] + 1) for x in train_items)
            ),
            "label_counts_publisher_dev": dict(Counter(str(x["target"] + 1) for x in dev_items)),
            "notes": [
                "publisher dev used once only after checkpoint selection per candidate",
                "official test split was not acquired or loaded",
                "frozen encoders; only shared scorer trained",
            ],
        },
    )


if __name__ == "__main__":
    main()
