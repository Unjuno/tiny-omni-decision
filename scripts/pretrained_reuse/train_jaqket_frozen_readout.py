"""Fit a small shared option scorer over a frozen Japanese text encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

WEIGHT_HASHES = {
    "ruri-v3-30m": "e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832",
    "line-distilbert-ja-first4": "9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb",
    "tohoku-bert-base-ja-v3": "e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty data file: {path}")
    for row in rows:
        if len(row["answer_candidates"]) != 20:
            raise ValueError("JAQKET input must have exactly 20 candidates per question")
        if (
            sum(
                _normalized(option) == _normalized(row["answer_entity"])
                for option in row["answer_candidates"]
            )
            != 1
        ):
            raise ValueError("answer_entity must appear exactly once in answer_candidates")
    return rows


def _normalized(value: str) -> str:
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


class SharedOptionScorer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_size, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, features: Tensor) -> Tensor:
        return self.score(features).squeeze(-1)


def metrics(logits: Tensor, targets: Tensor, bins: int = 15) -> dict[str, float]:
    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    nll = F.cross_entropy(logits, targets).item()
    one_hot = F.one_hot(targets, num_classes=20).to(probabilities.dtype)
    brier = (probabilities - one_hot).square().sum(dim=-1).mean().item()
    confidence = probabilities.max(dim=-1).values
    correct = predictions.eq(targets).to(probabilities.dtype)
    ece = torch.zeros((), device=logits.device)
    for bin_index in range(bins):
        low, high = bin_index / bins, (bin_index + 1) / bins
        mask = (confidence > low) & (
            confidence <= high if bin_index < bins - 1 else confidence <= high
        )
        if mask.any():
            ece += (
                mask.to(probabilities.dtype).mean()
                * (confidence[mask].mean() - correct[mask].mean()).abs()
            )
    return {
        "accuracy": correct.mean().item(),
        "nll": nll,
        "brier": brier,
        "ece_15": ece.item(),
        "mean_confidence": confidence.mean().item(),
    }


def encode_split(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    encoder: nn.Module,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
    option_only: bool,
    output_path: Path,
) -> dict[str, Any]:
    count = len(rows) * 20
    hidden_size = int(encoder.config.hidden_size)
    features = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=(len(rows), 20, hidden_size)
    )
    pairs = [
        ("" if option_only else str(row["question"]), str(option))
        for row in rows
        for option in row["answer_candidates"]
    ]
    max_raw_tokens = 0
    truncated_pairs = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            questions, options = zip(*pairs[start:end], strict=True)
            raw = tokenizer(list(questions), list(options), truncation=False, padding=False)
            lengths = [len(ids) for ids in raw["input_ids"]]
            max_raw_tokens = max(max_raw_tokens, max(lengths, default=0))
            truncated_pairs += sum(length > max_length for length in lengths)
            tokens = tokenizer(
                list(questions),
                list(options),
                truncation="longest_first",
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**tokens).last_hidden_state
            weights = tokens["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            array = pooled.float().cpu().numpy().astype(np.float16)
            flat_indices = np.arange(start, end)
            features[flat_indices // 20, flat_indices % 20] = array
            del tokens, raw, hidden, weights, pooled, array
    features.flush()
    return {
        "rows": len(rows),
        "option_pairs": count,
        "max_raw_tokens": max_raw_tokens,
        "truncated_pairs": truncated_pairs,
        "max_length": max_length,
        "seconds": time.perf_counter() - started,
        "feature_sha256": sha256_file(output_path),
    }


def _targets(rows: list[dict[str, Any]]) -> Tensor:
    import unicodedata

    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    return torch.tensor(
        [
            next(
                index
                for index, option in enumerate(row["answer_candidates"])
                if normalize(option) == normalize(row["answer_entity"])
            )
            for row in rows
        ],
        dtype=torch.long,
    )


def fit_readout(
    train_features: np.ndarray,
    train_targets: Tensor,
    validation_features: np.ndarray,
    validation_targets: Tensor,
    validation_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_x = torch.tensor(np.asarray(train_features, dtype=np.float32), device=device)
    validation_x = torch.tensor(np.asarray(validation_features, dtype=np.float32), device=device)
    train_y, validation_y = train_targets.to(device), validation_targets.to(device)
    model = SharedOptionScorer(train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    best_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        total_loss = 0.0
        for indices in order.split(batch_size):
            logits = model(train_x[indices])
            loss = F.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            validation_logits = model(validation_x)
            validation_metrics = metrics(validation_logits, validation_y)
        row = {
            "epoch": epoch,
            "train_ce": total_loss / len(train_x),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        if validation_metrics["nll"] < best_nll:
            best_nll = validation_metrics["nll"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch in {1, 2, 5, 10, 20, 40, 60, 80, 100, epochs}:
            print(json.dumps(row, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError("validation selector did not select a checkpoint")
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        best_logits = model(validation_x)
        selected_metrics = metrics(best_logits, validation_y)
    probabilities = best_logits.softmax(dim=-1).cpu().numpy()
    predictions = best_logits.argmax(dim=-1).cpu().numpy()
    target_indices = validation_targets.cpu().tolist()
    with (output_dir / "validation-predictions.jsonl").open("w", encoding="utf-8") as file:
        for row, target, prediction, probability in zip(
            validation_rows, target_indices, predictions.tolist(), probabilities, strict=True
        ):
            file.write(
                json.dumps(
                    {
                        "qid": row["qid"],
                        "target_index": target,
                        "prediction_index": int(prediction),
                        "target_entity": row["answer_entity"],
                        "predicted_entity": row["answer_candidates"][int(prediction)],
                        "probabilities": probability.tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (output_dir / "training-history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(best_state, output_dir / "best-readout.pt")
    torch.save(last_state, output_dir / "last-readout.pt")
    return {
        "epochs_completed": epochs,
        "selected_epoch": best_epoch,
        "selected_validation": selected_metrics,
        "final_epoch_validation": {
            key: value for key, value in history[-1].items() if key.startswith("validation_")
        },
        "wall_seconds": time.perf_counter() - started,
        "readout_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    experiment_started_utc = datetime.now(UTC).isoformat()
    experiment_started = time.perf_counter()
    if args.candidate not in WEIGHT_HASHES:
        raise ValueError(f"unsupported candidate: {args.candidate}")
    train_rows, validation_rows = read_jsonl(args.train), read_jsonl(args.validation)
    if {str(row["qid"]) for row in train_rows} & {str(row["qid"]) for row in validation_rows}:
        raise ValueError("train and validation qids overlap")
    device = torch.device(args.device)
    model_dir = args.model_dir
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if args.candidate == "line-distilbert-ja-first4":
        if config.model_type != "distilbert" or int(config.n_layers) < 5:
            raise ValueError(
                "expected a full LINE DistilBERT checkpoint to truncate to four layers"
            )
        config.n_layers = 4
    model_file = "model.safetensors" if args.candidate == "ruri-v3-30m" else "pytorch_model.bin"
    actual_weight_hash = sha256_file(model_dir / model_file)
    if actual_weight_hash != WEIGHT_HASHES[args.candidate]:
        raise ValueError(f"pinned model weight hash mismatch: {actual_weight_hash}")
    trust_remote_code = args.candidate == "line-distilbert-ja-first4"
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_fast=args.candidate == "ruri-v3-30m",
    )
    encoder, loading = AutoModel.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_safetensors=args.candidate == "ruri-v3-30m",
        output_loading_info=True,
    )
    missing = sorted(str(key) for key in loading["missing_keys"])
    unexpected = sorted(str(key) for key in loading["unexpected_keys"])
    if missing:
        raise ValueError(f"unexpected missing encoder weights: {missing[:8]}")
    if args.candidate == "line-distilbert-ja-first4":
        dropped = sorted(
            int(key.split("transformer.layer.", 1)[1].split(".", 1)[0])
            for key in unexpected
            if "transformer.layer." in key
            and key.split("transformer.layer.", 1)[1].split(".", 1)[0].isdigit()
        )
        dropped = sorted(set(dropped))
        if dropped != [4, 5]:
            raise ValueError(
                f"expected only pretrained DistilBERT layers 4 and 5 dropped: {dropped}"
            )
    elif args.candidate == "tohoku-bert-base-ja-v3" and any(
        not key.startswith("cls.") for key in unexpected
    ):
        raise ValueError(f"unexpected BERT-base checkpoint weights: {unexpected[:8]}")
    elif args.candidate == "ruri-v3-30m" and unexpected:
        raise ValueError(f"unexpected Ruri checkpoint weights: {unexpected[:8]}")
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    def frozen_state_hash() -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(encoder.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    frozen_hash_before = frozen_state_hash()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_meta = {}
    cached = {}
    for split_name, rows in (("train", train_rows), ("validation", validation_rows)):
        cache_path = args.output_dir / f"{split_name}-features.npy"
        feature_meta[split_name] = encode_split(
            rows,
            tokenizer,
            encoder,
            device,
            batch_size=args.encode_batch_size,
            max_length=args.max_length,
            option_only=args.option_only,
            output_path=cache_path,
        )
        cached[split_name] = np.load(cache_path, mmap_mode="r")
    frozen_hash_after = frozen_state_hash()
    if frozen_hash_before != frozen_hash_after:
        raise ValueError("frozen encoder weights changed during feature extraction")
    readout_result = fit_readout(
        cached["train"],
        _targets(train_rows),
        cached["validation"],
        _targets(validation_rows),
        validation_rows,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    if device.type == "cuda":
        readout_result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    result = {
        "experiment_id": "jaqket-frozen-readout-20261124",
        "candidate": args.candidate,
        "option_only_control": args.option_only,
        "seed": args.seed,
        "started_utc": experiment_started_utc,
        "selection": (
            "lowest validation NLL; fixed 100-epoch training run with best checkpoint retained"
        ),
        "source_split_hashes": {
            "train": sha256_file(args.train),
            "validation": sha256_file(args.validation),
        },
        "encoder": {
            "model_dir": str(model_dir),
            "weight_sha256": actual_weight_hash,
            "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
            ),
            "frozen_state_sha256_before": frozen_hash_before,
            "frozen_state_sha256_after": frozen_hash_after,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        },
        "encoding": feature_meta,
        "readout": readout_result,
        "total_wall_seconds": time.perf_counter() - experiment_started,
        "training_config": {
            "epochs": args.epochs,
            "batch_size_questions": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "constant",
            "early_stopping": False,
            "checkpoint_selection": "minimum validation NLL across all 100 epochs",
            "readout": "frozen pair mean-pooling -> shared Linear(hidden,128)-Tanh-Linear(128,1)",
            "max_length": args.max_length,
            "max_length_truncation": "longest_first",
        },
        "artifact_hashes": {
            "runner_sha256": sha256_file(Path(__file__)),
            "best_readout_sha256": sha256_file(args.output_dir / "best-readout.pt"),
            "last_readout_sha256": sha256_file(args.output_dir / "last-readout.pt"),
        },
        "torch": torch.__version__,
        "device": str(device),
    }
    result_text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    (args.output_dir / "effective-config-and-result.json").write_text(result_text, encoding="utf-8")
    (args.output_dir / "run-result.json").write_text(result_text, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(WEIGHT_HASHES), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--option-only", action="store_true")
    args = parser.parse_args()
    if args.epochs != 100:
        raise ValueError("the predeclared fit uses exactly 100 epochs")
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
