"""Train identical frozen-encoder readouts on a template-held-out Jamp split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from scripts.pretrained_reuse.build_jamp_grouped_split import LABELS, create_frozen_split
except ModuleNotFoundError:
    from build_jamp_grouped_split import LABELS, create_frozen_split

LINE_WEIGHT_SHA256 = "9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb"
RURI_WEIGHT_SHA256 = "e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832"
TOHOKU_WEIGHT_SHA256 = "e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4"
LINE_REVISION = "c32d05753d0d4be0f60482291a9452ec7e6f7d68"
RURI_REVISION = "24899e5de370b56d179604a007c0d727bf144504"
TOHOKU_REVISION = "65243d6e5629b969c77309f217bd7b1a79d43c7e"
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


class Readout(nn.Module):
    def __init__(self, hidden_size: int, readout_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, readout_size),
            nn.Tanh(),
            nn.Linear(readout_size, len(LABELS)),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


def _load_encoder(
    model_dir: Path,
    *,
    candidate: str,
    line_layers: int,
    device: torch.device,
) -> tuple[Any, nn.Module, dict[str, Any]]:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if line_layers:
        if config.model_type != "distilbert" or line_layers >= int(config.n_layers):
            raise ValueError("layer truncation is restricted to shallower DistilBERT")
        config.n_layers = line_layers
    trust_remote_code = config.model_type == "distilbert"
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=candidate == "ruri-v3-30m",
        trust_remote_code=trust_remote_code,
    )
    weight_file = "model.safetensors" if candidate == "ruri-v3-30m" else "pytorch_model.bin"
    expected_weight_sha = {
        "ruri-v3-30m": RURI_WEIGHT_SHA256,
        "line-distilbert-ja-first4": LINE_WEIGHT_SHA256,
        "tohoku-bert-base-ja-v3": TOHOKU_WEIGHT_SHA256,
    }[candidate]
    actual_weight_sha = sha256_file(model_dir / weight_file)
    if actual_weight_sha != expected_weight_sha:
        raise ValueError(f"pinned model weight hash mismatch for {candidate}")
    model, loading_info = AutoModel.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        output_loading_info=True,
        trust_remote_code=trust_remote_code,
        use_safetensors=candidate == "ruri-v3-30m",
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    missing = sorted(str(key) for key in loading_info["missing_keys"])
    unexpected = sorted(str(key) for key in loading_info["unexpected_keys"])
    if missing:
        raise ValueError(f"encoder checkpoint has missing keys for {candidate}: {missing[:8]}")
    dropped_layers = sorted(
        {
            int(key.split("transformer.layer.", 1)[1].split(".", 1)[0])
            for key in unexpected
            if "transformer.layer." in key
            and key.split("transformer.layer.", 1)[1].split(".", 1)[0].isdigit()
        }
    )
    if candidate == "line-distilbert-ja-first4" and dropped_layers != [4, 5]:
        raise ValueError(f"expected to drop pretrained DistilBERT layers 4 and 5: {dropped_layers}")
    expected_pretraining_heads = candidate == "tohoku-bert-base-ja-v3" and all(
        key.startswith("cls.") for key in unexpected
    )
    if candidate != "line-distilbert-ja-first4" and unexpected and not expected_pretraining_heads:
        raise ValueError(f"unexpected checkpoint keys for {candidate}: {unexpected[:8]}")
    embedding_rows = int(model.get_input_embeddings().num_embeddings)
    if int(tokenizer.vocab_size) > embedding_rows:
        raise ValueError("tokenizer vocabulary exceeds encoder embedding table")
    metadata = {
        "model_type": config.model_type,
        "layers": int(getattr(config, "n_layers", getattr(config, "num_hidden_layers", 0))),
        "hidden_size": int(config.hidden_size),
        "encoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_encoder_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "embedding_rows": embedding_rows,
        "tokenizer_vocabulary_size": int(tokenizer.vocab_size),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "dropped_distilbert_layers": dropped_layers,
        "weight_file": weight_file,
        "weight_sha256": actual_weight_sha,
        "model_files_sha256": {
            file.name: sha256_file(file)
            for file in sorted(model_dir.iterdir())
            if file.is_file() and file.name not in {"README.md"}
        },
    }
    if metadata["trainable_encoder_parameters"]:
        raise ValueError("frozen encoder unexpectedly contains trainable parameters")
    return tokenizer, model, metadata


def encode_features(
    model_dir: Path,
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    candidate: str,
    line_layers: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    tokenizer, encoder, metadata = _load_encoder(
        model_dir,
        candidate=candidate,
        line_layers=line_layers,
        device=device,
    )
    if max_length > int(encoder.config.max_position_embeddings):
        raise ValueError(f"max_length exceeds {candidate} positional capacity")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    features = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), metadata["hidden_size"]),
    )
    premises = [row["premise"] for row in rows]
    hypotheses = [row["hypothesis"] for row in rows]
    max_raw_tokens = 0
    truncated_pairs = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            raw = tokenizer(
                premises[start:end],
                hypotheses[start:end],
                truncation=False,
                padding=False,
            )
            lengths = [len(input_ids) for input_ids in raw["input_ids"]]
            max_raw_tokens = max(max_raw_tokens, max(lengths, default=0))
            truncated_pairs += sum(length > max_length for length in lengths)
            batch = tokenizer(
                premises[start:end],
                hypotheses[start:end],
                truncation="longest_first",
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            if int(batch["input_ids"].max()) >= metadata["embedding_rows"]:
                raise ValueError(f"tokenizer emitted out-of-range token IDs for {candidate}")
            hidden = encoder(**batch).last_hidden_state
            weights = batch["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            array = pooled.float().cpu().numpy()
            if array.shape != (end - start, metadata["hidden_size"]):
                raise ValueError(f"unexpected pooled feature shape for {candidate}")
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite frozen features for {candidate}")
            features[start:end] = array.astype(np.float16)
            del batch, raw, hidden, weights, pooled, array
    features.flush()
    metadata.update(
        {
            "feature_shape": [len(rows), metadata["hidden_size"]],
            "feature_dtype": "float16",
            "pooling": "attention-mask-aware mean over attended pair tokens",
            "max_untruncated_pair_tokens": max_raw_tokens,
            "truncated_pairs": truncated_pairs,
            "feature_extraction_seconds": time.perf_counter() - started,
            "peak_allocated_vram_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "feature_sha256": sha256_file(output_path),
        }
    )
    if any(
        parameter.requires_grad or parameter.grad is not None for parameter in encoder.parameters()
    ):
        raise ValueError("frozen encoder changed or acquired gradients")
    del encoder, tokenizer, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def metrics(logits: Tensor, targets: Tensor, *, bins: int = 15) -> dict[str, Any]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    correct = predictions.eq(targets)
    confidence = probabilities.max(dim=-1).values
    ece = 0.0
    bin_counts: list[int] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        count = int(mask.sum())
        bin_counts.append(count)
        if count:
            ece += (
                count
                / len(targets)
                * abs(float(confidence[mask].mean()) - float(correct[mask].float().mean()))
            )
    class_accuracy = {}
    for label_id, label in enumerate(LABELS):
        mask = targets.eq(label_id)
        class_accuracy[label] = float(correct[mask].float().mean()) if bool(mask.any()) else None
    return {
        "count": int(targets.numel()),
        "accuracy": float(correct.float().mean()),
        "macro_class_accuracy": float(
            np.mean([value for value in class_accuracy.values() if value is not None])
        ),
        "accuracy_by_label": class_accuracy,
        "nll": float(F.cross_entropy(logits.float(), targets)),
        "brier": float(
            (probabilities - F.one_hot(targets, num_classes=len(LABELS))).square().sum(-1).mean()
        ),
        "ece_15_bins": ece,
        "ece_bin_counts": bin_counts,
        "mean_confidence": float(confidence.mean()),
    }


def predict(
    scorer: nn.Module, features: np.ndarray, targets: Tensor, device: torch.device, batch_size: int
) -> tuple[Tensor, dict[str, Any]]:
    scorer.eval()
    logits: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch = torch.from_numpy(np.asarray(features[start:end], dtype=np.float32)).to(device)
            logits.append(scorer(batch).float().cpu())
    result = torch.cat(logits)
    return result, metrics(result, targets)


def fit_readout(
    train_features: np.ndarray,
    train_rows: list[dict[str, str]],
    validation_features: np.ndarray,
    validation_rows: list[dict[str, str]],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    min_delta: float,
) -> tuple[Readout, Readout, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    hidden_size = int(train_features.shape[-1])
    scorer = Readout(hidden_size).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_targets = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in train_rows])
    validation_targets = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in validation_rows])
    order_rng = random.Random(seed)
    best_nll, best_epoch, best_state = float("inf"), 0, None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_rows)))
        order_rng.shuffle(order)
        scorer.train()
        train_ce_sum, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = torch.from_numpy(np.asarray(train_features[indices], dtype=np.float32)).to(
                device
            )
            targets = train_targets[indices].to(device)
            logits = scorer(batch)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite readout training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
            if not torch.isfinite(norm):
                raise ValueError("non-finite readout gradient norm")
            optimizer.step()
            train_ce_sum += float(loss.detach()) * len(indices)
            seen += len(indices)
        train_logits, train_metrics = predict(scorer, train_features, train_targets, device, 512)
        validation_logits, validation_metrics = predict(
            scorer, validation_features, validation_targets, device, 512
        )
        current_nll = float(validation_metrics["nll"])
        history.append(
            {
                "epoch": epoch,
                "optimizer_updates": (len(train_rows) + batch_size - 1) // batch_size,
                "rolling_train_ce": train_ce_sum / seen,
                "fixed_train": train_metrics,
                "validation": validation_metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        state = {key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()}
        if current_nll < best_nll - min_delta:
            best_nll, best_epoch, best_state = current_nll, epoch, state
    if best_state is None:
        raise RuntimeError("validation failed to select a checkpoint")
    terminal_state = {
        key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
    }
    best = Readout(hidden_size)
    final = Readout(hidden_size)
    best.load_state_dict(best_state)
    final.load_state_dict(terminal_state)
    best.to(device).eval()
    final.to(device).eval()
    return (
        best,
        final,
        {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "selection_metric": "validation_nll",
            "selection_min_delta": min_delta,
            "early_stopping": False,
            "history": history,
            "trainable_readout_parameters": sum(
                parameter.numel() for parameter in best.parameters()
            ),
            "fit_elapsed_seconds": time.perf_counter() - started,
        },
    )


def _save_predictions(path: Path, rows: list[dict[str, str]], logits: Tensor) -> None:
    probabilities = torch.softmax(logits.float(), dim=-1).tolist()
    predictions = logits.argmax(-1).tolist()
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row, prediction, probs in zip(rows, predictions, probabilities, strict=True):
            file.write(
                json.dumps(
                    {
                        "id": row["num"],
                        "template_num": row["template_num"],
                        "category": row["category"],
                        "target": row["gold_label"],
                        "prediction": LABELS[prediction],
                        "probabilities": dict(zip(LABELS, probs, strict=True)),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _metrics_by(rows: list[dict[str, str]], logits: Tensor, field: str) -> dict[str, Any]:
    result = {}
    labels = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in rows])
    groups = sorted({row[field] for row in rows})
    for group in groups:
        indices = [index for index, row in enumerate(rows) if row[field] == group]
        result[group] = metrics(logits[indices], labels[indices])
    return result


def run_candidate(
    *,
    name: str,
    model_dir: Path,
    line_layers: int,
    revision: str,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    output_dir: Path,
    device: torch.device,
    config: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_start = time.perf_counter()
    train_path = output_dir / "train-features.npy"
    validation_path = output_dir / "validation-features.npy"
    train_encoder = encode_features(
        model_dir,
        train_rows,
        train_path,
        candidate=name,
        line_layers=line_layers,
        device=device,
        batch_size=config["encode_batch_size"],
        max_length=config["max_length"],
    )
    validation_encoder = encode_features(
        model_dir,
        validation_rows,
        validation_path,
        candidate=name,
        line_layers=line_layers,
        device=device,
        batch_size=config["encode_batch_size"],
        max_length=config["max_length"],
    )
    train_features = np.load(train_path, mmap_mode="r")
    validation_features = np.load(validation_path, mmap_mode="r")
    best, final, fit = fit_readout(
        train_features,
        train_rows,
        validation_features,
        validation_rows,
        device=device,
        seed=config["seed"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        min_delta=config["selection_min_delta"],
    )
    best_path, final_path = output_dir / "best-readout.pt", output_dir / "final-readout.pt"
    torch.save({key: value.detach().cpu() for key, value in best.state_dict().items()}, best_path)
    torch.save({key: value.detach().cpu() for key, value in final.state_dict().items()}, final_path)
    targets = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in validation_rows])
    best_logits, best_metrics = predict(best, validation_features, targets, device, 512)
    final_logits, final_metrics = predict(final, validation_features, targets, device, 512)
    reloaded = Readout(int(train_features.shape[-1]))
    reloaded.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    reloaded.to(device).eval()
    reloaded_logits, reload_metrics = predict(reloaded, validation_features, targets, device, 512)
    max_reload_delta = float((best_logits - reloaded_logits).abs().max())
    if max_reload_delta > 1e-6 or reload_metrics != best_metrics:
        raise ValueError("selected readout did not reproduce validation logits after reload")
    prediction_path = output_dir / "best-validation-predictions.jsonl"
    _save_predictions(prediction_path, validation_rows, best_logits)
    encoder_parameters = int(train_encoder["encoder_parameters"])
    return {
        "candidate": name,
        "revision": revision,
        "encoder": train_encoder,
        "validation_encoder": validation_encoder,
        "encoder_and_readout_wall_seconds": time.perf_counter() - feature_start,
        "encoder_parameters": encoder_parameters,
        "readout_trainable_parameters": fit["trainable_readout_parameters"],
        "total_text_parameters": encoder_parameters + fit["trainable_readout_parameters"],
        "four_tower_parameters_if_replacing_ruri": 178_689_152
        - 36_705_536
        + encoder_parameters
        + fit["trainable_readout_parameters"],
        "selected_checkpoint": {"epoch": fit["best_epoch"], "metrics": best_metrics},
        "terminal_checkpoint": {"epoch": fit["epochs_completed"], "metrics": final_metrics},
        "validation_by_category": _metrics_by(validation_rows, best_logits, "category"),
        "fit": fit,
        "best_readout_sha256": sha256_file(best_path),
        "final_readout_sha256": sha256_file(final_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "best_reload_max_logit_delta": max_reload_delta,
        "peak_allocated_vram_bytes": max(
            train_encoder["peak_allocated_vram_bytes"],
            validation_encoder["peak_allocated_vram_bytes"],
        ),
        "smoke_only": smoke,
        "product_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ruri-model-dir", type=Path, required=True)
    parser.add_argument("--line-distilbert-dir", type=Path, required=True)
    parser.add_argument("--tohoku-bert-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--selection-min-delta", type=float, default=0.001)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_num_threads(1)

    split_manifest_path = args.data_dir / "split-manifest.json"
    train_path, validation_path = args.data_dir / "train.tsv", args.data_dir / "validation.tsv"
    if not split_manifest_path.exists() or not train_path.exists() or not validation_path.exists():
        split_manifest = create_frozen_split(
            args.data_dir / "train_all.tsv", args.data_dir / "frozen-split-v1", seed=args.seed
        )
        split_manifest_path = args.data_dir / "frozen-split-v1" / "split-manifest.json"
        train_path = args.data_dir / "frozen-split-v1" / "train.tsv"
        validation_path = args.data_dir / "frozen-split-v1" / "validation.tsv"
    else:
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest["split"]["seed"] != args.seed:
        raise ValueError("split seed differs from the requested training seed")
    if sha256_file(train_path) != split_manifest["train_sha256"]:
        raise ValueError("frozen train split hash mismatch")
    if sha256_file(validation_path) != split_manifest["validation_sha256"]:
        raise ValueError("frozen validation split hash mismatch")
    train_rows, validation_rows = read_tsv(train_path), read_tsv(validation_path)
    if args.smoke_rows:
        train_rows = train_rows[: args.smoke_rows]
        validation_rows = validation_rows[: max(30, args.smoke_rows)]
    config = {
        "experiment_id": "jamp-frozen-japanese-text-reuse-v1",
        "seed": args.seed,
        "epochs": 2 if args.smoke_rows else args.epochs,
        "batch_size": args.batch_size,
        "encode_batch_size": args.encode_batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "selection_min_delta": args.selection_min_delta,
        "readout": "frozen pair encoder -> mean pooling -> 128 tanh -> 3 logits",
        "pooling": "attention-mask-aware mean over premise+hypothesis tokens",
        "checkpoint_selection": (
            "best validation NLL with minimum improvement 0.001; continue all epochs"
        ),
        "early_stopping": False,
        "device": str(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_train_sha256": split_manifest["source_file_sha256"],
        "frozen_train_sha256": split_manifest["train_sha256"],
        "frozen_validation_sha256": split_manifest["validation_sha256"],
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_seed": split_manifest["split"]["seed"],
        "source_license": "CC-BY-SA-4.0",
        "official_test_loaded": False,
        "sealed_audit_loaded": False,
        "cost_usd": 0,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    runner_path = Path(__file__)
    config["runner_sha256"] = sha256_file(runner_path)
    config["effective_config_sha256"] = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(args.output_dir / "effective-config.json", config)
    repository_root = Path(__file__).resolve().parents[2]
    source_commit = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    candidates = [
        (
            "ruri-v3-30m",
            args.ruri_model_dir,
            0,
            RURI_REVISION,
        ),
        (
            "line-distilbert-ja-first4",
            args.line_distilbert_dir,
            4,
            LINE_REVISION,
        ),
        (
            "tohoku-bert-base-ja-v3",
            args.tohoku_bert_dir,
            0,
            TOHOKU_REVISION,
        ),
    ]
    reports = []
    started = datetime.now(UTC)
    for name, model_dir, line_layers, revision in candidates:
        reports.append(
            run_candidate(
                name=name,
                model_dir=model_dir,
                line_layers=line_layers,
                revision=revision,
                train_rows=train_rows,
                validation_rows=validation_rows,
                output_dir=args.output_dir / name,
                device=device,
                config=config,
                smoke=bool(args.smoke_rows),
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "experiment_id": config["experiment_id"],
        "source_commit": source_commit,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "effective_config_sha256": config["effective_config_sha256"],
        "runner_sha256": config["runner_sha256"],
        "training_examples_per_candidate": len(train_rows),
        "validation_examples_per_candidate": len(validation_rows),
        "training_rows_consumed_across_candidates": sum(
            len(train_rows) * report["fit"]["epochs_completed"] for report in reports
        ),
        "unique_train_ids_per_candidate": len({row["num"] for row in train_rows}),
        "unique_validation_ids": len({row["num"] for row in validation_rows}),
        "device": str(device),
        "peak_allocated_vram_bytes": max(report["peak_allocated_vram_bytes"] for report in reports),
        "cost_usd": 0,
        "candidate_reports": reports,
        "evaluation_limitations": [
            "template-held-out validation created only from Jamp's published training data",
            "official Jamp test files were not loaded in this experiment",
            "synthetic controlled temporal NLI; not broad everyday Japanese reasoning",
            "CC BY-SA-4.0 attribution and share-alike obligations must be preserved",
        ],
        "smoke_only": bool(args.smoke_rows),
        "product_promotion": False,
    }
    write_json(args.output_dir / "comparison-report.json", result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "candidate_reports"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
