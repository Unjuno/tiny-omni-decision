"""Compare frozen Japanese text encoders on a train-only KUCI group split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
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
    from scripts.build_kuci_train_grouped_split import _assert_disjoint, sha256_file
except ModuleNotFoundError:
    from build_kuci_train_grouped_split import _assert_disjoint, sha256_file

LINE_WEIGHT_SHA256 = "9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb"
RURI_WEIGHT_SHA256 = "e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832"
RURI_REVISION = "24899e5de370b56d179604a007c0d727bf144504"
LINE_REVISION = "c32d05753d0d4be0f60482291a9452ec7e6f7d68"
FOUR_TOWER_BASE_PARAMETERS = 178_689_152
RURI_PARAMETERS = 36_705_536


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _model_files(path: Path) -> dict[str, str]:
    return {
        item.name: sha256_file(item)
        for item in sorted(path.iterdir())
        if item.is_file() and item.name not in {"README.md"}
    }


def _line_dropped_layers(keys: list[str]) -> set[int]:
    dropped: set[int] = set()
    for key in keys:
        match = re.search(r"transformer\.layer\.\{?([0-9, ]+)\}?\.", key)
        if match:
            dropped.update(int(value) for value in re.findall(r"\d+", match.group(1)))
    return dropped


class CandidateScorer(nn.Module):
    def __init__(self, hidden_size: int, readout_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, readout_size),
            nn.Tanh(),
            nn.Linear(readout_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != 4:
            raise ValueError("features must have shape [questions, four_options, hidden]")
        return self.layers(features).squeeze(-1)


def _masked_mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("masked mean pooling requires [batch, tokens, hidden] and [batch, tokens]")
    weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denominator


def _cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _load_encoder(
    model_dir: Path, *, line_layers: int, device: torch.device
) -> tuple[Any, nn.Module, dict[str, Any]]:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if line_layers:
        if config.model_type != "distilbert" or line_layers >= int(config.n_layers):
            raise ValueError("layer truncation is restricted to shallower DistilBERT")
        config.n_layers = line_layers
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=(config.model_type == "distilbert"),
    )
    model, loading_info = AutoModel.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        output_loading_info=True,
        use_safetensors=(line_layers == 0),
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    missing = sorted(str(key) for key in loading_info["missing_keys"])
    unexpected = sorted(str(key) for key in loading_info["unexpected_keys"])
    if missing:
        raise ValueError(f"encoder checkpoint has missing keys: {missing[:8]}")
    if line_layers and _line_dropped_layers(unexpected) != {4, 5}:
        raise ValueError("DistilBERT must load exactly pretrained layers 0 through 3")
    embedding_rows = int(model.get_input_embeddings().num_embeddings)
    if int(tokenizer.vocab_size) > embedding_rows:
        raise ValueError("tokenizer vocabulary exceeds encoder embedding table")
    metadata = {
        "model_type": config.model_type,
        "config_revision": getattr(config, "_commit_hash", None),
        "layers": int(getattr(config, "n_layers", getattr(config, "num_hidden_layers", 0))),
        "hidden_size": int(config.hidden_size),
        "encoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_encoder_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "embedding_rows": embedding_rows,
        "tokenizer_vocabulary_size": int(tokenizer.vocab_size),
        "missing_keys": missing,
        "unexpected_key_count": len(unexpected),
        "expected_dropped_distilbert_layers": sorted(_line_dropped_layers(unexpected)),
        "weight_files_sha256": _model_files(model_dir),
    }
    if any(
        parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()
    ):
        raise ValueError("frozen encoder has trainable parameters or gradients")
    return tokenizer, model, metadata


def _pair_texts(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    contexts: list[str] = []
    candidates: list[str] = []
    for row in rows:
        for letter in "abcd":
            contexts.append(str(row["context"]))
            candidates.append(str(row[f"choice_{letter}"]))
    return contexts, candidates


def encode_features(
    model_dir: Path,
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    device: torch.device,
    line_layers: int,
    batch_questions: int,
    max_length: int,
) -> dict[str, Any]:
    tokenizer, encoder, meta = _load_encoder(model_dir, line_layers=line_layers, device=device)
    if max_length > int(encoder.config.max_position_embeddings):
        raise ValueError("max_length exceeds encoder positional embedding limit")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    hidden_size = int(encoder.config.hidden_size)
    features = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 4, hidden_size),
    )
    contexts, candidates = _pair_texts(rows)
    max_raw_tokens, truncated_pairs = 0, 0
    started = time.perf_counter()
    with torch.inference_mode():
        for question_start in range(0, len(rows), batch_questions):
            question_end = min(question_start + batch_questions, len(rows))
            pair_start, pair_end = question_start * 4, question_end * 4
            raw = tokenizer(
                contexts[pair_start:pair_end],
                candidates[pair_start:pair_end],
                truncation=False,
                padding=False,
            )
            lengths = [len(input_ids) for input_ids in raw["input_ids"]]
            max_raw_tokens = max(max_raw_tokens, max(lengths, default=0))
            truncated_pairs += sum(length > max_length for length in lengths)
            batch = tokenizer(
                contexts[pair_start:pair_end],
                candidates[pair_start:pair_end],
                truncation="only_first",
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            if int(batch["input_ids"].max()) >= meta["embedding_rows"]:
                raise ValueError("tokenizer emitted an out-of-range token ID")
            hidden = _masked_mean_pool(
                encoder(**batch).last_hidden_state, batch["attention_mask"]
            ).float().cpu().numpy()
            if hidden.shape != ((question_end - question_start) * 4, hidden_size):
                raise ValueError("encoder returned unexpected CLS feature shape")
            if not np.isfinite(hidden).all():
                raise ValueError("encoder returned non-finite CLS features")
            features[question_start:question_end] = hidden.reshape(
                question_end - question_start, 4, hidden_size
            ).astype(np.float16)
            del batch, raw, hidden
    features.flush()
    meta.update(
        {
            "feature_shape": [len(rows), 4, hidden_size],
            "feature_dtype": "float16",
        "pooling": "attention-mask-aware mean over attended tokens, including special tokens",
            "encoded_candidate_pairs": len(rows) * 4,
            "max_untruncated_pair_tokens": max_raw_tokens,
            "truncated_candidate_pairs": truncated_pairs,
            "feature_extraction_seconds": time.perf_counter() - started,
            "feature_cache_sha256": sha256_file(output_path),
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        }
    )
    del encoder, tokenizer, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


def _metrics(logits: Tensor, targets: Tensor, bins: int = 15) -> dict[str, Any]:
    if logits.ndim != 2 or logits.shape[1] != 4 or logits.shape[0] != targets.numel():
        raise ValueError("metrics require matching [nonempty_rows, four_choices] logits/targets")
    if targets.numel() == 0 or bins <= 0:
        raise ValueError("metrics require at least one example and a positive ECE bin count")
    probabilities = torch.softmax(logits.float(), dim=-1)
    predictions = probabilities.argmax(-1)
    correct = predictions.eq(targets)
    confidence = probabilities.max(-1).values
    ece, bin_counts = 0.0, []
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
    return {
        "count": int(targets.numel()),
        "accuracy": float(correct.float().mean()),
        "nll": float(F.cross_entropy(logits.float(), targets)),
        "brier": float((probabilities - F.one_hot(targets, num_classes=4)).square().sum(-1).mean()),
        "ece_15_bins": ece,
        "ece_bin_counts": bin_counts,
        "mean_confidence": float(confidence.mean()),
    }


def _predict(
    scorer: nn.Module,
    features: np.ndarray,
    targets: Tensor,
    device: torch.device,
    batch_questions: int = 512,
) -> tuple[Tensor, dict[str, Any]]:
    scorer.eval()
    chunks: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_questions):
            end = min(start + batch_questions, len(features))
            batch = torch.from_numpy(np.asarray(features[start:end], dtype=np.float32)).to(device)
            chunks.append(scorer(batch).float().cpu())
    logits = torch.cat(chunks)
    return logits, _metrics(logits, targets, bins=15)


def _fit_readout(
    train_features: np.ndarray,
    train_rows: list[dict[str, Any]],
    validation_features: np.ndarray,
    validation_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_questions: int,
    learning_rate: float,
    weight_decay: float,
    min_delta: float,
) -> tuple[CandidateScorer, CandidateScorer, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    hidden_size = int(train_features.shape[-1])
    scorer = CandidateScorer(hidden_size).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_targets = torch.tensor(
        ["abcd".index(str(row["label"])) for row in train_rows], dtype=torch.long
    )
    validation_targets = torch.tensor(
        ["abcd".index(str(row["label"])) for row in validation_rows], dtype=torch.long
    )
    order_rng = random.Random(seed)
    best_nll, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    terminal_state: dict[str, Tensor] | None = None
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_rows)))
        order_rng.shuffle(order)
        scorer.train()
        loss_sum, seen = 0.0, 0
        for start in range(0, len(order), batch_questions):
            indices = order[start : start + batch_questions]
            batch = torch.from_numpy(np.asarray(train_features[indices], dtype=np.float32)).to(
                device
            )
            targets = train_targets[indices].to(device)
            logits = scorer(batch)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite readout loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            seen += len(indices)
        scorer.eval()
        validation_logits, validation_metrics = _predict(
            scorer, validation_features, validation_targets, device
        )
        validation_nll = float(validation_metrics["nll"])
        epoch_record = {
            "epoch": epoch,
            "optimizer_updates": (len(train_rows) + batch_questions - 1) // batch_questions,
            "cumulative_optimizer_updates": epoch
            * ((len(train_rows) + batch_questions - 1) // batch_questions),
            "rolling_train_ce": loss_sum / seen,
            "validation": validation_metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        terminal_state = {
            key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
        }
        if validation_nll < best_nll - min_delta:
            best_nll, best_epoch = validation_nll, epoch
            best_state = terminal_state
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or terminal_state is None:
        raise RuntimeError("validation failed to select a scorer checkpoint")
    best_scorer, final_scorer = CandidateScorer(hidden_size), CandidateScorer(hidden_size)
    best_scorer.load_state_dict(best_state)
    final_scorer.load_state_dict(terminal_state)
    best_scorer.to(device).eval()
    final_scorer.to(device).eval()
    return (
        best_scorer,
        final_scorer,
        {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "early_stopped": len(history) < epochs,
            "selection_metric": "validation_nll",
            "selection_min_delta": min_delta,
            "history": history,
            "readout_trainable_parameters": sum(p.numel() for p in best_scorer.parameters()),
            "fit_elapsed_seconds": time.perf_counter() - started,
        },
    )


def _stratified_metrics(logits: Tensor, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    targets = torch.tensor(["abcd".index(str(row["label"])) for row in rows])
    for agreement in sorted({int(row["agreement"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if int(row["agreement"]) == agreement]
        output[str(agreement)] = _metrics(logits[indices], targets[indices])
    return output


def _save_predictions(path: Path, rows: list[dict[str, Any]], logits: Tensor) -> None:
    probabilities = torch.softmax(logits.float(), dim=-1).tolist()
    predictions = logits.argmax(-1).tolist()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, predicted, probs in zip(rows, predictions, probabilities, strict=True):
            record = {
                "id": row["id"],
                "target": row["label"],
                "prediction": "abcd"[predicted],
                "probabilities_abcd": probs,
                "agreement": int(row["agreement"]),
                "core_event_pair_sha256": hashlib.sha256(
                    " ".join(str(row["core_event_pair"]).split()).encode("utf-8")
                ).hexdigest(),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _assert_option_order_equivariance(
    scorer: nn.Module, validation_features: np.ndarray, device: torch.device
) -> float:
    indices = list(range(min(128, len(validation_features))))
    batch = torch.from_numpy(np.asarray(validation_features[indices], dtype=np.float32)).to(device)
    permutation = [2, 0, 3, 1]
    with torch.inference_mode():
        original = scorer(batch)
        reordered = scorer(batch[:, permutation])
    delta = float((reordered - original[:, permutation]).abs().max())
    if delta > 1e-6:
        raise ValueError("candidate-order equivariance check failed")
    return delta


def _run_candidate(
    name: str,
    model_dir: Path,
    line_layers: int,
    revision: str,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    output_root: Path,
    device: torch.device,
    config: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    output_dir = output_root / name
    output_dir.mkdir(parents=False, exist_ok=False)
    train_feature_path = output_dir / "train-features.npy"
    validation_feature_path = output_dir / "validation-features.npy"
    encode_started = time.perf_counter()
    encoder_metadata = encode_features(
        model_dir,
        train_rows,
        train_feature_path,
        device=device,
        line_layers=line_layers,
        batch_questions=config["encode_batch_questions"],
        max_length=config["max_length"],
    )
    validation_encoder_metadata = encode_features(
        model_dir,
        validation_rows,
        validation_feature_path,
        device=device,
        line_layers=line_layers,
        batch_questions=config["encode_batch_questions"],
        max_length=config["max_length"],
    )
    train_features = np.load(train_feature_path, mmap_mode="r")
    validation_features = np.load(validation_feature_path, mmap_mode="r")
    best, final, fit = _fit_readout(
        train_features,
        train_rows,
        validation_features,
        validation_rows,
        device=device,
        seed=config["seed"],
        epochs=config["epochs"],
        patience=config["patience"],
        batch_questions=config["batch_questions"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        min_delta=config["selection_min_delta"],
    )
    best_path, final_path = output_dir / "best-readout.pt", output_dir / "final-readout.pt"
    torch.save(_cpu_state_dict(best), best_path)
    torch.save(_cpu_state_dict(final), final_path)
    selected_state = torch.load(best_path, map_location="cpu", weights_only=True)
    reloaded = CandidateScorer(int(train_features.shape[-1]))
    reloaded.load_state_dict(selected_state)
    reloaded.to(device).eval()
    best_logits, best_metrics = _predict(
        reloaded,
        validation_features,
        torch.tensor(["abcd".index(str(row["label"])) for row in validation_rows]),
        device,
    )
    final_logits, final_metrics = _predict(
        final,
        validation_features,
        torch.tensor(["abcd".index(str(row["label"])) for row in validation_rows]),
        device,
    )
    parity = _assert_option_order_equivariance(reloaded, validation_features, device)
    prediction_path = output_dir / "best-validation-predictions.jsonl"
    _save_predictions(prediction_path, validation_rows, best_logits)

    total_parameters = encoder_metadata["encoder_parameters"] + fit["readout_trainable_parameters"]
    report = {
        "candidate": name,
        "model_revision": revision,
        "model_files_sha256": encoder_metadata["weight_files_sha256"],
        "encoder": encoder_metadata,
        "validation_encoder": validation_encoder_metadata,
        "peak_allocated_vram_bytes": max(
            encoder_metadata["peak_allocated_vram_bytes"],
            validation_encoder_metadata["peak_allocated_vram_bytes"],
        ),
        "trainable_encoder_parameters": 0,
        "readout_trainable_parameters": fit["readout_trainable_parameters"],
        "text_model_total_parameters": total_parameters,
        "four_tower_total_if_replacing_ruri": FOUR_TOWER_BASE_PARAMETERS
        - RURI_PARAMETERS
        + total_parameters,
        "training_examples": len(train_rows),
        "training_rows_consumed": len(train_rows) * fit["epochs_completed"],
        "selected_checkpoint_training_rows": len(train_rows) * fit["best_epoch"],
        "validation_examples": len(validation_rows),
        "training_unique_event_pairs": len({str(row["core_event_pair"]) for row in train_rows}),
        "validation_unique_event_pairs": len(
            {str(row["core_event_pair"]) for row in validation_rows}
        ),
        "selected_checkpoint": {"epoch": fit["best_epoch"], "metrics": best_metrics},
        "terminal_checkpoint": {"epoch": fit["epochs_completed"], "metrics": final_metrics},
        "selected_by_validation_only": True,
        "validation_by_agreement": _stratified_metrics(best_logits, validation_rows),
        "option_order_max_logit_delta": parity,
        "fit": fit,
        "total_encoder_and_readout_wall_seconds": time.perf_counter() - encode_started,
        "best_readout_sha256": sha256_file(best_path),
        "final_readout_sha256": sha256_file(final_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "smoke_only": smoke,
        "sealed_audit_loaded": False,
        "official_publisher_dev_or_test_loaded_by_training_run": False,
        "product_promotion": False,
    }
    write_json(output_dir / "run-report.json", report)
    del train_features, validation_features, best, final, reloaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ruri-model-dir", type=Path, required=True)
    parser.add_argument("--line-distilbert-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-questions", type=int, default=256)
    parser.add_argument("--encode-batch-questions", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--selection-min-delta", type=float, default=0.001)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    repo_root = Path(__file__).resolve().parents[1]
    try:
        args.output_dir.resolve().relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise ValueError("large model/run artifacts must stay outside the Git worktree")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    train_path, validation_path = args.data_dir / "train.jsonl", args.data_dir / "validation.jsonl"
    split_manifest_path = args.data_dir / "split-manifest.json"
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if manifest["source_revision"] != "79afa1d63d1f262e6709e9be714d9ad5dae8d6f6":
        raise ValueError("unexpected KUCI source revision")
    if sha256_file(train_path) != manifest["train_jsonl_sha256"]:
        raise ValueError("frozen training split hash mismatch")
    if sha256_file(validation_path) != manifest["validation_jsonl_sha256"]:
        raise ValueError("frozen validation split hash mismatch")
    train_rows, validation_rows = read_jsonl(train_path), read_jsonl(validation_path)
    _assert_disjoint(train_rows, validation_rows)
    if {str(row["id"]) for row in train_rows} != set(manifest["train_ids"]):
        raise ValueError("training IDs do not match the split manifest")
    if {str(row["id"]) for row in validation_rows} != set(manifest["validation_ids"]):
        raise ValueError("validation IDs do not match the split manifest")
    if args.smoke_rows:
        train_rows = train_rows[: args.smoke_rows]
        validation_rows = validation_rows[: max(4, args.smoke_rows // 2)]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "experiment_id": "kuci-bert-temporal-probe-v1",
        "seed": args.seed,
        "epochs": args.epochs if not args.smoke_rows else 1,
        "patience": args.patience,
        "batch_questions": args.batch_questions,
        "encode_batch_questions": args.encode_batch_questions,
        "max_length": args.max_length,
        "selection_metric": "validation_nll",
        "selection_min_delta": args.selection_min_delta,
        "optimizer": "AdamW",
        "learning_rate": 0.002,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "readout": "hidden-size-128-1 tanh shared option scorer",
        "pooling": "attention-mask-aware mean over attended tokens for both encoders",
        "device": str(device),
        "smoke_only": bool(args.smoke_rows),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in (vars(args) | {"output_dir": output_dir}).items()
        },
        "source_train_sha256": manifest["source_train_sha256"],
        "frozen_train_sha256": manifest["train_jsonl_sha256"],
        "frozen_validation_sha256": manifest["validation_jsonl_sha256"],
        "split_assignment_sha256": manifest["sample_assignment_sha256"],
        "split_seed": manifest["split_seed"],
        "source_license": "CC-BY-SA-4.0",
        "cost_usd": 0,
    }
    runner_hash = sha256_file(Path(__file__))
    config["runner_sha256"] = runner_hash
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    config["effective_config_sha256"] = config_hash
    write_json(output_dir / "effective-config.json", config)
    repo_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = datetime.now(UTC)
    reports = []
    candidates = [
        ("ruri-v3-30m", args.ruri_model_dir, 0, RURI_REVISION),
        ("line-distilbert-ja-first4", args.line_distilbert_dir, 4, LINE_REVISION),
    ]
    for name, path, layers, revision in candidates:
        expected_sha = RURI_WEIGHT_SHA256 if layers == 0 else LINE_WEIGHT_SHA256
        weight_name = "model.safetensors" if layers == 0 else "pytorch_model.bin"
        if sha256_file(path / weight_name) != expected_sha:
            raise ValueError(f"pinned encoder weight hash mismatch: {name}")
        reports.append(
            _run_candidate(
                name,
                path,
                layers,
                revision,
                train_rows,
                validation_rows,
                output_root=output_dir,
                device=device,
                config=config,
                smoke=bool(args.smoke_rows),
            )
        )
    result = {
        "experiment_id": config["experiment_id"],
        "source_commit": repo_commit,
        "runner_sha256": runner_hash,
        "effective_config_sha256": config_hash,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "training_rows_consumed_across_candidates": sum(
            int(item["training_rows_consumed"]) for item in reports
        ),
        "unique_train_examples_per_candidate": len(train_rows),
        "validation_examples": len(validation_rows),
        "unique_validation_examples": len(validation_rows),
        "source_train_rows": manifest["source_rows"],
        "split_manifest": {
            key: value
            for key, value in manifest.items()
            if key not in {"train_ids", "validation_ids"}
        },
        "device": str(device),
        "peak_allocated_vram_bytes": max(
            int(item["peak_allocated_vram_bytes"]) for item in reports
        ),
        "cost_usd": 0,
        "candidate_reports": reports,
        "evaluation_limitations": [
            "development-only train-derived validation; not a blind test",
            "official KUCI dev was source-screened and is not used for selection",
            "official KUCI test was not loaded",
            "CC BY-SA trained readouts are retained locally and are not product-promoted",
        ],
        "smoke_only": bool(args.smoke_rows),
        "sealed_audit_loaded": False,
        "product_promotion": False,
    }
    write_json(output_dir / "comparison-report.json", result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "candidate_reports"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
