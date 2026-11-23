"""Test whether adapting the top half of a compact Japanese DistilBERT helps Jamp."""

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
    from scripts.pretrained_reuse.build_jamp_grouped_split import LABELS
    from scripts.pretrained_reuse.train_jamp_frozen_text_readout import (
        LABEL_TO_ID,
        LINE_REVISION,
        LINE_WEIGHT_SHA256,
        Readout,
        metrics,
        read_tsv,
        sha256_file,
        write_json,
    )
except ModuleNotFoundError:
    from build_jamp_grouped_split import LABELS
    from train_jamp_frozen_text_readout import (
        LABEL_TO_ID,
        LINE_REVISION,
        LINE_WEIGHT_SHA256,
        Readout,
        metrics,
        read_tsv,
        sha256_file,
        write_json,
    )


def freeze_except_top_layers(model: nn.Module, *, trainable_layers: int = 2) -> list[str]:
    layer_count = int(getattr(model.config, "n_layers", 0))
    if model.config.model_type != "distilbert" or layer_count < 2:
        raise ValueError("only DistilBERT with at least two layers is supported")
    if trainable_layers < 1 or trainable_layers >= layer_count:
        raise ValueError("trainable_layers must be positive and less than the layer count")
    first_trainable = layer_count - trainable_layers
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        layer_index = None
        for index, part in enumerate(parts[:-1]):
            if part == "layer" and parts[index + 1].isdigit():
                layer_index = int(parts[index + 1])
                break
        parameter.requires_grad_(layer_index is not None and layer_index >= first_trainable)
        if parameter.requires_grad:
            trainable.append(name)
    expected_prefix = f"transformer.layer.{first_trainable}."
    if not any(name.startswith(expected_prefix) for name in trainable):
        raise ValueError("could not find expected DistilBERT transformer layer names")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != any(
            name.startswith(f"transformer.layer.{layer}.")
            for layer in range(first_trainable, layer_count)
        ):
            raise ValueError(f"unexpected trainability assignment: {name}")
    return trainable


def _frozen_weights_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tokenize(
    tokenizer: Any, rows: list[dict[str, str]], *, max_length: int
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    encoded = tokenizer(
        [row["premise"] for row in rows],
        [row["hypothesis"] for row in rows],
        truncation=False,
        padding=False,
    )
    lengths = [len(tokens) for tokens in encoded["input_ids"]]
    truncated = sum(length > max_length for length in lengths)
    if truncated:
        raise ValueError(f"refusing to truncate {truncated} premise/hypothesis pairs")
    result = [
        {key: encoded[key][index] for key in tokenizer.model_input_names if key in encoded}
        for index in range(len(rows))
    ]
    return result, {"max_pair_tokens": max(lengths, default=0), "truncated_pairs": truncated}


def _batch_inputs(tokenizer: Any, rows: list[dict[str, list[int]]], device: torch.device) -> Any:
    return tokenizer.pad(rows, padding=True, return_tensors="pt").to(device)


def _encode_logits(
    model: nn.Module,
    readout: nn.Module,
    rows: list[dict[str, list[int]]],
    *,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    model.eval()
    readout.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = _batch_inputs(tokenizer, batch_rows, device)
            hidden = model(**batch).last_hidden_state
            weights = batch["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            logits.append(readout(pooled).float().cpu())
    return torch.cat(logits)


def _save_predictions(path: Path, source_rows: list[dict[str, str]], logits: Tensor) -> None:
    probs = torch.softmax(logits.float(), dim=-1).tolist()
    preds = logits.argmax(-1).tolist()
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row, predicted, probabilities in zip(source_rows, preds, probs, strict=True):
            file.write(
                json.dumps(
                    {
                        "id": row["num"],
                        "template_num": row["template_num"],
                        "category": row["category"],
                        "target": row["gold_label"],
                        "prediction": LABELS[predicted],
                        "probabilities": dict(zip(LABELS, probabilities, strict=True)),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _metrics_by_category(source_rows: list[dict[str, str]], logits: Tensor) -> dict[str, Any]:
    result = {}
    targets = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in source_rows])
    for category in sorted({row["category"] for row in source_rows}):
        indices = [index for index, row in enumerate(source_rows) if row["category"] == category]
        result[category] = metrics(logits[indices], targets[indices])
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    split_manifest_path = args.data_dir / "split-manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    train_path, validation_path = args.data_dir / "train.tsv", args.data_dir / "validation.tsv"
    if sha256_file(train_path) != split_manifest["train_sha256"]:
        raise ValueError("frozen training data hash mismatch")
    if sha256_file(validation_path) != split_manifest["validation_sha256"]:
        raise ValueError("frozen validation data hash mismatch")
    train_rows, validation_rows = read_tsv(train_path), read_tsv(validation_path)
    if args.smoke_rows:
        train_rows = train_rows[: args.smoke_rows]
        validation_rows = validation_rows[: args.smoke_rows]
    model_hash = sha256_file(args.model_dir / "pytorch_model.bin")
    if model_hash != LINE_WEIGHT_SHA256:
        raise ValueError("DistilBERT model weight hash mismatch")

    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    if config.model_type != "distilbert" or int(config.n_layers) != 6:
        raise ValueError("expected the pinned six-layer LINE DistilBERT checkpoint")
    if args.encoder_layers != 4:
        raise ValueError("this controlled adaptation run uses exactly the first four layers")
    config.n_layers = args.encoder_layers
    if args.max_length > int(config.max_position_embeddings):
        raise ValueError("max_length exceeds the pinned DistilBERT positional limit")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=True,
    )
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    encoder, loading_info = AutoModel.from_pretrained(
        args.model_dir,
        config=config,
        local_files_only=True,
        output_loading_info=True,
        trust_remote_code=True,
        use_safetensors=False,
    )
    missing_keys = sorted(str(key) for key in loading_info["missing_keys"])
    unexpected_keys = sorted(str(key) for key in loading_info["unexpected_keys"])
    if missing_keys:
        raise ValueError(f"checkpoint has missing DistilBERT keys: {missing_keys[:8]}")
    dropped_layers = sorted(
        {
            layer
            for layer in range(args.encoder_layers, 6)
            if any(f"layer.{layer}." in key for key in unexpected_keys)
        }
    )
    pretraining_heads = (
        "vocab_transform.",
        "vocab_layer_norm.",
        "vocab_projector.",
    )
    unrecognized = [
        key
        for key in unexpected_keys
        if not any(marker in key for marker in ("transformer.layer.4.", "transformer.layer.5."))
        and not key.startswith(pretraining_heads)
        and not any(
            marker in key
            for marker in ("vocab_transform.", "vocab_layer_norm.", "vocab_projector.")
        )
    ]
    if dropped_layers != [4, 5] or unrecognized:
        raise ValueError(
            "expected only pretrained DistilBERT layers 4/5 and MLM-head keys to be dropped; "
            f"layers={dropped_layers}, extra={unrecognized[:8]}"
        )
    encoder.to(device)
    encoder.train()
    trainable_names = freeze_except_top_layers(encoder, trainable_layers=args.trainable_layers)
    readout = Readout(int(config.hidden_size)).to(device)
    frozen_hash_before = _frozen_weights_hash(encoder)
    train_tokens, train_token_stats = _tokenize(tokenizer, train_rows, max_length=args.max_length)
    validation_tokens, validation_token_stats = _tokenize(
        tokenizer, validation_rows, max_length=args.max_length
    )
    train_targets = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in train_rows])
    validation_targets = torch.tensor([LABEL_TO_ID[row["gold_label"]] for row in validation_rows])
    trainable_encoder_parameters = [
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    ]
    readout_parameters = list(readout.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": readout_parameters, "lr": args.readout_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    effective_epochs = min(args.epochs, 2) if args.smoke_rows else args.epochs
    updates_per_epoch = (len(train_rows) + args.batch_size - 1) // args.batch_size
    total_updates = updates_per_epoch * effective_epochs
    warmup_updates = max(1, round(total_updates * args.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup_updates))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
    order_rng = random.Random(args.seed)
    best_nll, best_epoch, best_state = float("inf"), 0, None
    history: list[dict[str, Any]] = []
    runner_hash = sha256_file(Path(__file__))
    effective_config = {
        "experiment_id": "jamp-line-distilbert-first4-top2-adaptation-v1",
        "seed": args.seed,
        "epochs": effective_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "trainable_layers": args.trainable_layers,
        "encoder_learning_rate": args.encoder_learning_rate,
        "readout_learning_rate": args.readout_learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_updates": warmup_updates,
        "total_updates": total_updates,
        "selection_metric": "validation_nll",
        "selection_min_delta": args.selection_min_delta,
        "early_stopping": False,
        "model_revision": LINE_REVISION,
        "model_weight_sha256": model_hash,
        "encoder_layers": args.encoder_layers,
        "dropped_pretrained_layers": dropped_layers,
        "source_train_sha256": split_manifest["source_file_sha256"],
        "train_sha256": split_manifest["train_sha256"],
        "validation_sha256": split_manifest["validation_sha256"],
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "runner_sha256": runner_hash,
        "smoke_only": bool(args.smoke_rows),
    }
    effective_config_sha256 = hashlib.sha256(
        json.dumps(effective_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    effective_config["effective_config_sha256"] = effective_config_sha256
    write_json(args.output_dir / "effective-config.json", effective_config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, effective_epochs + 1):
        epoch_start = time.perf_counter()
        order = list(range(len(train_rows)))
        order_rng.shuffle(order)
        encoder.train()
        readout.train()
        loss_sum, seen, grad_norm_max = 0.0, 0, 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch = _batch_inputs(tokenizer, [train_tokens[index] for index in indices], device)
            targets = train_targets[indices].to(device)
            hidden = encoder(**batch).last_hidden_state
            weights = batch["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            logits = readout(pooled)
            loss = F.cross_entropy(logits.float(), targets)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at epoch={epoch}, row={start}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [*trainable_encoder_parameters, *readout_parameters], max_norm=1.0
            )
            if not torch.isfinite(gradient_norm):
                raise ValueError(f"non-finite gradient norm at epoch={epoch}, row={start}")
            grad_norm_max = max(grad_norm_max, float(gradient_norm.detach()))
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(indices)
            seen += len(indices)

        train_logits = _encode_logits(
            encoder,
            readout,
            train_tokens,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
        )
        validation_logits = _encode_logits(
            encoder,
            readout,
            validation_tokens,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.eval_batch_size,
        )
        train_metrics = metrics(train_logits, train_targets)
        validation_metrics = metrics(validation_logits, validation_targets)
        history.append(
            {
                "epoch": epoch,
                "optimizer_updates": updates_per_epoch,
                "cumulative_optimizer_updates": epoch * updates_per_epoch,
                "rolling_train_ce": loss_sum / seen,
                "fixed_train": train_metrics,
                "validation": validation_metrics,
                "encoder_lr_end": optimizer.param_groups[0]["lr"],
                "readout_lr_end": optimizer.param_groups[1]["lr"],
                "max_gradient_norm_before_clip": grad_norm_max,
                "elapsed_seconds": time.perf_counter() - epoch_start,
            }
        )
        current_nll = float(validation_metrics["nll"])
        if current_nll < best_nll - args.selection_min_delta:
            best_nll, best_epoch = current_nll, epoch
            best_state = {
                "encoder": {
                    key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()
                },
                "readout": {
                    key: value.detach().cpu().clone() for key, value in readout.state_dict().items()
                },
            }
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_ce": round(loss_sum / seen, 4),
                    "train_acc": round(train_metrics["accuracy"], 4),
                    "validation_nll": round(current_nll, 4),
                    "validation_acc": round(validation_metrics["accuracy"], 4),
                    "validation_brier": round(validation_metrics["brier"], 4),
                    "elapsed_seconds": round(time.perf_counter() - epoch_start, 2),
                }
            ),
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("validation did not select a checkpoint")
    final_state = {
        "encoder": {
            key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()
        },
        "readout": {
            key: value.detach().cpu().clone() for key, value in readout.state_dict().items()
        },
    }
    best_path, final_path = (
        args.output_dir / "best-checkpoint.pt",
        args.output_dir / "final-checkpoint.pt",
    )
    torch.save(best_state, best_path)
    torch.save(final_state, final_path)
    encoder.load_state_dict(best_state["encoder"])
    readout.load_state_dict(best_state["readout"])
    best_logits = _encode_logits(
        encoder,
        readout,
        validation_tokens,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.eval_batch_size,
    )
    best_metrics = metrics(best_logits, validation_targets)
    if abs(best_metrics["nll"] - best_nll) > 1e-6:
        raise ValueError("best checkpoint reload changed validation NLL")
    frozen_hash_after = _frozen_weights_hash(encoder)
    if frozen_hash_before != frozen_hash_after:
        raise ValueError("frozen embeddings/lower layers changed during adaptation")
    _save_predictions(
        args.output_dir / "best-validation-predictions.jsonl", validation_rows, best_logits
    )
    if device.type == "cuda":
        peak_vram = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_vram = 0
    report = {
        "experiment_id": "jamp-line-distilbert-first4-top2-adaptation-v1",
        "source_commit": subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "source_dataset_revision": split_manifest["source_revision"],
        "source_train_sha256": split_manifest["source_file_sha256"],
        "train_sha256": split_manifest["train_sha256"],
        "validation_sha256": split_manifest["validation_sha256"],
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "model": {
            "id": "line-corporation/line-distilbert-base-japanese",
            "revision": LINE_REVISION,
            "weight_sha256": model_hash,
            "layers": int(config.n_layers),
            "dropped_pretrained_layers": dropped_layers,
            "trainable_layer_count": args.trainable_layers,
            "trainable_parameter_names": trainable_names,
            "trainable_encoder_parameters": sum(
                parameter.numel() for parameter in trainable_encoder_parameters
            ),
            "frozen_encoder_parameters": sum(
                parameter.numel()
                for parameter in encoder.parameters()
                if not parameter.requires_grad
            ),
            "readout_trainable_parameters": sum(
                parameter.numel() for parameter in readout.parameters()
            ),
            "total_text_parameters": sum(parameter.numel() for parameter in encoder.parameters())
            + sum(parameter.numel() for parameter in readout.parameters()),
            "four_tower_parameters_if_replacing_ruri": 178_689_152
            - 36_705_536
            + sum(parameter.numel() for parameter in encoder.parameters())
            + sum(parameter.numel() for parameter in readout.parameters()),
            "frozen_weights_sha256_before": frozen_hash_before,
            "frozen_weights_sha256_after": frozen_hash_after,
        },
        "data": {
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "train_unique_template_groups": len({row["template_num"] for row in train_rows}),
            "validation_unique_template_groups": len(
                {row["template_num"] for row in validation_rows}
            ),
            "train_consumed_rows": len(train_rows) * effective_epochs,
            "best_checkpoint_consumed_rows": len(train_rows) * best_epoch,
            "train_token_stats": train_token_stats,
            "validation_token_stats": validation_token_stats,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_allocated_vram_bytes": peak_vram,
        },
        "config": {
            "epochs": effective_epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_length": args.max_length,
            "encoder_learning_rate": args.encoder_learning_rate,
            "readout_learning_rate": args.readout_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_updates": warmup_updates,
            "total_updates": total_updates,
            "selection_metric": "validation_nll",
            "selection_min_delta": args.selection_min_delta,
            "early_stopping": False,
        },
        "best_checkpoint": {"epoch": best_epoch, "metrics": best_metrics},
        "terminal_checkpoint": {"epoch": effective_epochs, "metrics": history[-1]["validation"]},
        "learning_curve": history,
        "best_checkpoint_sha256": sha256_file(best_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "validation_predictions_sha256": sha256_file(
            args.output_dir / "best-validation-predictions.jsonl"
        ),
        "wall_seconds": time.perf_counter() - started,
        "cost_usd": 0,
        "smoke_only": bool(args.smoke_rows),
        "official_test_loaded": False,
        "product_promotion": False,
    }
    report["runner_sha256"] = runner_hash
    report["effective_config_sha256"] = effective_config_sha256
    write_json(args.output_dir / "run-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--trainable-layers", type=int, default=2)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--readout-learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--selection-min-delta", type=float, default=0.001)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "experiment_id": report["experiment_id"],
                "best_epoch": report["best_checkpoint"]["epoch"],
                "best_validation": report["best_checkpoint"]["metrics"],
                "terminal_validation": report["terminal_checkpoint"],
                "train_consumed_rows": report["data"]["train_consumed_rows"],
                "peak_vram_bytes": report["environment"]["peak_allocated_vram_bytes"],
                "wall_seconds": report["wall_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
