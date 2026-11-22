"""Train a Banking77 readout while adapting only MiniLM-L3's final block."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from scripts.train_banking77_minilm_readout import (
    CLASS_COUNT,
    MODEL_REVISION,
    MODEL_SHA256,
    load_and_validate_data,
    metric_summary,
    sha256_file,
)


def configure_last_block(encoder: nn.Module) -> list[nn.Parameter]:
    """Freeze the encoder except its final Transformer block; return trainables."""
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if not layers:
        raise ValueError("Expected a BERT-family encoder exposing encoder.layer")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in layers[-1].parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Final encoder block has no trainable parameters")
    return trainable


def tokenize_rows(rows: list[dict[str, Any]], model_dir: Path) -> list[dict[str, torch.Tensor]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id != 0:
        raise ValueError(f"Unexpected MiniLM padding token id: {tokenizer.pad_token_id}")
    encoded = tokenizer(
        [str(row["text"]) for row in rows],
        truncation=True,
        max_length=128,
        return_tensors="pt",
        padding=True,
    )
    # Re-trim each item to its actual token length; batches are padded dynamically below.
    result: list[dict[str, torch.Tensor]] = []
    for item in range(len(rows)):
        length = int(encoded["attention_mask"][item].sum())
        result.append({key: value[item, :length].cpu() for key, value in encoded.items()})
    return result


def collate_dynamic(
    examples: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    pad_id = 0
    keys = examples[0][0]
    inputs: dict[str, torch.Tensor] = {}
    for key in keys:
        padding_value = pad_id if key == "input_ids" else 0
        inputs[key] = nn.utils.rnn.pad_sequence(
            [item[key] for item, _ in examples],
            batch_first=True,
            padding_value=padding_value,
        )
    return inputs, torch.stack([label for _, label in examples])


def iterate_batches(
    tokens: list[dict[str, torch.Tensor]], labels: torch.Tensor, batch_size: int
):
    for start in range(0, len(labels), batch_size):
        examples = [
            (tokens[index], labels[index])
            for index in range(start, min(start + batch_size, len(labels)))
        ]
        yield collate_dynamic(examples)


def pool_last_hidden(output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden = output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def train_last_block(
    encoder: nn.Module,
    train_tokens: list[dict[str, torch.Tensor]],
    train_labels: torch.Tensor,
    validation_tokens: list[dict[str, torch.Tensor]],
    validation_labels: torch.Tensor,
    *,
    seed: int = 17,
    encoder_learning_rate: float = 2e-5,
    head_learning_rate: float = 0.01,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    max_epochs: int = 12,
    patience: int = 4,
    min_delta: float = 1e-5,
) -> tuple[nn.Module, dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    encoder = encoder.cpu()
    encoder_parameters = configure_last_block(encoder)
    encoder.eval()
    with torch.no_grad():
        initial_features: list[torch.Tensor] = []
        for batch, _ in iterate_batches(train_tokens, train_labels, batch_size):
            initial_features.append(pool_last_hidden(encoder(**batch), batch["attention_mask"]))
        initial_train_features = torch.cat(initial_features)
    feature_mean = initial_train_features.mean(dim=0)
    feature_std = initial_train_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    head = nn.Linear(initial_train_features.shape[-1], CLASS_COUNT)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": encoder_learning_rate},
            {"params": head.parameters(), "lr": head_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.arange(len(train_labels)), train_labels)

    def collate_indices(batch):
        return collate_dynamic([(train_tokens[int(index)], label) for index, label in batch])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    loader.collate_fn = collate_indices
    history: list[dict[str, Any]] = []
    best_nll = float("inf")
    best_epoch = 0
    best_encoder: dict[str, torch.Tensor] | None = None
    best_head: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        encoder.train()
        head.train()
        losses: list[float] = []
        for batch in loader:
            inputs, labels = batch
            optimizer.zero_grad(set_to_none=True)
            pooled = pool_last_hidden(encoder(**inputs), inputs["attention_mask"])
            logits = head((pooled - feature_mean) / feature_std)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in (*encoder_parameters, *head.parameters())
            ):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            losses.append(loss.item())
        encoder.eval()
        head.eval()
        val_logits: list[torch.Tensor] = []
        with torch.inference_mode():
            for inputs, _ in iterate_batches(validation_tokens, validation_labels, batch_size):
                pooled = pool_last_hidden(encoder(**inputs), inputs["attention_mask"])
                val_logits.append(head((pooled - feature_mean) / feature_std))
            current_logits = torch.cat(val_logits)
            metrics = metric_summary(current_logits, validation_labels)
        history.append(
            {"epoch": epoch, "train_ce": sum(losses) / len(losses), "validation": metrics}
        )
        print(
            f"epoch={epoch} train_ce={history[-1]['train_ce']:.6f} "
            f"validation_nll={metrics['nll']:.6f} validation_accuracy={metrics['accuracy']:.6f}",
            flush=True,
        )
        if metrics["nll"] < best_nll - min_delta:
            best_nll = metrics["nll"]
            best_epoch = epoch
            best_encoder = {
                key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()
            }
            best_head = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_encoder is None or best_head is None or best_metrics is None:
        raise RuntimeError("No validation-selected checkpoint was produced")
    encoder.load_state_dict(best_encoder)
    head.load_state_dict(best_head)
    encoder.eval()
    head.eval()
    return head, {
        "encoder_state_dict": best_encoder,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "epochs_completed": len(history),
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "trainable_encoder_parameters": sum(parameter.numel() for parameter in encoder_parameters),
        "encoder": encoder,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    model_weights = args.model_dir / "model.safetensors"
    if sha256_file(model_weights) != MODEL_SHA256:
        raise ValueError(f"MiniLM weights mismatch (expected revision {MODEL_REVISION})")
    train_rows, validation_rows, split_manifest = load_and_validate_data(args.data_dir)
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    train_tokens = tokenize_rows(train_rows, args.model_dir)
    validation_tokens = tokenize_rows(validation_rows, args.model_dir)
    labels = torch.tensor([int(row["label"]) for row in train_rows], dtype=torch.long)
    val_labels = torch.tensor([int(row["label"]) for row in validation_rows], dtype=torch.long)
    encoder = AutoModel.from_pretrained(args.model_dir, local_files_only=True).cpu()
    head, fit = train_last_block(encoder, train_tokens, labels, validation_tokens, val_labels)
    encoder = fit.pop("encoder")
    with torch.inference_mode():
        logits: list[torch.Tensor] = []
        for inputs, _ in iterate_batches(validation_tokens, val_labels, 32):
            pooled = pool_last_hidden(encoder(**inputs), inputs["attention_mask"])
            logits.append(head((pooled - fit["feature_mean"]) / fit["feature_std"]))
        validation_logits = torch.cat(logits)
        probabilities = torch.softmax(validation_logits, dim=-1)
    args.output_dir.mkdir(parents=True)
    checkpoint_path = args.output_dir / "best-last-block.pt"
    torch.save(
        {
            "encoder_state_dict": fit["encoder_state_dict"],
            "head_state_dict": head.state_dict(),
            "feature_mean": fit["feature_mean"],
            "feature_std": fit["feature_std"],
            "best_epoch": fit["best_epoch"],
        },
        checkpoint_path,
    )
    history_path = args.output_dir / "training-history.jsonl"
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in fit["history"]), encoding="utf-8"
    )
    predictions_path = args.output_dir / "validation-predictions.jsonl"
    prediction_ids = validation_rows
    with predictions_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(prediction_ids):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "target": int(val_labels[index]),
                        "prediction": int(validation_logits[index].argmax()),
                        "probabilities": probabilities[index].tolist(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    metrics_path = args.output_dir / "validation-metrics.json"
    metrics = metric_summary(validation_logits, val_labels)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "experiment_id": "banking77-minilm-l3-last-block-v1",
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": sha256_file(model_weights),
        "split_manifest_sha256": hashlib.sha256(
            (args.data_dir / "split-manifest.json").read_bytes()
        ).hexdigest(),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "base_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "trainable_encoder_parameters": fit["trainable_encoder_parameters"],
        "trainable_head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "trainable_parameters_total": fit["trainable_encoder_parameters"]
        + sum(parameter.numel() for parameter in head.parameters()),
        "seed": 17,
        "device": "cpu",
        "best_epoch": fit["best_epoch"],
        "epochs_completed": fit["epochs_completed"],
        "best_validation_metrics": fit["best_validation_metrics"],
        "validation_metrics": metrics,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history_sha256": sha256_file(history_path),
        "predictions_sha256": sha256_file(predictions_path),
        "official_test_loaded": False,
        "split_format": split_manifest["format"],
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
