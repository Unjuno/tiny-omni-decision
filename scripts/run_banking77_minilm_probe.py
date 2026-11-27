"""Fit a frozen MiniLM option scorer on the pinned BANKING77 train-only split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

from scripts.audit_banking77_train_quality import EXPECTED_SHA256

MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
MODEL_WEIGHT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pool_features(
    texts: list[str], tokenizer: Any, encoder: torch.nn.Module, *, batch_size: int
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            outputs.append(pooled.cpu())
    return torch.cat(outputs, dim=0)


def _metrics(logits: torch.Tensor, targets: torch.Tensor, class_count: int) -> dict[str, Any]:
    probabilities = logits.softmax(-1)
    predictions = probabilities.argmax(-1)
    per_class: list[dict[str, float | int]] = []
    class_nll: list[float] = []
    class_brier: list[float] = []
    class_ece: list[float] = []
    for class_id in range(class_count):
        mask = targets == class_id
        if not mask.any():
            continue
        p = probabilities[mask]
        y = targets[mask]
        confidence = p[:, class_id]
        correct = predictions[mask] == y
        nll = torch.nn.functional.cross_entropy(logits[mask], y).item()
        brier = torch.sum((p - torch.nn.functional.one_hot(y, class_count)) ** 2, -1).mean().item()
        ece = _binary_ece(confidence, (y == class_id).float())
        per_class.append(
            {
                "class_id": class_id,
                "count": int(mask.sum()),
                "accuracy": correct.float().mean().item(),
                "nll": nll,
                "multiclass_brier": brier,
                "classwise_ece": ece,
            }
        )
        class_nll.append(nll)
        class_brier.append(brier)
        class_ece.append(ece)
    sample_nll = torch.nn.functional.cross_entropy(logits, targets).item()
    multiclass_brier = torch.sum(
        (probabilities - torch.nn.functional.one_hot(targets, class_count)) ** 2, -1
    ).mean().item()
    top_confidence = probabilities.max(-1).values
    top_correct = predictions == targets
    return {
        "count": len(targets),
        "accuracy": top_correct.float().mean().item(),
        "macro_accuracy": sum(x["accuracy"] for x in per_class) / len(per_class),
        "nll": sample_nll,
        "macro_nll": sum(class_nll) / len(class_nll),
        "multiclass_brier": multiclass_brier,
        "macro_multiclass_brier": sum(class_brier) / len(class_brier),
        "top_label_ece_15_bins": _binary_ece(top_confidence, top_correct.float(), bins=15),
        "macro_classwise_ece": sum(class_ece) / len(class_ece),
        "mean_confidence": top_confidence.mean().item(),
        "per_class": per_class,
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }


def _binary_ece(confidence: torch.Tensor, correct: torch.Tensor, *, bins: int = 10) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            ece += mask.float().mean().item() * abs(
                confidence[mask].mean().item() - correct[mask].mean().item()
            )
    return ece


def run_probe(
    train_csv: Path,
    split_json: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    seed: int = 17,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 10,
) -> dict[str, Any]:
    input_hash = _sha256(train_csv)
    if input_hash != EXPECTED_SHA256:
        raise ValueError(f"BANKING77 train SHA-256 mismatch: {input_hash}")
    model_dir = model_dir.resolve()
    if model_dir.name != MODEL_REVISION:
        raise ValueError("model directory must be the pinned MiniLM snapshot")
    weight_path = model_dir / "model.safetensors"
    model_hash = _sha256(weight_path)
    if model_hash != MODEL_WEIGHT_SHA256:
        raise ValueError(f"MiniLM weight SHA-256 mismatch: {model_hash}")
    split = json.loads(split_json.read_text(encoding="utf-8"))
    if split["source_sha256"] != input_hash or split["candidate_component_overlap"] != 0:
        raise ValueError("split manifest does not match data or has group overlap")

    csv_text = train_csv.read_text(encoding="utf-8-sig")
    records = list(csv.DictReader(io.StringIO(csv_text, newline="")))
    labels = sorted({row["category"] for row in records})
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_ids = split["train_row_ids"]
    validation_ids = split["validation_row_ids"]
    if set(train_ids) & set(validation_ids) or len(train_ids) + len(validation_ids) != len(records):
        raise ValueError("split IDs are overlapping or incomplete")
    texts = [row["text"] for row in records]
    targets = torch.tensor([label_to_id[row["category"]] for row in records], dtype=torch.long)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True
    ).eval().to("cpu")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    start = time.perf_counter()
    features = _pool_features(texts, tokenizer, encoder, batch_size=batch_size)
    extraction_seconds = time.perf_counter() - start
    encoder_parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    del encoder

    train_x, validation_x = features[train_ids], features[validation_ids]
    train_y, validation_y = targets[train_ids], targets[validation_ids]
    train_label_counts = Counter(train_y.tolist())
    majority_id = train_label_counts.most_common(1)[0][0]
    baseline_logits = torch.zeros((len(validation_ids), len(labels)))
    baseline_logits[:, majority_id] = 1.0
    baseline_metrics = _metrics(baseline_logits, validation_y, len(labels))
    baseline_metrics.pop("predictions")
    baseline_metrics.pop("probabilities")

    head = torch.nn.Linear(features.shape[-1], len(labels))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    fit_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = torch.randperm(len(train_y))
        for start_index in range(0, len(order), batch_size):
            batch_indices = order[start_index : start_index + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                head(train_x[batch_indices]), train_y[batch_indices]
            )
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            logits = head(validation_x)
            metrics = _metrics(logits, validation_y, len(labels))
            validation_macro_nll = metrics["macro_nll"]
        history.append(
            {
                "epoch": epoch,
                "train_ce_last_batch": float(loss.item()),
                "validation_macro_nll": validation_macro_nll,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_accuracy": metrics["macro_accuracy"],
            }
        )
        if validation_macro_nll < best_score - 1e-4:
            best_score = validation_macro_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    fit_seconds = time.perf_counter() - fit_start
    if best_state is None or best_metrics is None:
        raise RuntimeError("no validation-selected checkpoint was produced")

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "best-readout.safetensors"
    save_file(best_state, str(artifact_path))
    predictions = best_metrics.pop("predictions")
    probabilities = best_metrics.pop("probabilities")
    prediction_path = output_dir / "validation-predictions.json"
    prediction_path.write_text(
        json.dumps(
            [
                {
                    "source_row_id": row_id,
                    "target_intent": labels[int(validation_y[index])],
                    "predicted_intent": labels[predictions[index]],
                    "probabilities": probabilities[index],
                }
                for index, row_id in enumerate(validation_ids)
            ],
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "experiment_id": "banking77-frozen-minilm-l3-seed17-attempt-01",
        "task_scope": (
            "closed-set 77-intent English banking query classification; not "
            "arbitrary supplied-choice generalization"
        ),
        "seed": seed,
        "data_sha256": input_hash,
        "split_manifest_sha256": _sha256(split_json),
        "validation_row_id_sha256": split["validation_row_id_sha256"],
        "model_id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": model_hash,
        "encoder_parameters": encoder_parameter_count,
        "readout_parameters": sum(t.numel() for t in best_state.values()),
        "total_parameters": encoder_parameter_count + sum(t.numel() for t in best_state.values()),
        "encoder_trainable_parameters": 0,
        "input_max_length": 128,
        "pooling": "attention-mask mean pooling over final hidden states",
        "device": "cpu",
        "training_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "train_intent_counts": dict(
            sorted(Counter(records[i]["category"] for i in train_ids).items())
        ),
        "validation_intent_counts": dict(
            sorted(Counter(records[i]["category"] for i in validation_ids).items())
        ),
        "selection": {
            "criterion": "validation macro NLL",
            "min_delta": 1e-4,
            "patience_epochs": patience,
            "max_epochs": max_epochs,
            "selected_epoch": best_epoch,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "batch_size": batch_size,
        },
        "baseline_majority": baseline_metrics,
        "best_validation": best_metrics,
        "history": history,
        "runtime_seconds": {"feature_extraction": extraction_seconds, "readout_fit": fit_seconds},
        "prediction_file": prediction_path.name,
        "prediction_sha256": _sha256(prediction_path),
        "readout_file": artifact_path.name,
        "readout_sha256": _sha256(artifact_path),
        "publisher_test_opened": False,
        "sealed_audit_opened": False,
        "gpu_used": False,
        "limitations": [
            "BANKING77 has a known normalized duplicate with conflicting labels",
            "the source lacks user/session IDs; lexical components are only a leakage proxy",
            "validation labels are source labels with fine-grained overlapping intents",
            "this English banking task is not multimodal or general Decision evidence",
        ],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("split_json", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = run_probe(
        args.train_csv, args.split_json, args.model_dir, args.output_dir
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "selected_epoch": result["selection"]["selected_epoch"],
                "validation": {
                    key: result["best_validation"][key]
                    for key in (
                        "accuracy",
                        "macro_accuracy",
                        "nll",
                        "macro_nll",
                        "multiclass_brier",
                        "macro_multiclass_brier",
                        "top_label_ece_15_bins",
                        "macro_classwise_ece",
                        "mean_confidence",
                    )
                },
                "total_parameters": result["total_parameters"],
                "runtime_seconds": result["runtime_seconds"],
                "readout_sha256": result["readout_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
