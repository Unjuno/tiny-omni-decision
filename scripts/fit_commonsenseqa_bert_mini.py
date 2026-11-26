"""Fit a frozen BERT-Mini option scorer on the pinned train-only CSQA split.

The model encoder is frozen; only a shared scalar candidate scorer is learned.
The script runs on CPU, uses one training Parquet shard, and never reads an
official validation or test shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from audit_commonsenseqa_train import (
    SPLIT_SEED,
    answer_alias_distribution,
    audit_train_parquet,
    normalize,
    stable_split,
)
from safetensors.torch import save_file

MODEL_ID = "google/bert_uncased_L-4_H-256_A-4"
MODEL_REVISION = "387825ce42dbb39b87911cdf8e383ee3b25184f8"
EXPECTED_MODEL_FILES = {
    "config.json": "61716a972c73e9cc1d6f1165e27e577c1f4012a3cc480e431f427a78d5071ece",
    "pytorch_model.bin": "01671831dfec6818302153919efbfd6ee4ad91dd9fba7f3649bdd6cb7b0f16e4",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
EXPECTED_ENCODER_PARAMETERS = 11_170_560
CHOICE_LABELS = "ABCDE"


def soft_target_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def evaluate_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    concepts: list[str],
    bins: int = 10,
) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    per_row_nll = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    correct = targets.gather(1, probabilities.argmax(dim=-1, keepdim=True)).squeeze(1) > 0
    confidence = probabilities.max(dim=-1).values
    correctness = correct.float()
    calibration_error = torch.zeros((), dtype=logits.dtype)
    for bin_index in range(bins):
        lower, upper = bin_index / bins, (bin_index + 1) / bins
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            calibration_error += in_bin.float().mean() * (
                confidence[in_bin].mean() - correctness[in_bin].mean()
            ).abs()
    per_row_brier = ((probabilities - targets) ** 2).sum(dim=-1)

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, concept in enumerate(concepts):
        grouped[concept].append(index)
    macro_accuracy = []
    macro_nll = []
    macro_brier = []
    for indices in grouped.values():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        macro_accuracy.append(correct[index_tensor].float().mean().item())
        macro_nll.append(per_row_nll[index_tensor].mean().item())
        macro_brier.append(per_row_brier[index_tensor].mean().item())
    return {
        "rows": len(concepts),
        "concept_groups": len(grouped),
        "accuracy": correct.float().mean().item(),
        "nll": per_row_nll.mean().item(),
        "brier": per_row_brier.mean().item(),
        "ece_10": calibration_error.item(),
        "mean_confidence": confidence.mean().item(),
        "macro_concept_accuracy": sum(macro_accuracy) / len(macro_accuracy),
        "macro_concept_nll": sum(macro_nll) / len(macro_nll),
        "macro_concept_brier": sum(macro_brier) / len(macro_brier),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_files(model_dir: Path) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for name, expected_hash in EXPECTED_MODEL_FILES.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned model file: {name}")
        digest = _sha256(path)
        if digest != expected_hash:
            raise ValueError(f"pinned model hash mismatch: {name}")
        receipts[name] = {"bytes": path.stat().st_size, "sha256": digest}
    return receipts


def _encode_features(
    rows: list[dict[str, Any]],
    model_dir: Path,
    max_length: int,
    question_batch_size: int,
    cpu_threads: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=False,
        weights_only=True,
    ).cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_ENCODER_PARAMETERS:
        raise ValueError(f"unexpected encoder parameter count: {parameter_count}")

    feature_chunks: list[torch.Tensor] = []
    truncated_pairs = 0
    total_pairs = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), question_batch_size):
            chunk = rows[offset : offset + question_batch_size]
            questions = [str(row["question"]) for row in chunk]
            choices = [str(option) for row in chunk for option in row["choices"]["text"]]
            paired_questions = [question for question in questions for _ in range(5)]
            untruncated = tokenizer(
                paired_questions,
                choices,
                truncation=False,
                add_special_tokens=True,
            )
            truncated_pairs += sum(len(ids) > max_length for ids in untruncated["input_ids"])
            total_pairs += len(choices)
            inputs = tokenizer(
                paired_questions,
                choices,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            output = model(**inputs)
            feature_chunks.append(
                output.last_hidden_state[:, 0, :].reshape(len(chunk), 5, -1).cpu()
            )
    features = torch.cat(feature_chunks, dim=0)
    elapsed = time.perf_counter() - started
    return features, {
        "encoder_parameter_count": parameter_count,
        "trainable_encoder_parameters": 0,
        "tokenizer_is_fast": tokenizer.is_fast,
        "max_length": max_length,
        "total_candidate_pairs": total_pairs,
        "pairs_truncated": truncated_pairs,
        "feature_extraction_seconds": elapsed,
        "feature_shape": list(features.shape),
    }


def _metrics_for_head(
    head: torch.nn.Linear,
    features: torch.Tensor,
    targets: torch.Tensor,
    concepts: list[str],
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        logits = head(features).squeeze(-1)
    return logits, evaluate_logits(logits, targets, concepts)


def run_experiment(
    train_path: Path,
    model_dir: Path,
    output_dir: Path,
    seed: int = 17,
    max_length: int = 128,
    question_batch_size: int = 16,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
    max_epochs: int = 80,
    patience: int = 8,
    min_delta: float = 0.0001,
    cpu_threads: int = 4,
) -> dict[str, Any]:
    experiment_started = time.perf_counter()
    cpu_started = time.process_time()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; this experiment is CPU-only")
    if not train_path.name.casefold().startswith("train"):
        raise ValueError("only the pinned training shard may be read")
    model_file_receipts = _verify_model_files(model_dir)
    data_audit = audit_train_parquet(train_path)
    import pyarrow.parquet as parquet

    rows = parquet.read_table(train_path).to_pylist()
    train_rows = [row for row in rows if stable_split(str(row["question_concept"])) == "train"]
    validation_rows = [
        row for row in rows if stable_split(str(row["question_concept"])) == "validation"
    ]
    if not train_rows or not validation_rows:
        raise ValueError("fixed concept-group split produced an empty partition")

    ordered_rows = train_rows + validation_rows
    features, extraction_receipt = _encode_features(
        ordered_rows,
        model_dir,
        max_length,
        question_batch_size,
        cpu_threads,
    )
    train_count = len(train_rows)
    train_features = features[:train_count]
    validation_features = features[train_count:]

    def targets_for(selected: list[dict[str, Any]]) -> torch.Tensor:
        target_rows = []
        for row in selected:
            alias_distribution = answer_alias_distribution(row)
            target_rows.append(
                [alias_distribution.get(label, 0.0) for label in CHOICE_LABELS]
            )
        return torch.tensor(target_rows, dtype=torch.float32)

    train_targets = targets_for(train_rows)
    validation_targets = targets_for(validation_rows)
    train_concepts = [normalize(str(row["question_concept"])) for row in train_rows]
    validation_concepts = [normalize(str(row["question_concept"])) for row in validation_rows]

    torch.manual_seed(seed)
    head = torch.nn.Linear(EXPECTED_ENCODER_PARAMETERS * 0 + 256, 1)
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters())
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        head.train()
        order = torch.randperm(train_count, generator=generator)
        train_losses: list[float] = []
        for start in range(0, train_count, batch_size):
            indices = order[start : start + batch_size]
            logits = head(train_features[indices]).squeeze(-1)
            loss = soft_target_nll(logits, train_targets[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"nonfinite gradient at epoch {epoch}")
            optimizer.step()
            train_losses.append(loss.item())

        head.eval()
        train_logits, train_metrics = _metrics_for_head(
            head, train_features, train_targets, train_concepts
        )
        validation_logits, validation_metrics = _metrics_for_head(
            head, validation_features, validation_targets, validation_concepts
        )
        selector = validation_metrics["macro_concept_nll"]
        record = {
            "epoch": epoch,
            "optimizer_updates": (train_count + batch_size - 1) // batch_size,
            "mean_minibatch_train_nll": sum(train_losses) / len(train_losses),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "validation_macro_concept_nll_selector": selector,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        if selector < best_score - min_delta:
            best_score = selector
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("validation selector did not produce a best checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    selected_logits, selected_metrics = _metrics_for_head(
        head, validation_features, validation_targets, validation_concepts
    )
    permutation = torch.tensor([4, 1, 3, 0, 2])
    permuted_logits = head(validation_features[:, permutation]).squeeze(-1)
    order_error = (permuted_logits - selected_logits[:, permutation]).abs().max().item()
    if order_error > 1e-7:
        raise AssertionError("shared scorer is not permutation equivariant")

    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / "best-head.safetensors"
    save_file(
        {key: value.detach().contiguous().cpu() for key, value in head.state_dict().items()},
        str(head_path),
    )
    reloaded = torch.nn.Linear(256, 1)
    from safetensors.torch import load_file

    reloaded.load_state_dict(load_file(str(head_path), device="cpu"))
    reloaded_logits, reload_metrics = _metrics_for_head(
        reloaded, validation_features, validation_targets, validation_concepts
    )
    max_reload_error = (reloaded_logits - selected_logits).abs().max().item()
    if max_reload_error > 1e-7 or reload_metrics != selected_metrics:
        raise AssertionError("reloaded head changed validation outputs")

    predictions_path = output_dir / "validation-predictions.jsonl"
    probabilities = torch.softmax(selected_logits, dim=-1)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as sink:
        for index, row in enumerate(validation_rows):
            aliases = answer_alias_distribution(row)
            sink.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "target": [aliases.get(label, 0.0) for label in CHOICE_LABELS],
                        "probabilities": probabilities[index].tolist(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    history_path = output_dir / "training-history.jsonl"
    history_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in history),
        encoding="utf-8",
    )
    metadata = {
        "experiment_id": "commonsenseqa-bert-mini-frozen-head-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": "Apache-2.0",
        "model_files": model_file_receipts,
        "dataset_revision": "94630fe30dad47192a8546eb75f094926d47e155",
        "data_audit": data_audit,
        "split_seed": SPLIT_SEED,
        "split_counts": {"train": len(train_rows), "validation": len(validation_rows)},
        "split_id_order_sha256": data_audit["split_id_order_sha256"],
        "effective_config": {
            "seed": seed,
            "max_length": max_length,
            "question_batch_size_for_features": question_batch_size,
            "scorer_batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "selection_metric": "minimum macro-concept validation NLL",
            "target_semantics": (
                "uniform mass over labels with text identical to the annotated correct answer"
            ),
            "encoder_frozen": True,
            "feature": "last-layer CLS for question-option pair",
            "cpu_threads": cpu_threads,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": "cpu",
            "cuda_initialized": torch.cuda.is_initialized(),
            "feature_extraction": extraction_receipt,
            "epochs_completed": len(history),
            "selected_epoch": best_epoch,
            "trainable_parameter_count": trainable_parameters,
            "combined_parameter_count": EXPECTED_ENCODER_PARAMETERS + trainable_parameters,
            "permutation_equivariance_max_abs_error": order_error,
            "reload_max_abs_logit_error": max_reload_error,
        },
        "selected_validation_metrics": selected_metrics,
        "artifact_hashes": {
            head_path.name: _sha256(head_path),
            predictions_path.name: _sha256(predictions_path),
            history_path.name: _sha256(history_path),
        },
        "experiment_wall_seconds": time.perf_counter() - experiment_started,
        "experiment_cpu_seconds": time.process_time() - cpu_started,
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    if not args.train.is_file():
        raise FileNotFoundError(args.train)
    metadata = run_experiment(
        args.train, args.model_dir, args.output_dir, cpu_threads=args.cpu_threads
    )
    print(json.dumps(metadata["selected_validation_metrics"], indent=2, sort_keys=True))
    print(f"selected epoch: {metadata['runtime']['selected_epoch']}")
    print(f"artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
