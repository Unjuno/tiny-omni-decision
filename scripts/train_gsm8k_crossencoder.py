"""Train one fixed GSM8K-derived Decision probe on a pinned MiniLM cross-encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.schema import DecisionExample

MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MODEL_WEIGHT_SHA256 = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
TOTAL_PARAMETERS = 22_713_601
TRAINABLE_PARAMETERS = 1_774_849
SEED = 17
MAX_LENGTH = 512
EPOCHS = 3
BATCH_QUESTIONS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0


def configure_trainable(model: torch.nn.Module) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(("bert.encoder.layer.5.", "classifier."))
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if not names:
        raise ValueError("no final-block/classifier parameters were selected")
    return names


def frozen_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def epoch_option_order(
    example: DecisionExample, epoch: int, seed: int = SEED
) -> tuple[list[str], int]:
    ordered = sorted(
        example.options,
        key=lambda option: hashlib.sha256(
            f"seed={seed}:epoch={epoch}:{example.id}:{option}".encode()
        ).hexdigest(),
    )
    return ordered, ordered.index(example.target)


def grouped_choice_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("scores must have shape [questions, 4]")
    if targets.shape != (scores.shape[0],):
        raise ValueError("targets must contain one option index per question")
    return F.cross_entropy(scores, targets)


def _metrics(logits: list[torch.Tensor], targets: list[int], bins: int = 15) -> dict[str, Any]:
    probabilities = [scores.float().softmax(-1) for scores in logits]
    correct = torch.tensor(
        [
            int(int(probability.argmax()) == target)
            for probability, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    nll = torch.tensor(
        [
            -float(probability[target].clamp_min(1e-12).log())
            for probability, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    brier = torch.tensor(
        [
            float(
                (probability - F.one_hot(torch.tensor(target), num_classes=probability.numel()))
                .square()
                .sum()
            )
            for probability, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    confidence = torch.tensor([float(probability.max()) for probability in probabilities])
    ece = 0.0
    bin_counts = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        count = int(mask.sum())
        bin_counts.append(count)
        if count:
            ece += count / len(targets) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": len(targets),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
        "ece_bin_counts": bin_counts,
    }


def _load_examples(path: Path, expected_split: str) -> list[DecisionExample]:
    examples = [
        DecisionExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples:
        raise ValueError(f"empty {expected_split} file")
    for example in examples:
        if example.split != expected_split or example.modality != "text":
            raise ValueError("dataset split or modality mismatch")
        if example.source != "openai/grade-school-math":
            raise ValueError("unexpected dataset source")
        if example.source_revision != "3101c7d5072418e28b9008a6636bde82a006892c":
            raise ValueError("unexpected dataset source revision")
        if len(example.options) != 4:
            raise ValueError("GSM8K probe expects exactly four options")
    return examples


def _validate_splits(train: list[DecisionExample], validation: list[DecisionExample]) -> None:
    if {item.id for item in train} & {item.id for item in validation}:
        raise ValueError("train/validation sample ID overlap")
    train_groups = {item.source_record_id for item in train}
    validation_groups = {item.source_record_id for item in validation}
    if train_groups & validation_groups:
        raise ValueError("train/validation grouped source identity overlap")


def _tokenize_groups(
    examples: list[DecisionExample],
    tokenizer: Any,
    *,
    epoch: int | None = None,
    batch_indices: list[int] | None = None,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    indices = list(range(len(examples))) if batch_indices is None else batch_indices
    questions, options, targets = [], [], []
    for index in indices:
        example = examples[index]
        if epoch is None:
            ordered, target = example.options, example.options.index(example.target)
        else:
            ordered, target = epoch_option_order(example, epoch)
        questions.extend([f"Question: {example.question}"] * 4)
        options.extend([f"Candidate answer: {option}" for option in ordered])
        targets.append(target)
    encoded = tokenizer(
        questions,
        options,
        truncation=False,
        padding=True,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] > MAX_LENGTH:
        raise ValueError("question/option pair exceeds the frozen 512-token input limit")
    return encoded, targets


def _scores_for_groups(
    model: torch.nn.Module,
    examples: list[DecisionExample],
    tokenizer: Any,
    device: torch.device,
    indices: list[int],
    *,
    question_map: dict[int, str] | None = None,
) -> tuple[list[torch.Tensor], list[int]]:
    questions, options, targets = [], [], []
    for index in indices:
        example = examples[index]
        question = question_map[index] if question_map is not None else example.question
        questions.extend([f"Question: {question}"] * 4)
        options.extend(f"Candidate answer: {option}" for option in example.options)
        targets.append(example.options.index(example.target))
    encoded = tokenizer(
        questions,
        options,
        truncation=False,
        padding=True,
        return_tensors="pt",
    ).to(device)
    if encoded["input_ids"].shape[1] > MAX_LENGTH:
        raise ValueError("question/option pair exceeds the frozen 512-token input limit")
    logits = model(**encoded).logits.reshape(len(indices), 4)
    if not torch.isfinite(logits).all():
        raise ValueError("model produced non-finite choice scores")
    return [row.detach().cpu() for row in logits], targets


def _evaluate(
    model: torch.nn.Module,
    examples: list[DecisionExample],
    tokenizer: Any,
    device: torch.device,
    *,
    question_map: dict[int, str] | None = None,
) -> tuple[list[torch.Tensor], list[int]]:
    model.eval()
    logits: list[torch.Tensor] = []
    targets: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(examples), BATCH_QUESTIONS):
            indices = list(range(start, min(start + BATCH_QUESTIONS, len(examples))))
            values, batch_targets = _scores_for_groups(
                model,
                examples,
                tokenizer,
                device,
                indices,
                question_map=question_map,
            )
            logits.extend(values)
            targets.extend(batch_targets)
    return logits, targets


def _delta(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_predictions(
    path: Path, examples: list[DecisionExample], logits: list[torch.Tensor]
) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for example, scores in zip(examples, logits, strict=True):
            row = {
                "id": example.id,
                "source_record_id": example.source_record_id,
                "options": example.options,
                "target_index": example.options.index(example.target),
                "predicted_index": int(scores.argmax()),
                "probabilities": scores.softmax(-1).tolist(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _paired_bootstrap(
    examples: list[DecisionExample],
    baseline: list[torch.Tensor],
    candidate: list[torch.Tensor],
    targets: list[int],
    *,
    replicates: int = 10_000,
    seed: int = SEED,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.source_record_id, []).append(index)
    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("paired bootstrap requires at least two question groups")

    def per_item(values: list[torch.Tensor]) -> tuple[list[float], list[float], list[float]]:
        accuracy, nll, brier = [], [], []
        for scores, target in zip(values, targets, strict=True):
            probability = scores.float().softmax(-1)
            accuracy.append(float(int(probability.argmax()) == target))
            nll.append(float(-probability[target].clamp_min(1e-12).log()))
            truth = F.one_hot(torch.tensor(target), num_classes=4).float()
            brier.append(float((probability - truth).square().sum()))
        return accuracy, nll, brier

    base_metrics, candidate_metrics = per_item(baseline), per_item(candidate)
    rng = random.Random(seed)
    differences = {name: [] for name in ("accuracy", "nll", "brier")}
    for _ in range(replicates):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        indices = [index for group in sampled for index in groups[group]]
        for metric_index, name in enumerate(differences):
            left = sum(base_metrics[metric_index][index] for index in indices) / len(indices)
            right = sum(candidate_metrics[metric_index][index] for index in indices) / len(indices)
            differences[name].append(right - left)
    return {
        "unit": "question/source_record_id groups (writer identity unknown)",
        "replicates": replicates,
        "seed": seed,
        "interval": "percentile 95%",
        "candidate_minus_pretrained": {
            name: {
                "mean": float(np.mean(values)),
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
            }
            for name, values in differences.items()
        },
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    base_weights = args.model_dir / "model.safetensors"
    if sha256_file(base_weights) != MODEL_WEIGHT_SHA256:
        raise ValueError("pinned MiniLM weight SHA-256 mismatch")
    train_path = args.dataset_dir / "train.jsonl"
    validation_path = args.dataset_dir / "validation.jsonl"
    split_manifest_path = args.dataset_dir / "split-manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    hashes = split_manifest["output_hashes"]
    if (
        sha256_file(train_path) != hashes["train"]
        or sha256_file(validation_path) != hashes["validation"]
    ):
        raise ValueError("generated split hash mismatch")
    train = _load_examples(train_path, "train")
    validation = _load_examples(validation_path, "validation")
    _validate_splits(train, validation)
    source_train_count, source_validation_count = len(train), len(validation)
    if args.smoke:
        train, validation = train[:16], validation[:8]
    args.output_dir.mkdir(parents=True, exist_ok=False)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device = torch.device("cpu")
    total_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, use_safetensors=True
    ).to(device)
    model_loaded_seconds = time.perf_counter() - total_started
    names = configure_trainable(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if total_parameters != TOTAL_PARAMETERS or trainable_count != TRAINABLE_PARAMETERS:
        raise ValueError("pinned model or trainable parameter count mismatch")
    frozen_before = frozen_hash(model)

    # Pre-tokenization length scan fails closed instead of silently truncating.
    max_pair_tokens = 0
    for examples in (train, validation):
        first = [f"Question: {item.question}" for item in examples]
        second = [f"Candidate answer: {option}" for item in examples for option in item.options]
        paired = [question for question in first for _ in range(4)]
        encoded_lengths = tokenizer(
            paired, second, truncation=False, padding=False, add_special_tokens=True
        )["input_ids"]
        max_pair_tokens = max(max_pair_tokens, max(map(len, encoded_lengths)))
    if max_pair_tokens > MAX_LENGTH:
        raise ValueError(f"max question/option input is {max_pair_tokens}, above {MAX_LENGTH}")

    baseline_started = time.perf_counter()
    baseline_logits, validation_targets = _evaluate(model, validation, tokenizer, device)
    baseline_eval_seconds = time.perf_counter() - baseline_started
    baseline_metrics = _metrics(baseline_logits, validation_targets)
    _write_predictions(
        args.output_dir / "pretrained-validation-predictions.jsonl", validation, baseline_logits
    )

    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    updates_per_epoch = math.ceil(len(train) / BATCH_QUESTIONS)
    config = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "seed": SEED,
        "dataset_source": split_manifest["source"],
        "dataset_revision": split_manifest["source_revision"],
        "train_source_sha256": split_manifest["train_source_sha256"],
        "train_jsonl_sha256": hashes["train"],
        "validation_jsonl_sha256": hashes["validation"],
        "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
        "model_weight_bytes": base_weights.stat().st_size,
        "tokenizer_file_sha256": {
            path.name: sha256_file(path)
            for path in sorted(args.model_dir.iterdir())
            if path.is_file()
        },
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_count,
        "trainable_parameter_names": names,
        "initialized_from": (
            "fresh pinned pretrained checkpoint; no prior adapter, optimizer, or QAT checkpoint"
        ),
        "epochs": 1 if args.smoke else EPOCHS,
        "smoke": args.smoke,
        "batch_questions": BATCH_QUESTIONS,
        "updates_per_epoch": updates_per_epoch,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "constant (no scheduler object)",
        "gradient_clip": GRADIENT_CLIP,
        "dropout": "disabled by model.eval() while gradients remain enabled",
        "training_option_shuffle": "deterministic per question and epoch by SHA-256 rank",
        "checkpoint_selection": (
            "earliest minimum validation NLL across pretrained baseline and epochs 1..3"
        ),
        "baseline_validation_metrics": baseline_metrics,
        "max_question_option_tokens": max_pair_tokens,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "source_train_examples": source_train_count,
        "source_validation_examples": source_validation_count,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else None,
        "model_dtype": str(next(model.parameters()).dtype),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "model_load_seconds": model_loaded_seconds,
        "pretrained_baseline_eval_seconds": baseline_eval_seconds,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "official_test_read": False,
        "writer_disjointness": "unknown; source does not expose writer IDs",
    }
    config["runner_sha256"] = sha256_file(Path(__file__))
    _write_json(args.output_dir / "effective-config.json", config)

    started_local = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    history = [
        {
            "epoch": 0,
            "step": 0,
            "rolling_train_ce": None,
            "validation": baseline_metrics,
            "role": "fresh pretrained baseline",
        }
    ]
    best_nll = baseline_metrics["nll"]
    best_epoch, best_step = 0, 0
    best_logits = baseline_logits
    torch.save(_delta(model), args.output_dir / "best-trainable-weights.pt")
    _write_predictions(
        args.output_dir / "validation-epoch-000-predictions.jsonl", validation, baseline_logits
    )

    run_epochs = 1 if args.smoke else EPOCHS
    for epoch in range(1, run_epochs + 1):
        model.eval()
        order = list(range(len(train)))
        random.Random(SEED + epoch).shuffle(order)
        total_loss, seen, last_norm = 0.0, 0, 0.0
        for start in range(0, len(order), BATCH_QUESTIONS):
            indices = order[start : start + BATCH_QUESTIONS]
            questions, options, target_indices = [], [], []
            for index in indices:
                example = train[index]
                choices, target_index = epoch_option_order(example, epoch)
                questions.extend([f"Question: {example.question}"] * 4)
                options.extend(f"Candidate answer: {choice}" for choice in choices)
                target_indices.append(target_index)
            encoded = tokenizer(
                questions,
                options,
                truncation=False,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits.reshape(len(indices), 4)
            target_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
            loss = grouped_choice_loss(logits, target_tensor)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            last_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP, error_if_nonfinite=True)
            )
            if not any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in trainable
            ):
                raise ValueError("trainable parameters received no non-zero gradient")
            if any(
                parameter.grad is not None
                for parameter in model.parameters()
                if not parameter.requires_grad
            ):
                raise ValueError("a frozen parameter received gradients")
            optimizer.step()
            batch_size = len(indices)
            total_loss += float(loss.detach()) * batch_size
            seen += batch_size

        step = epoch * updates_per_epoch
        validation_logits, validation_targets = _evaluate(model, validation, tokenizer, device)
        validation_metrics = _metrics(validation_logits, validation_targets)
        if frozen_hash(model) != frozen_before:
            raise ValueError("frozen pretrained parameters changed")
        history.append(
            {
                "epoch": epoch,
                "step": step,
                "rolling_train_ce": total_loss / seen,
                "validation": validation_metrics,
                "last_gradient_norm": last_norm,
            }
        )
        _write_predictions(
            args.output_dir / f"validation-epoch-{epoch:03d}-predictions.jsonl",
            validation,
            validation_logits,
        )
        if validation_metrics["nll"] < best_nll:
            best_nll = validation_metrics["nll"]
            best_epoch, best_step = epoch, step
            best_logits = validation_logits
            torch.save(_delta(model), args.output_dir / "best-trainable-weights.pt")
        _write_json(args.output_dir / "learning-history.json", history)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "step": step,
                    "train_ce": total_loss / seen,
                    "validation": validation_metrics,
                    "seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    final_delta = _delta(model)
    torch.save(final_delta, args.output_dir / "final-trainable-weights.pt")
    chosen_delta = torch.load(
        args.output_dir / "best-trainable-weights.pt", weights_only=True, map_location="cpu"
    )
    model.load_state_dict(chosen_delta, strict=False)
    reloaded_logits, reloaded_targets = _evaluate(model, validation, tokenizer, device)
    if reloaded_targets != validation_targets or not all(
        torch.equal(left, right) for left, right in zip(reloaded_logits, best_logits, strict=True)
    ):
        raise ValueError("best checkpoint reload does not reproduce selected validation scores")

    reversed_examples = [
        example.model_copy(update={"options": list(reversed(example.options))})
        for example in validation
    ]
    reversed_logits, _ = _evaluate(model, reversed_examples, tokenizer, device)
    reverse_delta = 0.0
    reverse_argmax_flips = 0
    for original, reversed_score in zip(reloaded_logits, reversed_logits, strict=True):
        remapped = reversed_score.flip(0)
        left, right = original.softmax(-1), remapped.softmax(-1)
        reverse_delta = max(reverse_delta, float((left - right).abs().max()))
        reverse_argmax_flips += int(left.argmax() != right.argmax())

    swap_map = {
        index: validation[(index + 1) % len(validation)].question
        for index in range(len(validation))
    }
    swapped_logits, swapped_targets = _evaluate(
        model, validation, tokenizer, device, question_map=swap_map
    )
    swapped_metrics = _metrics(swapped_logits, swapped_targets)
    selected_metrics = _metrics(reloaded_logits, reloaded_targets)
    bootstrap = _paired_bootstrap(
        validation,
        baseline_logits,
        reloaded_logits,
        validation_targets,
    )
    _write_predictions(
        args.output_dir / "selected-validation-predictions.jsonl", validation, reloaded_logits
    )
    _write_predictions(
        args.output_dir / "question-swapped-predictions.jsonl", validation, swapped_logits
    )
    result = {
        "status": "PASS",
        "claim_scope": (
            "GSM8K-derived English arithmetic four-choice internal development split only"
        ),
        "best_epoch": best_epoch,
        "best_step": best_step,
        "final_epoch": run_epochs,
        "final_step": run_epochs * updates_per_epoch,
        "updates_per_epoch": updates_per_epoch,
        "train_examples_per_epoch": len(train),
        "total_question_exposures": len(train) * EPOCHS,
        "pretrained_validation_metrics": baseline_metrics,
        "selected_validation_metrics": selected_metrics,
        "final_epoch_validation_metrics": history[-1]["validation"],
        "paired_group_bootstrap": bootstrap,
        "option_order_control": {
            "max_probability_delta_after_reverse_and_remap": reverse_delta,
            "argmax_flips": reverse_argmax_flips,
            "count": len(validation),
        },
        "question_swap_control": swapped_metrics,
        "frozen_parameter_sha256": frozen_before,
        "best_weights_sha256": sha256_file(args.output_dir / "best-trainable-weights.pt"),
        "final_weights_sha256": sha256_file(args.output_dir / "final-trainable-weights.pt"),
        "training_and_selection_seconds": time.perf_counter() - started,
        "model_load_seconds": model_loaded_seconds,
        "pretrained_baseline_eval_seconds": baseline_eval_seconds,
        "total_process_seconds": time.perf_counter() - total_started,
        "started_local": started_local,
        "finished_local": datetime.now().astimezone().isoformat(),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "official_test_read": False,
        "writer_group_independence": "unknown",
        "product_promotion": False,
        "smoke": args.smoke,
    }
    _write_json(args.output_dir / "result.json", result)


if __name__ == "__main__":
    main()
