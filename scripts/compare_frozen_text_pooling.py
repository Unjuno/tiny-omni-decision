"""Compare frozen MiniLM token-pooling variants on fixed Typed Decisions Synth splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_text_probe import (
    CandidateScorer,
    _assert_disjoint,
    _identity_sets,
    _load_examples,
    _metrics,
    _predict,
)
from tiny_omni_decision.dataset import DatasetManifest, sha256_file

VARIANTS = ("final_layer_masked_mean", "middle_layer_masked_mean", "final_layer_first_token")
EXPECTED_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrained_reuse/path_a_text_pooling.yaml")
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _pool_hidden(hidden: Tensor, mask: Tensor, variant: str) -> Tensor:
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("hidden must be [batch, sequence, width] and mask [batch, sequence]")
    if variant == "final_layer_first_token":
        return hidden[:, 0]
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    if variant in {"final_layer_masked_mean", "middle_layer_masked_mean"}:
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    raise ValueError(f"unknown pooling variant: {variant}")


def _extract_variant_features(
    examples: list[Any],
    tokenizer: Any,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, list[Tensor]], list[int]]:
    contexts = [f"State: {example.state}\nQuestion: {example.question}" for example in examples]
    options = [f"Candidate answer: {option}" for example in examples for option in example.options]
    ranges: list[tuple[int, int]] = []
    offset = 0
    targets = []
    for example in examples:
        end = offset + len(example.options)
        ranges.append((offset, end))
        offset = end
        targets.append(example.options.index(example.target))

    all_text = contexts + options
    pooled_by_variant: dict[str, list[Tensor]] = {variant: [] for variant in VARIANTS}
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(all_text), batch_size):
            encoded = tokenizer(
                all_text[start : start + batch_size],
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            output = encoder(**encoded, output_hidden_states=True)
            hidden_states = output.hidden_states
            if hidden_states is None or len(hidden_states) < 3:
                raise ValueError(
                    "encoder did not return enough hidden states for middle-layer pooling"
                )
            if hidden_states[-1].shape != output.last_hidden_state.shape:
                raise ValueError(
                    "last hidden state differs from the final hidden-state tuple entry"
                )
            mask = encoded["attention_mask"]
            pools = {
                "final_layer_masked_mean": _pool_hidden(
                    hidden_states[-1], mask, "final_layer_masked_mean"
                ),
                "middle_layer_masked_mean": _pool_hidden(
                    hidden_states[-2], mask, "middle_layer_masked_mean"
                ),
                "final_layer_first_token": _pool_hidden(
                    hidden_states[-1], mask, "final_layer_first_token"
                ),
            }
            for variant, pooled in pools.items():
                if not torch.isfinite(pooled).all():
                    raise ValueError(f"non-finite features for {variant}")
                pooled_by_variant[variant].append(pooled.float().cpu())

    features_by_variant: dict[str, list[Tensor]] = {}
    for variant, chunks in pooled_by_variant.items():
        pooled = torch.cat(chunks)
        context_features = pooled[: len(contexts)]
        option_features = pooled[len(contexts) :]
        examples_features = []
        for context, (lo, hi) in zip(context_features, ranges, strict=True):
            candidates = option_features[lo:hi]
            expanded = context.unsqueeze(0).expand(candidates.shape[0], -1)
            examples_features.append(
                torch.cat(
                    [expanded, candidates, (expanded - candidates).abs(), expanded * candidates],
                    dim=-1,
                )
            )
        if len(option_features) != offset:
            raise AssertionError("candidate text ranges do not cover encoded option features")
        features_by_variant[variant] = examples_features
    return features_by_variant, targets


def _train_variant(
    features: list[Tensor],
    targets: list[int],
    validation_features: list[Tensor],
    validation_targets: list[int],
    *,
    seed: int,
    epochs: int,
    batch_questions: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Tensor], list[Tensor], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    scorer = CandidateScorer(features[0].shape[-1]).to(device)
    parameter_count = sum(parameter.numel() for parameter in scorer.parameters())
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    order_rng = random.Random(seed)
    history = []
    best_nll = float("inf")
    best_state = None
    best_train_logits: list[Tensor] = []
    best_val_logits: list[Tensor] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(range(len(features)))
        order_rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), batch_questions):
            indices = order[start : start + batch_questions]
            rows = [features[index].to(device) for index in indices]
            batch_targets = torch.tensor([targets[index] for index in indices], device=device)
            widths = [row.shape[0] for row in rows]
            scores = scorer(torch.cat(rows, dim=0))
            padded = scores.new_full((len(rows), max(widths)), -1e4)
            offset = 0
            for index, width in enumerate(widths):
                padded[index, :width] = scores[offset : offset + width]
                offset += width
            loss = F.cross_entropy(padded, batch_targets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in scorer.parameters()
            ):
                raise ValueError("non-finite readout gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        train_logits = _predict(scorer, features, device)
        val_logits = _predict(scorer, validation_features, device)
        train_metrics = _metrics(train_logits, targets)
        val_metrics = _metrics(val_logits, validation_targets)
        history.append(
            {
                "epoch": epoch,
                "train_ce": sum(losses) / len(losses),
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
            }
        )
        if val_metrics["nll"] < best_nll:
            best_nll = float(val_metrics["nll"])
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
            best_train_logits = train_logits
            best_val_logits = val_logits
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("validation did not select a checkpoint")
    report = {
        "trainable_parameters": parameter_count,
        "best_epoch": min(history, key=lambda item: item["validation_metrics"]["nll"])["epoch"],
        "train_metrics_at_best": _metrics(best_train_logits, targets),
        "validation_metrics_at_best": _metrics(best_val_logits, validation_targets),
        "training_seconds": elapsed,
        "history": history,
    }
    return report, best_state, best_val_logits, history


def _predictions(examples: list[Any], logits: list[Tensor]) -> list[dict[str, Any]]:
    rows = []
    for example, scores in zip(examples, logits, strict=True):
        probabilities = torch.softmax(scores.float(), dim=-1)
        rows.append(
            {
                "id": example.id,
                "source_record_id": example.source_record_id,
                "target": example.target,
                "options": example.options,
                "prediction": example.options[int(scores.argmax())],
                "probabilities": probabilities.tolist(),
            }
        )
    return rows


def main() -> None:
    args = _args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["encoder"]["revision"] != EXPECTED_REVISION:
        raise ValueError("unexpected MiniLM revision in pooling comparison config")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset_cfg = config["dataset"]
    encoder_cfg = config["encoder"]
    readout_cfg = config["readout"]
    train_path = Path(dataset_cfg["train_file"])
    validation_path = Path(dataset_cfg["validation_file"])
    manifest_path = Path(dataset_cfg["manifest"])
    if not train_path.is_file() or not validation_path.is_file() or not args.model.is_dir():
        raise FileNotFoundError("configured datasets or local model checkpoint are missing")

    torch.set_num_threads(4)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device = torch.device("cpu")
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    )
    train_manifest = manifest.model_copy(update={"split": "train"})
    validation_manifest = manifest.model_copy(update={"split": "validation"})
    train = _load_examples(train_path, train_manifest)
    validation = _load_examples(validation_path, validation_manifest)
    _assert_disjoint(train, validation)
    train_groups, _ = _identity_sets(train)
    validation_groups, _ = _identity_sets(validation)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, local_files_only=True, add_pooling_layer=False
    ).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    feature_started = time.perf_counter()
    train_features, train_targets = _extract_variant_features(
        train,
        tokenizer,
        encoder,
        device,
        int(config.get("feature_batch_size", 64)),
        int(encoder_cfg["max_length"]),
    )
    validation_features, validation_targets = _extract_variant_features(
        validation,
        tokenizer,
        encoder,
        device,
        int(config.get("feature_batch_size", 64)),
        int(encoder_cfg["max_length"]),
    )
    feature_seconds = time.perf_counter() - feature_started
    expected_variants = tuple(encoder_cfg["pooling_variants"])
    if expected_variants != VARIANTS:
        raise ValueError(
            f"pooling variants in config differ from implementation: {expected_variants}"
        )

    feature_path = output / "pooled-features.pt"
    torch.save(
        {
            "train": train_features,
            "train_targets": train_targets,
            "validation": validation_features,
            "validation_targets": validation_targets,
        },
        feature_path,
    )
    option_counts = Counter(len(example.options) for example in train)
    variant_reports = {}
    for variant in VARIANTS:
        variant_dir = output / variant
        variant_dir.mkdir()
        report, state, val_logits, _ = _train_variant(
            train_features[variant],
            train_targets,
            validation_features[variant],
            validation_targets,
            seed=int(config["seed"]),
            epochs=int(readout_cfg["epochs"]),
            batch_questions=int(readout_cfg["batch_questions"]),
            device=device,
        )
        torch.save(state, variant_dir / "best-readout.pt")
        prediction_rows = _predictions(validation, val_logits)
        prediction_path = variant_dir / "validation-predictions.jsonl"
        prediction_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
            encoding="utf-8",
        )
        variant_reports[variant] = {
            **report,
            "readout_sha256": sha256_file(variant_dir / "best-readout.pt"),
            "validation_predictions_sha256": sha256_file(prediction_path),
            "prediction_count": len(prediction_rows),
        }

    report = {
        "schema_version": 1,
        "status": "complete_text_pooling_ablation_development_probe",
        "task_claim": "Synthetic Typed Decisions Synth only; no real-world or blind-audit claim.",
        "seed": int(config["seed"]),
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(Path(__file__)),
        "source_revision": manifest.revision,
        "model_revision": encoder_cfg["revision"],
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "tokenizer_json_sha256": sha256_file(args.model / "tokenizer.json"),
        "model_weights_sha256": sha256_file(args.model / "model.safetensors"),
        "train_file_sha256": sha256_file(train_path),
        "validation_file_sha256": sha256_file(validation_path),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_unique_states": len(train_groups),
        "validation_unique_states": len(validation_groups),
        "train_validation_state_overlap": 0,
        "train_validation_content_overlap": 0,
        "train_sample_order_sha256": hashlib.sha256(
            "\n".join(example.id for example in train).encode()
        ).hexdigest(),
        "validation_sample_order_sha256": hashlib.sha256(
            "\n".join(example.id for example in validation).encode()
        ).hexdigest(),
        "option_count_distribution": dict(sorted(option_counts.items())),
        "feature_cache_bytes": feature_path.stat().st_size,
        "feature_cache_sha256": sha256_file(feature_path),
        "feature_extraction_seconds": feature_seconds,
        "frozen_encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "frozen_encoder_parameter_dtype": str(next(encoder.parameters()).dtype),
        "frozen_encoder_trainable_parameters": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "device": str(device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "pooling_variants": variant_reports,
        "readout_spec": {
            "architecture": "shared_candidate_mlp",
            "input": "context, option, absolute_difference, elementwise_product",
            "hidden_size": int(readout_cfg["hidden_size"]),
            "optimizer": readout_cfg["optimizer"],
            "learning_rate": float(readout_cfg["learning_rate"]),
            "weight_decay": float(readout_cfg["weight_decay"]),
            "epochs": int(readout_cfg["epochs"]),
            "batch_questions": int(readout_cfg["batch_questions"]),
            "selection": readout_cfg["checkpoint_selection"],
        },
        "sealed_audit_loaded": False,
        "legacy_final_evaluation_loaded": False,
        "note": (
            "Same split/order, seed, option scoring architecture, optimizer and budget "
            "across variants. "
            "Validation selected each variant's checkpoint; results are development evidence only."
        ),
    }
    (output / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
