"""Train a frozen-feature Speech Commands scorer for multiple question types per clip."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_audio_probe import LABELS, _rows, _validate_splits
from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from tiny_omni_decision.dataset import sha256_file

QUESTION_TASKS = ("keyword_identity", "direction_presence", "response_presence", "switch_presence")
DIRECTION_WORDS = {"up", "down", "left", "right"}
RESPONSE_WORDS = {"yes", "no"}
SWITCH_WORDS = {"on", "off"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrained_reuse/path_a_audio_multiquery.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def build_query_specs(label: str) -> list[dict[str, Any]]:
    if label not in LABELS:
        raise ValueError(f"unknown Speech Commands label: {label}")
    return [
        {
            "question_type": "keyword_identity",
            "question": "Which spoken English keyword is present in the clip?",
            "options": list(LABELS),
            "target_index": LABELS.index(label),
        },
        {
            "question_type": "direction_presence",
            "question": "Is a direction command spoken in the clip?",
            "options": ["direction command", "not a direction command"],
            "target_index": 0 if label in DIRECTION_WORDS else 1,
        },
        {
            "question_type": "response_presence",
            "question": "Is an affirmative or negative response spoken in the clip?",
            "options": ["yes-or-no response", "not a yes-or-no response"],
            "target_index": 0 if label in RESPONSE_WORDS else 1,
        },
        {
            "question_type": "switch_presence",
            "question": "Is an on-or-off control word spoken in the clip?",
            "options": ["on-or-off control word", "not an on-or-off control word"],
            "target_index": 0 if label in SWITCH_WORDS else 1,
        },
    ]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode_texts(
    texts: list[str], model_path: Path, device: torch.device, batch_size: int = 64
) -> tuple[dict[str, Tensor], int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vectors: dict[str, Tensor] = {}
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=64,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float()
            if not torch.isfinite(pooled).all():
                raise ValueError("MiniLM produced non-finite query or option embeddings")
            vectors.update(
                (text, vector.cpu()) for text, vector in zip(batch_texts, pooled, strict=True)
            )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vectors, parameter_count


def _pair_features(audio: Tensor, question: Tensor, candidates: Tensor) -> Tensor:
    if audio.ndim != 1 or question.shape != audio.shape or candidates.ndim != 2:
        raise ValueError("audio, question, and option vectors must have compatible dimensions")
    if candidates.shape[-1] != audio.shape[-1]:
        raise ValueError("audio, question, and option vectors must have compatible dimensions")
    audio_rows = audio.unsqueeze(0).expand(len(candidates), -1)
    question_rows = question.unsqueeze(0).expand(len(candidates), -1)
    return torch.cat(
        [
            audio_rows,
            question_rows,
            candidates,
            (question_rows - candidates).abs(),
            (audio_rows - candidates).abs(),
            question_rows * candidates,
            audio_rows * candidates,
        ],
        dim=-1,
    )


def make_query_examples(
    rows: list[dict[str, Any]], audio_features: Tensor, text_vectors: dict[str, Tensor]
) -> list[dict[str, Any]]:
    if len(rows) != len(audio_features):
        raise ValueError("audio rows and cached feature count differ")
    examples = []
    for row, audio in zip(rows, audio_features, strict=True):
        for spec in build_query_specs(str(row["target"])):
            question_text = f"Question: {spec['question']}"
            option_texts = [f"Candidate answer: {option}." for option in spec["options"]]
            try:
                question_vector = text_vectors[question_text]
                candidate_vectors = torch.stack([text_vectors[text] for text in option_texts])
            except KeyError as exc:
                raise ValueError(f"missing frozen text embedding for {exc.args[0]}") from exc
            examples.append(
                {
                    "sample_id": str(row["id"]),
                    "target_label": str(row["target"]),
                    "question_type": spec["question_type"],
                    "question": spec["question"],
                    "options": list(spec["options"]),
                    "target_index": int(spec["target_index"]),
                    "features": _pair_features(audio.float(), question_vector, candidate_vectors),
                }
            )
    return examples


def _batch_logits(
    scorer: nn.Module, batch: list[dict[str, Any]], device: torch.device
) -> tuple[Tensor, Tensor]:
    max_options = max(len(example["options"]) for example in batch)
    input_size = int(batch[0]["features"].shape[-1])
    features = torch.zeros((len(batch), max_options, input_size), dtype=torch.float32)
    valid = torch.zeros((len(batch), max_options), dtype=torch.bool)
    targets = []
    for index, example in enumerate(batch):
        count = len(example["options"])
        features[index, :count] = example["features"]
        valid[index, :count] = True
        targets.append(int(example["target_index"]))
    logits = scorer(features.to(device).reshape(-1, input_size)).reshape(len(batch), max_options)
    logits = logits.masked_fill(~valid.to(device), torch.finfo(logits.dtype).min)
    return logits, torch.tensor(targets, dtype=torch.long, device=device)


def _evaluate(
    scorer: nn.Module, examples: list[dict[str, Any]], device: torch.device, batch_size: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scorer.eval()
    grouped_logits: dict[str, list[Tensor]] = defaultdict(list)
    grouped_targets: dict[str, list[int]] = defaultdict(list)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            logits, targets = _batch_logits(scorer, batch, device)
            for index, example in enumerate(batch):
                count = len(example["options"])
                row_logits = logits[index, :count].detach().cpu()
                target = int(targets[index].item())
                grouped_logits[example["question_type"]].append(row_logits)
                grouped_targets[example["question_type"]].append(target)
                predicted = int(row_logits.argmax().item())
                predictions.append(
                    {
                        "sample_id": example["sample_id"],
                        "target_label": example["target_label"],
                        "question_type": example["question_type"],
                        "question": example["question"],
                        "options": example["options"],
                        "target_index": target,
                        "prediction_index": predicted,
                        "target": example["options"][target],
                        "prediction": example["options"][predicted],
                        "logits": row_logits.tolist(),
                        "probabilities": torch.softmax(row_logits, dim=-1).tolist(),
                    }
                )
    by_type = {
        key: {
            **_metrics(grouped_logits[key], grouped_targets[key]),
            "query_count": len(grouped_targets[key]),
        }
        for key in QUESTION_TASKS
    }
    macro = {
        key: sum(float(by_type[name][key]) for name in QUESTION_TASKS) / len(QUESTION_TASKS)
        for key in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    return {"by_question_type": by_type, "macro_question_type": macro}, predictions


def _select_validation_nll(metrics: dict[str, Any]) -> float:
    return float(metrics["macro_question_type"]["nll"])


def _option_order_delta(
    scorer: nn.Module, examples: list[dict[str, Any]], device: torch.device
) -> float:
    scorer.eval()
    max_delta = 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), 64):
            batch = examples[start : start + 64]
            sizes = [len(example["options"]) for example in batch]
            features = torch.cat([example["features"] for example in batch])
            reversed_features = torch.cat([example["features"].flip(0) for example in batch])
            base = scorer(features.to(device)).squeeze(-1).cpu().split(sizes)
            reversed_rows = scorer(reversed_features.to(device)).squeeze(-1).cpu().split(sizes)
            for base_row, reversed_row in zip(base, reversed_rows, strict=True):
                delta = float((base_row - reversed_row.flip(0)).abs().max().item())
                max_delta = max(max_delta, delta)
    return max_delta


def _count_model_parameters(model_path: Path) -> int:
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    count = sum(parameter.numel() for _, parameter in model.named_parameters())
    del model
    return count


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _save_state(path: Path, state: dict[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = _args()
    run_started = datetime.now(UTC)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device or config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("only CPU and CUDA are supported")
    sample_dir = Path(config["source"]["sample_dir"]).resolve()
    cache_path = Path(config["source"]["feature_cache"]).resolve()
    text_model_path = Path(config["encoder"]["text_model_path"]).resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse multi-query output directory: {output}")
    if _hash(sample_dir / "train.jsonl") != config["source"]["train_manifest_sha256"]:
        raise ValueError("frozen audio train manifest hash mismatch")
    if _hash(sample_dir / "validation.jsonl") != config["source"]["validation_manifest_sha256"]:
        raise ValueError("frozen audio validation manifest hash mismatch")
    if _hash(cache_path) != config["source"]["feature_cache_sha256"]:
        raise ValueError("frozen audio feature cache hash mismatch")
    weights_path = text_model_path / "model.safetensors"
    if _hash(weights_path) != config["encoder"]["text_weights_sha256"]:
        raise ValueError("frozen MiniLM weight hash mismatch")
    audio_model_path = Path(config["encoder"]["audio_model_path"]).resolve()
    audio_weights_path = audio_model_path / "model.safetensors"
    if _hash(audio_weights_path) != config["encoder"]["audio_weights_sha256"]:
        raise ValueError("frozen Whisper weight hash mismatch")
    audio_parameter_count = _count_model_parameters(audio_model_path)
    if audio_parameter_count != int(config["encoder"]["audio_parameters"]):
        raise ValueError("frozen Whisper parameter count mismatch")

    train_rows = _rows(sample_dir / "train.jsonl")
    validation_rows = _rows(sample_dir / "validation.jsonl")
    _validate_splits(sample_dir, train_rows, validation_rows)
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        if len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError(f"duplicate {split} audio sample IDs")
        if len({str(row["media_sha256"]) for row in rows}) != len(rows):
            raise ValueError(f"duplicate {split} underlying audio assets")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache.get("encoder_revision") != config["encoder"]["audio_revision"]:
        raise ValueError("audio feature cache encoder revision mismatch")
    if cache.get("train_ids") != [str(row["id"]) for row in train_rows]:
        raise ValueError("audio feature cache train ID order mismatch")
    if cache.get("validation_ids") != [str(row["id"]) for row in validation_rows]:
        raise ValueError("audio feature cache validation ID order mismatch")
    train_audio = cache["train_features"].float()
    validation_audio = cache["validation_features"].float()
    if train_audio.shape != (len(train_rows), 384) or validation_audio.shape != (
        len(validation_rows),
        384,
    ):
        raise ValueError("unexpected frozen Whisper feature tensor shapes")
    if not torch.isfinite(train_audio).all() or not torch.isfinite(validation_audio).all():
        raise ValueError("audio feature cache contains non-finite values")

    query_texts = {
        f"Question: {spec['question']}"
        for label in LABELS
        for spec in build_query_specs(label)
    }
    candidate_texts = {
        f"Candidate answer: {option}."
        for label in LABELS
        for spec in build_query_specs(label)
        for option in spec["options"]
    }
    text_vectors, text_parameter_count = _encode_texts(
        sorted(query_texts | candidate_texts), text_model_path, device
    )
    if text_parameter_count != int(config["encoder"]["text_parameters"]):
        raise ValueError("frozen MiniLM parameter count mismatch")
    train_examples = make_query_examples(train_rows, train_audio, text_vectors)
    validation_examples = make_query_examples(validation_rows, validation_audio, text_vectors)
    if len(train_examples) != len(train_rows) * len(QUESTION_TASKS) or len(
        validation_examples
    ) != len(validation_rows) * len(QUESTION_TASKS):
        raise AssertionError("each underlying audio sample must receive every frozen query type")

    if args.preflight_only:
        query_counts = {
            task: {
                "train": sum(example["question_type"] == task for example in train_examples),
                "validation": sum(
                    example["question_type"] == task for example in validation_examples
                ),
            }
            for task in QUESTION_TASKS
        }
        source_counts = {
            "train": {
                label: sum(row["target"] == label for row in train_rows) for label in LABELS
            },
            "validation": {
                label: sum(row["target"] == label for row in validation_rows) for label in LABELS
            },
        }
        print(
            json.dumps(
                {
                    "preflight": "pass",
                    "config_path": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "train_manifest_sha256": _hash(sample_dir / "train.jsonl"),
                    "validation_manifest_sha256": _hash(sample_dir / "validation.jsonl"),
                    "feature_cache_sha256": _hash(cache_path),
                    "audio_weights_sha256": _hash(audio_weights_path),
                    "text_weights_sha256": _hash(weights_path),
                    "device": torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu",
                    "unique_audio_events": {
                        "train": len({row["id"] for row in train_rows}),
                        "validation": len({row["id"] for row in validation_rows}),
                    },
                    "unique_audio_asset_hashes": {
                        "train": len({row["media_sha256"] for row in train_rows}),
                        "validation": len({row["media_sha256"] for row in validation_rows}),
                    },
                    "query_examples": {
                        "train": len(train_examples),
                        "validation": len(validation_examples),
                    },
                    "query_counts_by_type": query_counts,
                    "source_label_counts": source_counts,
                    "train_validation_speaker_overlap": 0,
                    "train_validation_audio_sha256_overlap": 0,
                    "audio_cache_id_order_exact": True,
                    "sealed_audit_loaded": False,
                    "test_split_loaded": False,
                },
                indent=2,
            )
        )
        return

    output.mkdir(parents=True, exist_ok=False)
    effective_config_path = output / "effective-config.yaml"
    effective_config_path.write_bytes(config_path.read_bytes())
    effective_config_sha256 = sha256_file(effective_config_path)

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)
    scorer = CandidateScorer(int(train_examples[0]["features"].shape[-1])).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=float(config["readout"]["learning_rate"]),
        weight_decay=float(config["readout"]["weight_decay"]),
    )
    epochs = int(config["readout"]["epochs"])
    batch_size = int(config["readout"]["batch_queries"])
    history = []
    best_state = None
    best_epoch = 0
    best_nll = float("inf")
    best_checkpoint_path = output / "best-readout.pt"
    history_path = output / "training-history.json"
    update_count = 0
    gradient_norms = []
    train_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_examples)))
        random.Random(seed + epoch).shuffle(order)
        scorer.train()
        batch_losses = []
        for start in range(0, len(order), batch_size):
            batch = [train_examples[index] for index in order[start : start + batch_size]]
            logits, targets = _batch_logits(scorer, batch, device)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite audio multi-query loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in scorer.parameters()
            ):
                raise FloatingPointError(f"non-finite audio multi-query gradient at epoch {epoch}")
            squared_norm = sum(
                parameter.grad.detach().float().pow(2).sum()
                for parameter in scorer.parameters()
                if parameter.grad is not None
            )
            gradient_norms.append(float(squared_norm.sqrt().cpu()))
            optimizer.step()
            update_count += 1
            batch_losses.append(float(loss.detach().cpu()))
        val_metrics, _ = _evaluate(scorer, validation_examples, device, batch_size)
        epoch_record = {
            "epoch": epoch,
            "train_ce": sum(batch_losses) / len(batch_losses),
            "validation": val_metrics,
        }
        history.append(epoch_record)
        _write_json(history_path, history)
        score = _select_validation_nll(val_metrics)
        if score < best_nll:
            best_nll = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
            _save_state(best_checkpoint_path, best_state)
    train_seconds = time.perf_counter() - train_started
    if best_state is None:
        raise RuntimeError("validation checkpoint selection produced no model state")
    final_checkpoint_path = output / "final-readout.pt"
    final_state = {
        key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
    }
    _save_state(final_checkpoint_path, final_state)
    scorer.load_state_dict(best_state)
    inference_started = time.perf_counter()
    final_metrics, predictions = _evaluate(scorer, validation_examples, device, batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    option_delta = _option_order_delta(scorer, validation_examples, device)
    if option_delta > 1e-7:
        raise ValueError(f"option order changed aligned candidate logits: {option_delta}")

    readout_path = best_checkpoint_path
    predictions_path = output / "validation-multiquery-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    _write_json(history_path, history)
    sample_hashes = {
        "train_manifest": sha256_file(sample_dir / "train.jsonl"),
        "validation_manifest": sha256_file(sample_dir / "validation.jsonl"),
        "feature_cache": sha256_file(cache_path),
        "audio_model_weights": config["encoder"]["audio_weights_sha256"],
        "text_model_weights": config["encoder"]["text_weights_sha256"],
    }
    question_counts = {
        task: {
            "train_queries": sum(example["question_type"] == task for example in train_examples),
            "validation_queries": sum(
                example["question_type"] == task for example in validation_examples
            ),
        }
        for task in QUESTION_TASKS
    }
    unique_train = len({row["id"] for row in train_rows})
    unique_validation = len({row["id"] for row in validation_rows})
    report = {
        "schema_version": 1,
        "status": "complete_frozen_audio_multiquery_development_experiment",
        "experiment_id": config["experiment_id"],
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "started_utc": run_started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "effective_config_path": str(effective_config_path),
        "effective_config_sha256": effective_config_sha256,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seed": seed,
        "model": {
            "audio_id": config["encoder"]["audio_id"],
            "audio_revision": config["encoder"]["audio_revision"],
            "audio_weights_sha256": config["encoder"]["audio_weights_sha256"],
            "audio_encoder_parameters": audio_parameter_count,
            "text_id": config["encoder"]["text_id"],
            "text_revision": config["encoder"]["text_revision"],
            "text_weights_sha256": config["encoder"]["text_weights_sha256"],
            "text_encoder_parameters": text_parameter_count,
            "trainable_readout_parameters": trainable_parameters,
            "readout_parameters": sum(parameter.numel() for parameter in scorer.parameters()),
            "full_component_parameter_sum": audio_parameter_count
            + text_parameter_count
            + sum(parameter.numel() for parameter in scorer.parameters()),
            "resident_parameters_during_cached_readout": sum(
                parameter.numel() for parameter in scorer.parameters()
            ),
            "readout_state_dict_sha256": sha256_file(readout_path),
            "final_readout_sha256": sha256_file(final_checkpoint_path),
        },
        "data_hashes": sample_hashes,
        "sample_counts": {
            "unique_train_audio_events": unique_train,
            "unique_validation_audio_events": unique_validation,
            "unique_train_audio_assets": len(
                {row["media_sha256"] for row in train_rows}
            ),
            "unique_validation_audio_assets": len(
                {row["media_sha256"] for row in validation_rows}
            ),
            "train_query_examples": len(train_examples),
            "validation_query_examples": len(validation_examples),
            "unique_train_speakers": len({row["speaker_group_sha256"] for row in train_rows}),
            "unique_validation_speakers": len(
                {row["speaker_group_sha256"] for row in validation_rows}
            ),
            "train_validation_speaker_overlap": 0,
            "train_validation_audio_sha256_overlap": 0,
            "query_counts_by_type": question_counts,
        },
        "cache_reuse": {
            "encoder_executions_this_experiment": 0,
            "validation_unique_audio_events": unique_validation,
            "validation_query_count": len(validation_examples),
            "query_level_encoder_executions_if_uncached": len(validation_examples),
            "query_level_encoder_executions_with_cache": unique_validation,
            "encoder_executions_avoided": len(validation_examples) - unique_validation,
            "queries_per_validation_event": len(QUESTION_TASKS),
            "validation_audio_feature_bytes": validation_audio.nelement()
            * validation_audio.element_size(),
        },
        "training": {
            "epochs": epochs,
            "selected_epoch": best_epoch,
            "selection_rule": config["readout"]["checkpoint_selection"],
            "best_macro_validation_nll": best_nll,
            "train_seconds": train_seconds,
            "optimizer_updates": update_count,
            "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
            "gradient_norm_max": max(gradient_norms),
            "validation_inference_seconds": inference_seconds,
            "validation_ms_per_query": inference_seconds * 1000 / len(validation_examples),
        },
        "validation_metrics": final_metrics,
        "option_order_max_abs_delta_after_alignment": option_delta,
        "validation_prediction_count": len(predictions),
        "validation_predictions_sha256": sha256_file(predictions_path),
        "training_history_sha256": sha256_file(history_path),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "cuda_peak_scope": "readout training/inference only; frozen audio/text features are cached",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
        },
        "test_split_loaded": False,
        "sealed_audit_loaded": False,
        "backbone_parameters_updated": False,
        "note": (
            "The same audio event feature is reused for four deterministic questions. "
            "All validation "
            "items are development data; this is not a blind estimate or general speech QA result."
        ),
    }
    _write_json(output / "run-report.json", report)
    print(
        json.dumps(
            {"status": report["status"], "output_dir": str(output), "report": report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
