"""Train and compare one small option head on two frozen EmbeddingGemma 2 backbones."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _processor_inputs_for_missing_options(processor: Any, options: list[str]) -> Any:
    from tiny_omni_decision.student import decision_option_text, processor_inputs_for_options

    if not options:
        raise ValueError("at least one missing option is required")
    if len(options) == 1:
        return processor(text=[decision_option_text(options[0])], return_tensors="pt")
    return processor_inputs_for_options(processor, options)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(sample_ids: list[str]) -> str:
    payload = "\n".join(sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _load_examples(path: Path, *, expected_split: str) -> list[Any]:
    from tiny_omni_decision.schema import DecisionExample

    examples = [
        DecisionExample.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples or any(example.split != expected_split for example in examples):
        raise ValueError(f"{path} must contain non-empty {expected_split} examples only")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError(f"duplicate sample IDs in {path}")
    return examples


def _features_for_examples(
    examples: list[Any],
    option_embeddings: dict[str, Any],
    query_embeddings: list[Any],
    *,
    teacher_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import torch

    from tiny_omni_decision.student import decision_option_text

    queries = torch.stack(query_embeddings).to(dtype=torch.float32).contiguous()
    option_rows: list[Any] = []
    offsets = [0]
    teacher_rows: list[float] = []
    targets: list[int] = []
    for example in examples:
        rows = [option_embeddings[decision_option_text(option)] for option in example.options]
        option_rows.extend(rows)
        offsets.append(len(option_rows))
        targets.append(example.options.index(example.target))
        if teacher_by_id is not None:
            teacher_rows.extend(teacher_by_id[example.id]["teacher_option_probabilities"])
    result = {
        "query_embeddings": queries,
        "option_embeddings": torch.stack(option_rows).to(dtype=torch.float32).contiguous(),
        "option_offsets": torch.tensor(offsets, dtype=torch.int64),
        "targets": torch.tensor(targets, dtype=torch.int64),
    }
    if teacher_by_id is not None:
        result["teacher_probabilities"] = torch.tensor(teacher_rows, dtype=torch.float32)
    return result


def _measure_predictions(
    examples: list[Any], predictions: list[dict[str, Any]], *, ece_bins: int
) -> dict[str, Any]:
    from tiny_omni_decision.student_eval import _measure

    if len(examples) != len(predictions):
        raise ValueError("prediction count differs from validation count")
    grouped: dict[str, list[tuple[list[float], int]]] = defaultdict(list)
    for example, row in zip(examples, predictions, strict=True):
        if row["sample_id"] != example.id or row["target"] != example.options.index(example.target):
            raise ValueError("prediction is not aligned to the fixed validation order")
        probabilities = row["option_probabilities"]
        if len(probabilities) != len(example.options) or not math.isclose(
            sum(probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-6
        ):
            raise ValueError(f"invalid probability vector for {example.id}")
        value = (probabilities, row["target"])
        grouped["all"].append(value)
        grouped[f"modality:{example.modality}"].append(value)
    metrics = {key: _measure(rows, ece_bins=ece_bins) for key, rows in sorted(grouped.items())}
    modalities = [metrics[f"modality:{name}"] for name in ("text", "image", "audio", "video")]
    metrics["macro_modality"] = {
        metric: sum(float(row[metric]) for row in modalities) / len(modalities)
        for metric in ("accuracy", "nll", "brier", "ece", "mean_confidence")
    }
    metrics["macro_modality"]["count"] = len(modalities)
    metrics["minimum_modality_accuracy"] = min(float(row["accuracy"]) for row in modalities)
    return metrics


def _predictions(
    examples: list[Any],
    query_embeddings: Any,
    option_embeddings: Any,
    option_offsets: Any,
    *,
    method: str,
    head: Any | None,
    temperature: float,
    device: Any,
) -> tuple[list[dict[str, Any]], float]:
    import torch

    from tiny_omni_decision.student import supplied_option_logits

    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, example in enumerate(examples):
            start, end = int(option_offsets[index]), int(option_offsets[index + 1])
            query = query_embeddings[index].to(device=device, dtype=torch.float32)
            options = option_embeddings[start:end].to(device=device, dtype=torch.float32)
            logits = (
                supplied_option_logits(query, options, temperature=temperature)
                if method == "cosine"
                else head(query, options)
            )
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
            rows.append(
                {
                    "sample_id": example.id,
                    "source": example.source,
                    "modality": example.modality,
                    "target": example.options.index(example.target),
                    "options": list(example.options),
                    "option_logits": logits.float().cpu().tolist(),
                    "option_probabilities": probabilities,
                    "prediction": prediction,
                    "confidence": probabilities[prediction],
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return rows, time.perf_counter() - started


def _head_latency(
    head: Any, query_embeddings: Any, option_embeddings: Any, offsets: Any, device: Any
) -> dict[str, float]:
    import torch

    count = min(64, int(query_embeddings.shape[0]))
    pairs = []
    for index in range(count):
        start, end = int(offsets[index]), int(offsets[index + 1])
        pairs.append(
            (
                query_embeddings[index].to(device=device, dtype=torch.float32),
                option_embeddings[start:end].to(device=device, dtype=torch.float32),
            )
        )
    with torch.no_grad():
        for query, options in pairs[: min(8, count)]:
            head(query, options)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = []
        for _repeat in range(4):
            for query, options in pairs:
                started = time.perf_counter()
                head(query, options)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed.append(time.perf_counter() - started)
    ordered = sorted(elapsed)
    return {
        "choice_sets_measured": len(elapsed),
        "p50_ms": statistics.median(ordered) * 1000,
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)] * 1000,
    }


def _extract_variant_features(
    variant: str,
    *,
    config: dict[str, Any],
    repo: Path,
    train_examples: list[Any],
    validation_examples: list[Any],
    teacher_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
    progress: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoProcessor

    from tiny_omni_decision.student import (
        decision_option_text,
        model_sentence_embeddings,
        processor_inputs_for_decision_example,
    )
    from tiny_omni_decision.ternary import load_packed_ternary_overlay

    backbone = config["backbone"]
    model_path = Path(backbone["model_path"])
    device = torch.device(config["device"])
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(config["device"])
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    overlay_info = None
    if variant == "ternary":
        overlay_info = load_packed_ternary_overlay(
            model,
            backbone["ternary_overlay_dir"],
            expected_base_model_id=backbone["repo_id"],
            expected_base_revision=backbone["revision"],
        )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters:
        raise RuntimeError("frozen backbone unexpectedly has trainable parameters")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.perf_counter()

    def extract_split_features(
        model: Any, processor: Any, examples: list[Any], *, role: str
    ) -> tuple[list[Any], dict[str, Any]]:
        query_rows: list[Any] = []
        option_cache: dict[str, Any] = {}
        video_frame_cache = (
            {}
            if role == "validation"
            and bool(config["feature_extraction"]["validation_video_frame_cache"])
            else None
        )
        for index, example in enumerate(examples, start=1):
            query_inputs = processor_inputs_for_decision_example(
                processor,
                example,
                data_root=Path(config["data"]["data_root"]),
                video_frame_cache=video_frame_cache,
            )
            with torch.no_grad():
                query = model_sentence_embeddings(model, query_inputs)[0].detach().float().cpu()
            if query.shape != (int(config["head"]["embedding_dim"]),):
                raise ValueError("query encoder output differs from configured 768-D embeddings")
            query_rows.append(query.clone())

            missing_options = [
                option
                for option in example.options
                if decision_option_text(option) not in option_cache
            ]
            if missing_options:
                option_inputs = _processor_inputs_for_missing_options(processor, missing_options)
                with torch.no_grad():
                    embeddings = (
                        model_sentence_embeddings(model, option_inputs).detach().float().cpu()
                    )
                if embeddings.shape != (
                    len(missing_options),
                    int(config["head"]["embedding_dim"]),
                ):
                    raise ValueError(
                        "option encoder output differs from configured 768-D embeddings"
                    )
                option_cache.update(
                    {
                        decision_option_text(option): embedding.clone()
                        for option, embedding in zip(missing_options, embeddings, strict=True)
                    }
                )
            if index % 100 == 0 or index == len(examples):
                print(
                    f"{variant}: {role} examples {index}/{len(examples)} "
                    f"({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
        return query_rows, option_cache

    train_queries, train_option_embeddings = extract_split_features(
        model, processor, train_examples, role="train"
    )
    validation_queries, validation_option_embeddings = extract_split_features(
        model, processor, validation_examples, role="validation"
    )
    train_features = _features_for_examples(
        train_examples, train_option_embeddings, train_queries, teacher_by_id=teacher_by_id
    )
    validation_features = _features_for_examples(
        validation_examples, validation_option_embeddings, validation_queries
    )
    feature_path = output_dir / f"{variant}-frozen-features.safetensors"
    tensors = {
        **{f"train_{key}": value.contiguous().cpu() for key, value in train_features.items()},
        **{
            f"validation_{key}": value.contiguous().cpu()
            for key, value in validation_features.items()
        },
    }
    save_file(
        tensors,
        str(feature_path),
        metadata={
            "variant": variant,
            "base_repo_id": backbone["repo_id"],
            "base_revision": backbone["revision"],
            "overlay_manifest_sha256": (
                config["backbone"]["ternary_overlay_manifest_sha256"]
                if variant == "ternary"
                else "not_applied"
            ),
            "validation_count": str(len(validation_examples)),
        },
    )
    extraction_seconds = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_allocated(device))
    metadata = {
        "variant": variant,
        "base_model_parameter_count": parameter_count,
        "base_trainable_parameter_count": trainable_parameters,
        "option_embedding_count": len(train_option_embeddings) + len(validation_option_embeddings),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_id_order_sha256": _ids_sha256([example.id for example in train_examples]),
        "validation_id_order_sha256": _ids_sha256([example.id for example in validation_examples]),
        "feature_cache_path": str(feature_path),
        "feature_cache_sha256": _sha256(feature_path),
        "feature_cache_bytes": feature_path.stat().st_size,
        "extraction_wall_seconds": extraction_seconds,
        "peak_allocated_vram_bytes": peak_vram,
        "overlay_applied": overlay_info is not None,
        "overlay_tensor_sha256": (
            overlay_info["tensor_file_sha256"] if overlay_info is not None else None
        ),
        "all_backbone_parameters_frozen": trainable_parameters == 0,
        "backbone_gradients_present": any(
            parameter.grad is not None for parameter in model.parameters()
        ),
    }
    if metadata["backbone_gradients_present"]:
        raise RuntimeError("frozen backbone unexpectedly accumulated gradients")
    progress.setdefault("feature_extraction", {})[variant] = metadata
    _atomic_json(output_dir / "progress.json", progress)
    del model, processor, train_option_embeddings, validation_option_embeddings, tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train_features, validation_features, metadata


def _train_and_evaluate_variant(
    variant: str,
    *,
    config: dict[str, Any],
    train_examples: list[Any],
    validation_examples: list[Any],
    train_features: dict[str, Any],
    validation_features: dict[str, Any],
    output_dir: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    from tiny_omni_decision.decision_head import (
        OptionDecisionHead,
        decision_head_parameter_count,
        train_decision_head_one_pass,
    )

    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"])
    head_config = config["head"]
    head = OptionDecisionHead(
        embedding_dim=int(head_config["embedding_dim"]),
        hidden_dim=int(head_config["hidden_dim"]),
    ).to(device=device, dtype=torch.float32)
    train_started = time.perf_counter()
    history_path = output_dir / f"{variant}-training-history.jsonl"
    with history_path.open("w", encoding="utf-8", newline="\n") as history_stream:

        def write_step(row: dict[str, Any]) -> None:
            row["sample_id"] = train_examples[row["update"] - 1].id
            history_stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            if row["update"] % 100 == 0:
                history_stream.flush()
                print(f"{variant}: head updates {row['update']}/{len(train_examples)}", flush=True)

        loss = head_config["loss"]
        training = train_decision_head_one_pass(
            head,
            train_features["query_embeddings"],
            train_features["option_embeddings"],
            train_features["option_offsets"],
            train_features["targets"],
            train_features["teacher_probabilities"],
            learning_rate=float(head_config["learning_rate"]),
            weight_decay=float(head_config["weight_decay"]),
            option_kl_weight=float(loss["option_kl"]),
            cross_entropy_weight=float(loss["cross_entropy"]),
            brier_weight=float(loss["brier"]),
            on_step=write_step,
        )
        history_stream.flush()
    training["wall_seconds"] = time.perf_counter() - train_started
    if training["examples_consumed"] != len(train_examples):
        raise RuntimeError("head training did not consume exactly one full pass")
    if any(not parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("head unexpectedly has frozen parameters")
    head_path = output_dir / f"{variant}-decision-head.safetensors"
    save_file(
        {
            key: value.detach().float().cpu().contiguous()
            for key, value in head.state_dict().items()
        },
        str(head_path),
        metadata={
            "format": "tiny-omni-option-decision-head-v1",
            "embedding_dim": str(head.embedding_dim),
            "hidden_dim": str(head.hidden_dim),
            "activation": "gelu",
            "dtype": "float32",
            "training_examples": str(training["examples_consumed"]),
            "seed": str(seed),
        },
    )
    reloaded = OptionDecisionHead(
        embedding_dim=int(head_config["embedding_dim"]),
        hidden_dim=int(head_config["hidden_dim"]),
    ).to(device=device, dtype=torch.float32)
    reloaded.load_state_dict(load_file(str(head_path), device=str(device)))
    reloaded.eval()
    with torch.no_grad():
        first_query = validation_features["query_embeddings"][0].to(device)
        first_end = int(validation_features["option_offsets"][1])
        first_options = validation_features["option_embeddings"][:first_end].to(device)
        if not torch.allclose(
            head(first_query, first_options),
            reloaded(first_query, first_options),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("saved Decision Head failed its reload check")
    del head
    head = reloaded

    ece_bins = int(config["evaluation"]["ece_bins"])
    cosine_rows, cosine_seconds = _predictions(
        validation_examples,
        validation_features["query_embeddings"],
        validation_features["option_embeddings"],
        validation_features["option_offsets"],
        method="cosine",
        head=None,
        temperature=float(config["evaluation"]["cosine_temperature"]),
        device=device,
    )
    head_rows, head_seconds = _predictions(
        validation_examples,
        validation_features["query_embeddings"],
        validation_features["option_embeddings"],
        validation_features["option_offsets"],
        method="head",
        head=head,
        temperature=1.0,
        device=device,
    )
    cosine_metrics = _measure_predictions(validation_examples, cosine_rows, ece_bins=ece_bins)
    head_metrics = _measure_predictions(validation_examples, head_rows, ece_bins=ece_bins)
    prediction_path = output_dir / f"{variant}-validation-predictions.jsonl"
    head_prediction_path = output_dir / f"{variant}-head-validation-predictions.jsonl"
    _write_jsonl(prediction_path, cosine_rows)
    _write_jsonl(head_prediction_path, head_rows)
    latency = _head_latency(
        head,
        validation_features["query_embeddings"],
        validation_features["option_embeddings"],
        validation_features["option_offsets"],
        device,
    )
    criterion = config["evaluation"]["adoption_rule"]
    macro_cosine = cosine_metrics["macro_modality"]
    macro_head = head_metrics["macro_modality"]
    accuracies_within_rule = all(
        float(head_metrics[f"modality:{name}"]["accuracy"])
        >= float(cosine_metrics[f"modality:{name}"]["accuracy"])
        - 1.0 / int(config["evaluation"]["per_modality_count"])
        for name in ("text", "image", "audio", "video")
    )
    adopted = (
        macro_head["nll"] < macro_cosine["nll"]
        and macro_head["brier"] < macro_cosine["brier"]
        and accuracies_within_rule
    )
    result = {
        "variant": variant,
        "training": training,
        "head": {
            "parameter_count": decision_head_parameter_count(head),
            "parameter_bytes_fp32": decision_head_parameter_count(head) * 4,
            "artifact_path": str(head_path),
            "artifact_sha256": _sha256(head_path),
            "artifact_bytes": head_path.stat().st_size,
            "reload_verified": True,
        },
        "validation": {
            "count": len(validation_examples),
            "id_order_sha256": _ids_sha256([example.id for example in validation_examples]),
            "cosine_metrics": cosine_metrics,
            "head_metrics": head_metrics,
            "delta_head_minus_cosine": {
                key: {
                    metric: float(head_metrics[key][metric]) - float(cosine_metrics[key][metric])
                    for metric in ("accuracy", "nll", "brier", "ece", "mean_confidence")
                }
                for key in (
                    "all",
                    "macro_modality",
                    "modality:text",
                    "modality:image",
                    "modality:audio",
                    "modality:video",
                )
            },
            "cosine_scoring_seconds": cosine_seconds,
            "head_scoring_seconds": head_seconds,
            "head_only_latency": latency,
            "cosine_predictions_path": str(prediction_path),
            "cosine_predictions_sha256": _sha256(prediction_path),
            "head_predictions_path": str(head_prediction_path),
            "head_predictions_sha256": _sha256(head_prediction_path),
        },
        "adoption": {
            "rule": criterion,
            "per_modality_accuracy_within_one_example": accuracies_within_rule,
            "head_improves_macro_nll": macro_head["nll"] < macro_cosine["nll"],
            "head_improves_macro_brier": macro_head["brier"] < macro_cosine["brier"],
            "adopt_head_for_this_backbone": adopted,
        },
    }
    progress.setdefault("training_and_evaluation", {})[variant] = result
    _atomic_json(output_dir / "progress.json", progress)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported Decision Head experiment config")
    repo = Path(__file__).resolve().parents[1]
    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Decision Head output: {output_dir}")
    config["backbone"]["model_manifest"] = str(
        (repo / config["backbone"]["model_manifest"]).resolve()
    )
    config["data"]["validation_manifest"] = str(
        (repo / config["data"]["validation_manifest"]).resolve()
    )
    effective_config_text = yaml.safe_dump(config, sort_keys=False)
    sys.path.insert(0, str(repo / "src"))

    import torch

    from tiny_omni_decision.schema import BaseModelManifest
    from tiny_omni_decision.student_eval import (
        load_fixed_validation_snapshot,
        validate_local_media_paths,
    )
    from tiny_omni_decision.student_evaluate import _verify_local_model_files
    from tiny_omni_decision.student_training import load_teacher_option_cache

    torch.set_num_threads(int(config["head"].get("torch_num_threads", 4)))
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this measured experiment requires the existing local CUDA GPU")
    free_vram, total_vram = torch.cuda.mem_get_info()
    if free_vram < int(config["head"]["minimum_free_vram_bytes"]):
        raise RuntimeError(f"insufficient free VRAM: {free_vram / 1024**3:.2f} GiB")

    backbone = config["backbone"]
    model_manifest_path = Path(backbone["model_manifest"])
    manifest = BaseModelManifest.model_validate(
        yaml.safe_load(model_manifest_path.read_text(encoding="utf-8"))
    )
    if manifest.repo_id != backbone["repo_id"] or manifest.revision != backbone["revision"]:
        raise ValueError("model manifest does not match configured pinned backbone")
    model_file_hashes = _verify_local_model_files(Path(backbone["model_path"]), manifest)
    qat_run_dir = Path(backbone["qat_run_dir"])
    qat_metadata_path = qat_run_dir / "run-metadata.json"
    qat_metadata = json.loads(qat_metadata_path.read_text(encoding="utf-8"))
    shadow_path = Path(backbone["selected_shadow_path"])
    overlay_dir = Path(backbone["ternary_overlay_dir"])
    overlay_manifest_path = overlay_dir / "manifest.json"
    overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
    overlay_tensor_path = overlay_dir / overlay_manifest["tensor_file"]
    integrity_checks = {
        "qat_metadata_sha256": _sha256(qat_metadata_path),
        "selected_shadow_sha256": _sha256(shadow_path),
        "overlay_manifest_sha256": _sha256(overlay_manifest_path),
        "overlay_tensor_sha256": _sha256(overlay_tensor_path),
    }
    expected_checks = {
        "qat_metadata_sha256": backbone["qat_metadata_sha256"],
        "selected_shadow_sha256": backbone["selected_shadow_sha256"],
        "overlay_manifest_sha256": backbone["ternary_overlay_manifest_sha256"],
        "overlay_tensor_sha256": backbone["ternary_overlay_tensor_sha256"],
    }
    if integrity_checks != expected_checks:
        raise ValueError(f"selected QAT artifact hashes changed: {integrity_checks}")
    if (
        qat_metadata.get("state") != "complete"
        or int(qat_metadata.get("best_validation_step", -1)) != int(backbone["selected_step"])
        or qat_metadata.get("best_shadow_sha256") != backbone["selected_shadow_sha256"]
        or overlay_manifest.get("base_model_id") != backbone["repo_id"]
        or overlay_manifest.get("base_revision") != backbone["revision"]
    ):
        raise ValueError("step-512 QAT shadow, overlay, or base model do not align")

    data = config["data"]
    train_path = Path(data["train_corpus"])
    if _sha256(train_path) != data["train_corpus_sha256"]:
        raise ValueError("frozen training corpus hash mismatch")
    corpus_examples = _load_examples(train_path, expected_split="train")
    plan_path = Path(data["train_sampling_plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    order = plan.get("sample_ids")
    if (
        not isinstance(order, list)
        or len(order) != len(corpus_examples)
        or len(set(order)) != len(order)
        or _ids_sha256(order) != data["train_id_order_sha256"]
        or plan.get("sample_id_order_sha256") != data["train_id_order_sha256"]
    ):
        raise ValueError("frozen QAT train sample order failed integrity checks")
    corpus_by_id = {example.id: example for example in corpus_examples}
    if set(corpus_by_id) != set(order):
        raise ValueError("frozen sampling plan does not cover the exact training corpus")
    train_examples = [corpus_by_id[sample_id] for sample_id in order]
    if len(train_examples) != int(qat_metadata["train_examples"]):
        raise ValueError("train example count differs from frozen QAT run")

    cache_path = Path(data["teacher_cache"])
    cache_metadata_path = Path(data["teacher_cache_metadata"])
    if _sha256(cache_path) != data["teacher_cache_sha256"]:
        raise ValueError("frozen Teacher option cache hash mismatch")
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    cache_identity = cache_metadata["identity"]
    if (
        cache_metadata.get("state") != "complete"
        or cache_identity.get("train_corpus_sha256") != data["train_corpus_sha256"]
        or cache_identity.get("teacher_id") != data["teacher_id"]
        or cache_identity.get("teacher_revision") != data["teacher_revision"]
        or float(cache_identity.get("temperature")) != float(data["teacher_temperature"])
        or int(cache_metadata.get("record_count", -1)) != len(train_examples)
    ):
        raise ValueError("frozen Teacher cache metadata does not match training corpus/config")
    teacher_by_id = load_teacher_option_cache(
        cache_path,
        train_examples,
        expected_teacher_id=data["teacher_id"],
        expected_teacher_revision=data["teacher_revision"],
        expected_temperature=float(data["teacher_temperature"]),
    )
    validation_examples, validation_teacher_rows = load_fixed_validation_snapshot(
        data["validation_snapshot"],
        data["validation_manifest"],
        source_validation_path=data["source_validation"],
        teacher_predictions_path=data["teacher_validation_predictions"],
    )
    if len(validation_examples) != int(config["evaluation"]["required_validation_count"]):
        raise ValueError("fixed validation count differs from the declared 256 examples")
    modality_counts = defaultdict(int)
    for example in validation_examples:
        modality_counts[example.modality] += 1
    if dict(modality_counts) != {
        name: int(config["evaluation"]["per_modality_count"])
        for name in ("text", "image", "audio", "video")
    }:
        raise ValueError(
            f"fixed validation does not contain 64 examples per modality: {dict(modality_counts)}"
        )
    media_counts = validate_local_media_paths(
        [*train_examples, *validation_examples], Path(data["data_root"])
    )
    sampling_plan_sha256 = _sha256(plan_path)
    cache_sha256 = _sha256(cache_path)
    progress: dict[str, Any] = {
        "state": "preflight_passed",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "effective_config_sha256": hashlib.sha256(
            effective_config_text.encode("utf-8")
        ).hexdigest(),
        "model_manifest_sha256": _sha256(model_manifest_path),
        "model_files": model_file_hashes,
        "model_parameter_count_from_manifest": manifest.notes.get(
            "loaded_checkpoint_inspection", {}
        ).get("model_parameter_count"),
        "qat_artifact_hashes": integrity_checks,
        "qat_selected_step": qat_metadata["best_validation_step"],
        "train_corpus_sha256": _sha256(train_path),
        "train_examples": len(train_examples),
        "train_unique_examples": len(set(example.id for example in train_examples)),
        "train_id_order_sha256": _ids_sha256([example.id for example in train_examples]),
        "sampling_plan_sha256": sampling_plan_sha256,
        "teacher_cache_sha256": cache_sha256,
        "teacher_cache_metadata_sha256": _sha256(cache_metadata_path),
        "teacher_id": data["teacher_id"],
        "teacher_revision": data["teacher_revision"],
        "teacher_cache_records": len(teacher_by_id),
        "validation_snapshot_sha256": _sha256(Path(data["validation_snapshot"])),
        "validation_manifest_sha256": _sha256(Path(data["validation_manifest"])),
        "validation_examples": len(validation_examples),
        "validation_id_order_sha256": _ids_sha256([example.id for example in validation_examples]),
        "validation_modalities": dict(sorted(modality_counts.items())),
        "validation_teacher_rows_checked": len(validation_teacher_rows),
        "local_media_counts": media_counts,
        "device": config["device"],
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free_bytes_at_start": int(free_vram),
        "gpu_total_bytes": int(total_vram),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "python_version": sys.version,
        "platform": platform.platform(),
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "video_decoder_version": __import__("av").__version__,
        "sealed_audit_accessed": False,
        "backbone_training_or_qat": False,
    }
    print(
        json.dumps(
            {
                "status": "preflight_passed",
                "train_examples": len(train_examples),
                "train_id_order_sha256": progress["train_id_order_sha256"],
                "validation_examples": len(validation_examples),
                "validation_id_order_sha256": progress["validation_id_order_sha256"],
                "validation_modalities": progress["validation_modalities"],
                "local_media_counts": media_counts,
                "qat_selected_step": progress["qat_selected_step"],
                "sealed_audit_accessed": False,
                "gpu_free_bytes": progress["gpu_free_bytes_at_start"],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.preflight_only:
        return
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Decision Head output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    effective_config_path = output_dir / "effective-config.yaml"
    effective_config_path.write_text(effective_config_text, encoding="utf-8")
    _atomic_json(output_dir / "progress.json", progress)

    variants = {}
    for variant in ("ternary", "unquantized"):
        progress["state"] = f"extracting_{variant}_features"
        _atomic_json(output_dir / "progress.json", progress)
        train_features, validation_features, feature_metadata = _extract_variant_features(
            variant,
            config=config,
            repo=repo,
            train_examples=train_examples,
            validation_examples=validation_examples,
            teacher_by_id=teacher_by_id,
            output_dir=output_dir,
            progress=progress,
        )
        progress["state"] = f"training_{variant}_head"
        _atomic_json(output_dir / "progress.json", progress)
        result = _train_and_evaluate_variant(
            variant,
            config=config,
            train_examples=train_examples,
            validation_examples=validation_examples,
            train_features=train_features,
            validation_features=validation_features,
            output_dir=output_dir,
            progress=progress,
        )
        result["feature_extraction"] = feature_metadata
        variants[variant] = result
        del train_features, validation_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_model_bytes = sum(
        path.stat().st_size for path in Path(backbone["model_path"]).rglob("*") if path.is_file()
    )
    overlay_bytes = sum(
        path.stat().st_size
        for path in Path(backbone["ternary_overlay_dir"]).rglob("*")
        if path.is_file()
    )
    head_bytes = {name: row["head"]["artifact_bytes"] for name, row in variants.items()}
    head_parameter_count = next(iter(variants.values()))["head"]["parameter_count"]
    result_payload = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "started_at_utc": progress["started_at_utc"],
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "evidence": progress,
        "training_config": config["head"],
        "storage_and_parameters": {
            "head_parameter_count": head_parameter_count,
            "head_parameter_dtype": "FP32",
            "combined_logical_parameter_count": int(progress["model_parameter_count_from_manifest"])
            + head_parameter_count,
            "head_artifact_bytes_by_backbone": head_bytes,
            "base_model_directory_bytes": base_model_bytes,
            "packed_ternary_overlay_directory_bytes": overlay_bytes,
            "ternary_self_contained_weight_payload_bytes": base_model_bytes
            + overlay_bytes
            + head_bytes["ternary"],
            "unquantized_self_contained_weight_payload_bytes": base_model_bytes
            + head_bytes["unquantized"],
            "overlay_is_not_a_self_contained_model": True,
        },
        "variants": variants,
        "conclusion": {
            "ternary_head_adopted": variants["ternary"]["adoption"]["adopt_head_for_this_backbone"],
            "unquantized_head_adopted": variants["unquantized"]["adoption"][
                "adopt_head_for_this_backbone"
            ],
            "sealed_audit_used": False,
            "follow_on_training_started": False,
        },
    }
    _atomic_json(output_dir / "results.json", result_payload)
    progress["state"] = "complete"
    progress["ended_at_utc"] = result_payload["ended_at_utc"]
    progress["results_sha256"] = _sha256(output_dir / "results.json")
    _atomic_json(output_dir / "progress.json", progress)
    print(f"Complete: {output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
