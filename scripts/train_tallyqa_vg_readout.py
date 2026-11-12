"""Fit/evaluate a frozen-feature Decision readout on the TallyQA VG adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_MODEL_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
IMAGE_PARAMETERS = 114_365_184
TEXT_PARAMETERS = 17_389_824


def _pooled_texts(
    texts: list[str], model_path: Path, device: torch.device, batch_size: int = 128
) -> Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    actual_count = sum(parameter.numel() for parameter in encoder.parameters())
    if actual_count != TEXT_PARAMETERS:
        raise ValueError(f"MiniLM parameter count mismatch: {actual_count}")
    output: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                max_length=96,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            output.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float().cpu())
    del encoder, tokenizer
    return torch.cat(output)


def _cache_features(
    cache_path: Path,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> dict[str, Tensor]:
    with np.load(cache_path, allow_pickle=False) as cache:
        train_ids = cache["train_ids"].astype(str).tolist()
        validation_ids = cache["validation_ids"].astype(str).tolist()
        train_media = cache["train_media_sha256"].astype(str).tolist()
        validation_media = cache["validation_media_sha256"].astype(str).tolist()
        train_features = cache["train_features"]
        validation_features = cache["validation_features"]
        ids = train_ids + validation_ids
        media = train_media + validation_media
        values = np.concatenate([train_features, validation_features])
    expected = {
        str(row["image_id"]): row["media_sha256"] for row in train_rows + validation_rows
    }
    actual_cache = dict(zip(ids, media, strict=True))
    if (
        len(ids) != len(set(ids))
        or len(media) != len(set(media))
        or not set(expected).issubset(actual_cache)
        or any(
            actual_cache.get(image_id) != media_hash
            for image_id, media_hash in expected.items()
        )
        or train_features.shape != (len(train_ids), 768)
        or validation_features.shape != (len(validation_ids), 768)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "feature cache split identity, media hash, shape, or finite-value validation failed"
        )
    wanted = set(expected)
    return {
        image_id: torch.from_numpy(value.copy())
        for image_id, value in zip(ids, values, strict=True)
        if image_id in wanted
    }


def _encode_rows(
    rows: list[dict[str, Any]], image_features: dict[str, Tensor], text_features: dict[str, Tensor]
) -> tuple[list[Tensor], list[int]]:
    candidates: list[Tensor] = []
    targets: list[int] = []
    expected_option_count: int | None = None
    for row in rows:
        question_id = str(row["question_id"])
        image_id = str(row["image_id"])
        if question_id not in text_features or image_id not in image_features:
            raise ValueError("missing question/image feature")
        option_count = len(row["options"])
        if expected_option_count is None:
            expected_option_count = option_count
        elif option_count != expected_option_count:
            raise ValueError("all rows in a readout batch must have the same option count")
        if option_count < 2 or not 0 <= int(row["target"]) < option_count:
            raise ValueError("each row must have at least two options and an in-range target")
        q = text_features[question_id].expand(option_count, -1)
        option = torch.stack(
            [text_features[f"option:{question_id}:{i}"] for i in range(option_count)]
        )
        image = image_features[image_id].expand(option_count, -1)
        candidates.append(torch.cat([image, q, option], dim=-1))
        targets.append(int(row["target"]))
    return candidates, targets


def _batch_logits(
    scorer: nn.Module,
    tasks: list[Tensor],
    device: torch.device,
    batch_indices: list[int] | None = None,
    image_zero: bool = False,
    shuffled_images: dict[str, Tensor] | None = None,
    rows: list[dict[str, Any]] | None = None,
    image_features: dict[str, Tensor] | None = None,
) -> Tensor:
    indices = list(range(len(tasks))) if batch_indices is None else batch_indices
    batch = []
    for index in indices:
        candidate = tasks[index].clone()
        if image_zero:
            candidate[:, :768] = 0
        if shuffled_images is not None and rows is not None:
            image_id = str(rows[index]["image_id"])
            candidate[:, :768] = shuffled_images[image_id]
        batch.append(candidate)
    return scorer(torch.stack(batch).to(device)).cpu()


def _quality(logits: Tensor, targets: list[int]) -> dict[str, float]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    correct = torch.tensor(
        [
            int(int(prob.argmax()) == target)
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    nll = torch.tensor(
        [
            -float(prob[target].clamp_min(1e-12).log())
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    brier = torch.tensor(
        [
            float((prob - F.one_hot(torch.tensor(target), prob.numel())).square().sum())
            for prob, target in zip(probabilities, targets, strict=True)
        ],
        dtype=torch.float32,
    )
    confidence = torch.tensor([float(prob.max()) for prob in probabilities])
    ece = 0.0
    bins = 15
    bin_counts = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        upper = confidence < high if index < bins - 1 else confidence <= high
        mask = (confidence >= low) & upper
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
        "ece_bin_counts": bin_counts,
    }


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float, np.number)):
        return bool(np.isfinite(value))
    return True


def _prediction_rows(
    rows: list[dict[str, Any]], logits: Tensor, targets: list[int]
) -> list[dict[str, Any]]:
    predictions = []
    for row, scores, target in zip(rows, logits, targets, strict=True):
        probabilities = torch.softmax(scores.float(), dim=-1)
        options = [option["id"] for option in row["options"]]
        predictions.append(
            {
                "question_id": row["question_id"],
                "image_id": row["image_id"],
                "target": options[target],
                "prediction": options[int(probabilities.argmax())],
                "options": options,
                "probabilities": probabilities.tolist(),
            }
        )
    return predictions


def _paired_group_bootstrap(
    rows: list[dict[str, Any]],
    targets: list[int],
    baseline: Tensor,
    comparisons: dict[str, Tensor],
    *,
    seed: int = 17,
    replicates: int = 2000,
) -> dict[str, Any]:
    groups = sorted({int(row["image_id"]) for row in rows})
    indices_by_group = {
        group: [index for index, row in enumerate(rows) if int(row["image_id"]) == group]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    baseline_array = baseline.numpy()
    targets_array = np.asarray(targets)
    result: dict[str, Any] = {
        "unit": "validation image_id cluster with all questions retained",
        "seed": seed,
        "replicates": replicates,
        "selection_and_reuse_note": (
            "Validation selected checkpoints and is reused for inference comparisons; "
            "intervals are descriptive, not independent confirmatory inference."
        ),
        "comparisons": {},
    }
    for name, comparison in comparisons.items():
        comparison_array = comparison.numpy()
        accuracy_deltas = []
        nll_deltas = []
        for _ in range(replicates):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            indices = [index for group in sampled_groups for index in indices_by_group[int(group)]]
            y = targets_array[indices]
            base_logits = baseline_array[indices]
            compare_logits = comparison_array[indices]
            base_log_probs = base_logits - np.logaddexp.reduce(base_logits, axis=1, keepdims=True)
            compare_log_probs = compare_logits - np.logaddexp.reduce(
                compare_logits, axis=1, keepdims=True
            )
            base_accuracy = (base_logits.argmax(axis=1) == y).mean()
            compare_accuracy = (compare_logits.argmax(axis=1) == y).mean()
            base_nll = -base_log_probs[np.arange(len(y)), y].mean()
            compare_nll = -compare_log_probs[np.arange(len(y)), y].mean()
            accuracy_deltas.append(float(compare_accuracy - base_accuracy))
            nll_deltas.append(float(compare_nll - base_nll))
        result["comparisons"][name] = {
            "accuracy_delta_95_percentile_interval": np.quantile(
                accuracy_deltas, [0.025, 0.975]
            ).tolist(),
            "nll_delta_95_percentile_interval": np.quantile(nll_deltas, [0.025, 0.975]).tolist(),
            "delta_definition": "comparison minus joint image+question baseline",
        }
    return result


def _fit(
    tasks: list[Tensor],
    targets: list[int],
    *,
    validation_tasks: list[Tensor],
    validation_targets: list[int],
    feature_size: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    scorer = FrozenFeatureCandidateScorer(feature_size, hidden_size=128).to(device)
    count = sum(parameter.numel() for parameter in scorer.parameters())
    expected = (feature_size + 1) * 128 + 129
    if count != expected:
        raise ValueError("unexpected readout parameter count")
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    best_nll = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        indices = list(range(len(tasks)))
        rng.shuffle(indices)
        scorer.train()
        losses = []
        for start in range(0, len(indices), batch_size):
            batch_ids = indices[start : start + batch_size]
            logits = scorer(torch.stack([tasks[i] for i in batch_ids]).to(device))
            loss = F.cross_entropy(
                logits.float(), torch.tensor([targets[i] for i in batch_ids], device=device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scorer.eval()
        with torch.inference_mode():
            validation_logits = torch.cat(
                [
                    _batch_logits(
                        scorer,
                        validation_tasks,
                        device,
                        list(range(i, min(i + batch_size, len(validation_tasks)))),
                    )
                    for i in range(0, len(validation_tasks), batch_size)
                ]
            )
            train_logits = torch.cat(
                [
                    _batch_logits(
                        scorer, tasks, device, list(range(i, min(i + batch_size, len(tasks))))
                    )
                    for i in range(0, len(tasks), batch_size)
                ]
            )
        metrics = _quality(validation_logits, validation_targets)
        history.append(
            {
                "epoch": epoch,
                "train_ce": float(np.mean(losses)),
                "train_metrics": _quality(train_logits, targets),
                "validation": metrics,
            }
        )
        if metrics["nll"] < best_nll:
            best_nll, best_epoch = metrics["nll"], epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("no selected checkpoint")
    scorer.load_state_dict(best_state)
    return scorer, {
        "trainable_parameters": count,
        "best_epoch": best_epoch,
        "checkpoint_selection": "minimum_validation_nll",
        "validation_examples_per_epoch": len(validation_targets),
        "history": history,
        "training_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (
        config["seed"] != 17
        or config["task"]["official_test_opened"] is not False
        or config["encoder"]["checkpoint_sha256"]
        != "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
    ):
        raise ValueError("effective config does not match frozen task/source")
    task_path = args.task_manifest.resolve(strict=True)
    feature_path = args.feature_cache.resolve(strict=True)
    report_path = args.cache_report.resolve(strict=True)
    text_path = args.text_model.resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    task = json.loads(task_path.read_text(encoding="utf-8"))
    cache_task_sha = report.get("feature_cache_source_task_manifest_sha256") or report.get(
        "task_manifest_sha256"
    )
    expected_cache_task_sha = task.get("source_legacy_task_manifest_sha256") or task.get(
        "parent_task_manifest_sha256", sha256_file(task_path)
    )
    if (
        report.get("status") != "complete_tallyqa_vg_videoprism_features"
        or report.get("cache_sha256") != sha256_file(feature_path)
        or cache_task_sha != expected_cache_task_sha
    ):
        raise ValueError("feature cache report/hash mismatch")
    if sha256_file(text_path / "model.safetensors") != TEXT_MODEL_SHA256:
        raise ValueError("MiniLM weights do not match the pinned candidate hash")
    train_rows, val_rows = task["train"], task["validation"]
    option_counts = {len(row["options"]) for row in train_rows + val_rows}
    if (
        len(option_counts) != 1
        or next(iter(option_counts)) != int(config["task"].get("option_count", 4))
        or task["task"].get("id", task["task"].get("name")) != config["task"]["id"]
    ):
        raise ValueError("task ID or option count does not match the effective config")
    option_count = next(iter(option_counts))
    image_features = _cache_features(feature_path, train_rows, val_rows)
    unique_texts: dict[str, str] = {}
    for row in train_rows + val_rows:
        qid = str(row["question_id"])
        unique_texts[qid] = row["question"]
        for index, option in enumerate(row["options"]):
            unique_texts[f"option:{qid}:{index}"] = f"Candidate count: {option['text']}."
    text_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded = _pooled_texts(list(unique_texts.values()), text_path, text_device)
    text_features = dict(zip(unique_texts, encoded, strict=True))
    train_tasks, train_targets = _encode_rows(train_rows, image_features, text_features)
    val_tasks, val_targets = _encode_rows(val_rows, image_features, text_features)
    del text_features
    if text_device.type == "cuda":
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    joint, joint_fit = _fit(
        train_tasks,
        train_targets,
        validation_tasks=val_tasks,
        validation_targets=val_targets,
        feature_size=768 + 384 + 384,
        seed=17,
        device=device,
        epochs=16,
        batch_size=64,
    )
    question_train = [item[:, 768:] for item in train_tasks]
    question_val = [item[:, 768:] for item in val_tasks]
    question_only, question_fit = _fit(
        question_train,
        train_targets,
        validation_tasks=question_val,
        validation_targets=val_targets,
        feature_size=768,
        seed=17,
        device=device,
        epochs=16,
        batch_size=64,
    )
    with torch.inference_mode():
        joint_logits = torch.cat(
            [
                _batch_logits(joint, val_tasks, device, list(range(i, min(i + 64, len(val_tasks)))))
                for i in range(0, len(val_tasks), 64)
            ]
        )
        q_logits = torch.cat(
            [
                _batch_logits(
                    question_only, question_val, device, list(range(i, min(i + 64, len(val_tasks))))
                )
                for i in range(0, len(val_tasks), 64)
            ]
        )
        zero_logits = torch.cat(
            [
                _batch_logits(
                    joint,
                    val_tasks,
                    device,
                    list(range(i, min(i + 64, len(val_tasks)))),
                    image_zero=True,
                )
                for i in range(0, len(val_tasks), 64)
            ]
        )
    # Seeded cyclic derangement moves only the visual observation, never labels/text.
    image_order = sorted(
        {str(row["image_id"]) for row in val_rows},
        key=lambda value: hashlib.sha256(f"17:{value}".encode()).digest(),
    )
    if len(image_order) < 2:
        raise ValueError("validation requires at least two unique images for image shuffle")
    image_features_val = {key: image_features[key] for key in image_order}
    shuffled = {
        image_order[i]: image_features_val[image_order[(i + 1) % len(image_order)]]
        for i in range(len(image_order))
    }
    with torch.inference_mode():
        shuffled_logits = torch.cat(
            [
                _batch_logits(
                    joint,
                    val_tasks,
                    device,
                    list(range(i, min(i + 64, len(val_tasks)))),
                    shuffled_images=shuffled,
                    rows=val_rows,
                    image_features=image_features,
                )
                for i in range(0, len(val_tasks), 64)
            ]
        )
    output.mkdir(parents=True)
    joint_path, question_path = (
        output / "best-joint-readout.pt",
        output / "best-question-only-readout.pt",
    )
    torch.save(joint.state_dict(), joint_path)
    torch.save(question_only.state_dict(), question_path)
    joint.load_state_dict(torch.load(joint_path, map_location=device, weights_only=True))
    question_only.load_state_dict(torch.load(question_path, map_location=device, weights_only=True))
    with torch.inference_mode():
        reloaded = torch.cat(
            [
                _batch_logits(joint, val_tasks, device, list(range(i, min(i + 64, len(val_tasks)))))
                for i in range(0, len(val_tasks), 64)
            ]
        )
    if not torch.allclose(reloaded, joint_logits, atol=1e-6, rtol=1e-6):
        raise ValueError("best checkpoint reload predictions differ")
    metric_set = {
        "joint_image_question": _quality(joint_logits, val_targets),
        "question_only": _quality(q_logits, val_targets),
        "joint_checkpoint_image_zero": _quality(zero_logits, val_targets),
        "joint_checkpoint_seed17_shuffled_image": _quality(shuffled_logits, val_targets),
    }
    bootstrap = _paired_group_bootstrap(
        val_rows,
        val_targets,
        joint_logits,
        {
            "question_only": q_logits,
            "image_zero": zero_logits,
            "image_shuffled": shuffled_logits,
        },
    )
    predictions = {
        name: _prediction_rows(val_rows, logits, val_targets)
        for name, logits in (
            ("joint", joint_logits),
            ("question_only", q_logits),
            ("image_zero", zero_logits),
            ("image_shuffled", shuffled_logits),
        )
    }
    if not _finite_tree(metric_set):
        raise ValueError("non-finite quality metric")
    result = {
        "status": "validation_complete_development_only",
        "config": config,
        "config_sha256": sha256_file(args.config),
        "task_manifest_sha256": sha256_file(task_path),
        "feature_cache_sha256": sha256_file(feature_path),
        "feature_cache_report_sha256": sha256_file(report_path),
        "text_model_revision": TEXT_REVISION,
        "text_model_sha256": TEXT_MODEL_SHA256,
        "parameter_count": {
            "video_prism_frozen": IMAGE_PARAMETERS,
            "minilm_frozen": TEXT_PARAMETERS,
            "joint_readout_trainable": joint_fit["trainable_parameters"],
            "question_only_readout_trainable": question_fit["trainable_parameters"],
            "joint_total_prequant": IMAGE_PARAMETERS
            + TEXT_PARAMETERS
            + joint_fit["trainable_parameters"],
        },
        "training": {"joint": joint_fit, "question_only": question_fit},
        "validation_count": len(val_rows),
        "validation_unique_images": len(image_order),
        "validation_option_target_position_counts": {
            str(i): sum(row["target"] == i for row in val_rows) for i in range(option_count)
        },
        "metrics": metric_set,
        "paired_image_group_bootstrap": bootstrap,
        "predictions": predictions,
        "artifacts": {
            "joint_state_dict_sha256": sha256_file(joint_path),
            "question_only_state_dict_sha256": sha256_file(question_path),
        },
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "runtime_versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
        },
        "sealed_audit_loaded": False,
        "limitations": [
            "single seed",
            "one question-conditioned natural-image counting adaptation",
            f"{option_count}-choice conversion differs from original open-response task",
            "validation used for checkpoint selection",
            "no claim about video temporal reasoning or general multimodal quality",
        ],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2)
    )


if __name__ == "__main__":
    main()
