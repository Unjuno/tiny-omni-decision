"""Nested train-group CV for the frozen-feature TallyQA-VG Decision probe.

This script uses only the task manifest's TRAIN rows. The official validation,
official test, and sealed audit partitions are never loaded into the fit or
scoring path. Outer image/normalized-question components are scored once;
inner components select checkpoints by NLL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

from scripts.screen_tallyqa_vg_train import normalize_question
from scripts.train_tallyqa_vg_readout import (
    IMAGE_PARAMETERS,
    TEXT_MODEL_SHA256,
    TEXT_PARAMETERS,
    TEXT_REVISION,
    _batch_logits,
    _encode_rows,
    _fit,
    _metrics,
    _pooled_texts,
    _prediction_rows,
)
from tiny_omni_decision.dataset import sha256_file

SEED = 17
FOLDS = 5
EPOCHS = 16
BATCH_SIZE = 64
IMAGE_WIDTH = 768
TEXT_WIDTH = 384


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            low, high = sorted((root_first, root_second))
            self.parent[high] = low


def group_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Join all rows sharing an image or normalized question string."""
    row_keys = [str(row["question_id"]) for row in rows]
    union = _UnionFind(row_keys)
    first_by_image: dict[str, str] = {}
    first_by_question: dict[str, str] = {}
    for row, key in zip(rows, row_keys, strict=True):
        image_id = str(row["image_id"])
        question = normalize_question(str(row["question"]))
        if not question:
            raise ValueError("normalized question must not be empty")
        for mapping, identity in ((first_by_image, image_id), (first_by_question, question)):
            if identity in mapping:
                union.union(key, mapping[identity])
            else:
                mapping[identity] = key
    return [union.find(key) for key in row_keys]


def _group_label_counts(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row, key in zip(rows, keys, strict=True):
        target = int(row["target"])
        options = row["options"]
        if not 0 <= target < len(options):
            raise ValueError("target index is outside the option list")
        answer = str(options[target]["id"])
        counts[key][answer] += 1
    return dict(counts)


def assign_group_folds(
    rows: list[dict[str, Any]], keys: list[str], *, n_folds: int = FOLDS, seed: int = SEED
) -> dict[str, int]:
    """Assign connected groups to balanced folds using targets and stable hashes."""
    if len(rows) != len(keys) or n_folds < 2:
        raise ValueError("invalid row/group counts or fold count")
    group_counts = _group_label_counts(rows, keys)
    groups = sorted(group_counts)
    if len(groups) < n_folds:
        raise ValueError("fewer connected groups than folds")
    totals: Counter[str] = Counter()
    for counts in group_counts.values():
        totals.update(counts)
    labels = sorted(totals)
    target_per_fold = {label: totals[label] / n_folds for label in labels}
    fold_counts = [Counter() for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    fold_group_counts = [0] * n_folds
    ordered = sorted(
        groups,
        key=lambda group: (
            -sum(group_counts[group].values()),
            hashlib.sha256(f"tallyqa-train-group-cv-v1|{seed}|{group}".encode()).digest(),
        ),
    )
    assignments: dict[str, int] = {}
    for group in ordered:
        group_vector = group_counts[group]
        def cost(fold: int, counts: Counter[str] = group_vector) -> tuple[float, int, int]:
            score = 0.0
            for label in labels:
                expected = target_per_fold[label]
                delta = fold_counts[fold][label] + counts[label] - expected
                score += (delta / max(expected, 1.0)) ** 2
            return score, fold_sizes[fold], fold
        empty_folds = [fold for fold, count in enumerate(fold_group_counts) if count == 0]
        if empty_folds:
            selected = min(empty_folds, key=lambda fold: (fold_sizes[fold], fold))
        else:
            selected = min(range(n_folds), key=cost)
        assignments[group] = selected
        fold_counts[selected].update(group_vector)
        fold_sizes[selected] += sum(group_vector.values())
        fold_group_counts[selected] += 1
    if set(assignments) != set(keys):
        raise ValueError("fold assignment did not cover all groups")
    if any(len({assignments[key] for key in keys if key == group}) != 1 for group in groups):
        raise ValueError("connected group was split across folds")
    return assignments


def train_validation_indices(
    rows: list[dict[str, Any]], keys: list[str], *, seed: int
) -> tuple[list[int], list[int], dict[str, int]]:
    """Use one deterministic fold as the inner checkpoint-selection partition."""
    assignment = assign_group_folds(rows, keys, n_folds=5, seed=seed)
    validation_groups = {key for key, fold in assignment.items() if fold == 0}
    train_indices = [i for i, key in enumerate(keys) if key not in validation_groups]
    validation_indices = [i for i, key in enumerate(keys) if key in validation_groups]
    if not train_indices or not validation_indices:
        raise ValueError("inner split produced an empty partition")
    return train_indices, validation_indices, assignment


def _read_train_image_features(
    cache_path: Path, train_rows: list[dict[str, Any]]
) -> dict[str, torch.Tensor]:
    expected: dict[str, str] = {}
    for row in train_rows:
        image_id = str(row["image_id"])
        media_hash = str(row["media_sha256"])
        previous = expected.setdefault(image_id, media_hash)
        if previous != media_hash:
            raise ValueError("one TRAIN image ID has conflicting media hashes")
    with np.load(cache_path, allow_pickle=False) as cache:
        ids = cache["train_ids"].astype(str).tolist()
        media = cache["train_media_sha256"].astype(str).tolist()
        features = cache["train_features"]
    actual = dict(zip(ids, media, strict=True))
    if (
        actual != expected
        or features.shape != (len(ids), IMAGE_WIDTH)
        or not np.isfinite(features).all()
    ):
        raise ValueError("TRAIN image feature identity, shape or finite-value check failed")
    return {
        image_id: torch.from_numpy(value.copy())
        for image_id, value in zip(ids, features, strict=True)
    }


def _hash_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _metrics_for(logits: torch.Tensor, targets: list[int]) -> dict[str, float]:
    if len(logits) != len(targets):
        raise ValueError("logit/target counts differ")
    return _metrics([logits[i] for i in range(len(targets))], targets)


def _group_bootstrap(
    rows: list[dict[str, Any]],
    targets: list[int],
    joint: torch.Tensor,
    question_only: torch.Tensor,
    *,
    replicates: int = 5000,
    seed: int = SEED,
) -> dict[str, Any]:
    row_groups = group_keys(rows)
    groups = sorted(set(row_groups))
    by_group = {
        group: [index for index, value in enumerate(row_groups) if value == group]
        for group in groups
    }
    target_array = np.asarray(targets, dtype=np.int64)
    joint_array = joint.numpy()
    text_array = question_only.numpy()
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {name: [] for name in ("accuracy", "nll", "brier")}
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = [i for group in sampled for i in by_group[str(group)]]
        target = target_array[indices]
        current: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, scores in (("joint", joint_array), ("question", text_array)):
            selected = scores[indices]
            shifted = selected - np.logaddexp.reduce(selected, axis=1, keepdims=True)
            probability = np.exp(shifted)
            accuracy = float((selected.argmax(axis=1) == target).mean())
            nll = float(-shifted[np.arange(len(target)), target].mean())
            one_hot = np.eye(selected.shape[1])[target]
            brier = float(np.square(probability - one_hot).sum(axis=1).mean())
            current[name] = (np.asarray([accuracy, nll, brier]), selected)
        difference = current["joint"][0] - current["question"][0]
        for name, value in zip(deltas, difference, strict=True):
            deltas[name].append(float(value))
    return {
        "unit": "transitive image_id/normalized-question component",
        "replicates": replicates,
        "seed": seed,
        "delta": "joint image+question minus question-only",
        "intervals_95_percentile": {
            metric: np.quantile(values, [0.025, 0.975]).tolist()
            for metric, values in deltas.items()
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze all input hashes and nested group assignments before fitting."""
    config_path = args.config.resolve(strict=True)
    task_path = args.task_manifest.resolve(strict=True)
    cache_path = args.feature_cache.resolve(strict=True)
    report_path = args.cache_report.resolve(strict=True)
    model_path = args.text_model.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    joint_config = config["readouts"]["joint"]
    question_config = config["readouts"]["question_only"]
    if (
        config["seed"] != SEED
        or config["task"]["official_test_opened"] is not False
        or config["encoder"]["checkpoint_sha256"]
        != "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
        or int(joint_config["epochs"]) != EPOCHS
        or int(question_config["epochs"]) != EPOCHS
        or int(joint_config["batch_questions"]) != BATCH_SIZE
        or int(question_config["batch_questions"]) != BATCH_SIZE
        or int(joint_config["hidden_size"]) != 128
        or int(question_config["hidden_size"]) != 128
        or joint_config["optimizer"] != "AdamW"
        or question_config["optimizer"] != "AdamW"
        or float(joint_config["learning_rate"]) != 0.001
        or float(question_config["learning_rate"]) != 0.001
        or float(joint_config["weight_decay"]) != 0.0001
        or float(question_config["weight_decay"]) != 0.0001
    ):
        raise ValueError("config differs from the fixed CV settings")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    rows = task.get("train")
    if (
        task.get("split_scope") != "train_only"
        or any(name in task for name in ("validation", "test", "sealed_audit"))
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("preflight requires a train-only manifest")
    if any(len(row.get("options", [])) != 4 for row in rows):
        raise ValueError("all TRAIN task rows must retain the frozen four-option task")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    text_hash = sha256_file(model_path / "model.safetensors")
    if (
        report.get("status") != "complete_tallyqa_vg_videoprism_features"
        or report.get("cache_sha256") != sha256_file(cache_path)
        or report.get("task_manifest_sha256") != task.get("source_task_manifest_sha256")
        or text_hash != TEXT_MODEL_SHA256
    ):
        raise ValueError("pinned train/cache/model hashes do not agree")
    image_features = _read_train_image_features(cache_path, rows)
    del image_features
    keys = group_keys(rows)
    outer = assign_group_folds(rows, keys, n_folds=FOLDS, seed=SEED)
    folds = []
    for fold in range(FOLDS):
        test_indices = [i for i, key in enumerate(keys) if outer[key] == fold]
        train_indices = [i for i, key in enumerate(keys) if outer[key] != fold]
        train_rows = [rows[i] for i in train_indices]
        train_keys = [keys[i] for i in train_indices]
        inner = assign_group_folds(
            train_rows, train_keys, n_folds=FOLDS, seed=SEED + 1009 * (fold + 1)
        )
        stop_groups = {key for key, value in inner.items() if value == 0}
        fit_groups = set(train_keys) - stop_groups
        score_groups = {keys[i] for i in test_indices}
        if not fit_groups or not stop_groups or (
            fit_groups & stop_groups or fit_groups & score_groups or stop_groups & score_groups
        ):
            raise ValueError("nested grouped split is empty or overlapping")
        folds.append({
            "fold": fold,
            "fit_rows": sum(key in fit_groups for key in train_keys),
            "inner_stop_rows": sum(key in stop_groups for key in train_keys),
            "outer_score_rows": len(test_indices),
            "fit_groups_sha256": _hash_ids(sorted(fit_groups)),
            "inner_stop_groups_sha256": _hash_ids(sorted(stop_groups)),
            "outer_score_groups_sha256": _hash_ids(sorted(score_groups)),
            "inner_assignment_sha256": hashlib.sha256(
                json.dumps(inner, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    ordered_ids = [str(row["question_id"]) for row in rows]
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "UNKNOWN"
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "scripts" / "train_tallyqa_vg_readout.py",
        Path(__file__).resolve().parents[1] / "scripts" / "screen_tallyqa_vg_train.py",
        Path(__file__).resolve().parents[1] / "src" / "tiny_omni_decision" / "decision.py",
    ]
    return {
        "schema_version": 1,
        "protocol": "tallyqa-vg-train-only-nested-group-cv-v1",
        "source_commit": source_commit,
        "source_code_sha256": {path.name: sha256_file(path) for path in code_paths},
        "seed": SEED,
        "outer_folds": FOLDS,
        "inner_folds": FOLDS,
        "inner_stop_fold": 0,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "rows": len(rows),
        "unique_images": len({str(row["image_id"]) for row in rows}),
        "connected_groups": len(set(keys)),
        "ordered_question_ids_sha256": hashlib.sha256(
            "\n".join(ordered_ids).encode("utf-8")
        ).hexdigest(),
        "outer_assignment_sha256": hashlib.sha256(
            json.dumps(outer, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "train_only_manifest_sha256": sha256_file(task_path),
        "source_task_manifest_sha256": task["source_task_manifest_sha256"],
        "config_sha256": sha256_file(config_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "cache_report_sha256": sha256_file(report_path),
        "text_model_sha256": text_hash,
        "text_model_revision": TEXT_REVISION,
        "folds": folds,
        "official_validation_rows_loaded": False,
        "official_test_rows_loaded": False,
        "sealed_audit_rows_loaded": False,
        "checkpoint_selector": "earliest minimum inner-stop NLL",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (
        config["seed"] != SEED
        or config["task"]["official_test_opened"] is not False
        or config["encoder"]["checkpoint_sha256"]
        != "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
    ):
        raise ValueError("config differs from the pinned TallyQA task/encoder")
    task_path = args.task_manifest.resolve(strict=True)
    cache_path = args.feature_cache.resolve(strict=True)
    cache_report_path = args.cache_report.resolve(strict=True)
    text_model_path = args.text_model.resolve(strict=True)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    train_rows = task.get("train")
    if (
        task.get("split_scope") != "train_only"
        or any(key in task for key in ("validation", "test", "sealed_audit"))
        or not isinstance(train_rows, list)
        or not train_rows
    ):
        raise ValueError("runner requires a train-only manifest with no held-out rows")
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    if (
        cache_report.get("status") != "complete_tallyqa_vg_videoprism_features"
        or cache_report.get("cache_sha256") != sha256_file(cache_path)
        or cache_report.get("task_manifest_sha256") != task.get("source_task_manifest_sha256")
        or sha256_file(text_model_path / "model.safetensors") != TEXT_MODEL_SHA256
    ):
        raise ValueError("pinned TRAIN-only input/cache/text-model integrity check failed")
    if args.preflight_receipt is None or args.expected_preflight_sha256 is None:
        raise ValueError("training requires the frozen preflight receipt and expected SHA-256")
    preflight_path = args.preflight_receipt.resolve(strict=True)
    if sha256_file(preflight_path) != args.expected_preflight_sha256:
        raise ValueError("preflight receipt SHA-256 mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    recomputed_preflight = build_preflight(args)
    if preflight != recomputed_preflight:
        raise ValueError("preflight receipt differs from recomputed input/split hashes")
    image_features = _read_train_image_features(cache_path, train_rows)
    unique_texts: dict[str, str] = {}
    for row in train_rows:
        question_id = str(row["question_id"])
        unique_texts[question_id] = str(row["question"])
        for index, option in enumerate(row["options"]):
            unique_texts[f"option:{question_id}:{index}"] = f"Candidate count: {option['text']}."
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
    text_embeddings = _pooled_texts(list(unique_texts.values()), text_model_path, device)
    text_features = dict(zip(unique_texts, text_embeddings, strict=True))
    tasks, targets = _encode_rows(train_rows, image_features, text_features)
    del text_features, text_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()

    groups = group_keys(train_rows)
    outer_assignment = assign_group_folds(train_rows, groups, n_folds=FOLDS, seed=SEED)
    folds: list[dict[str, Any]] = []
    joint_oof = torch.empty((len(train_rows), 4), dtype=torch.float32)
    question_oof = torch.empty_like(joint_oof)
    image_zero_oof = torch.empty_like(joint_oof)
    assigned = torch.zeros(len(train_rows), dtype=torch.bool)
    output.mkdir(parents=True)
    started = time.perf_counter()

    for fold in range(FOLDS):
        test_indices = [i for i, group in enumerate(groups) if outer_assignment[group] == fold]
        outer_train_indices = [
            i for i, group in enumerate(groups) if outer_assignment[group] != fold
        ]
        outer_train_rows = [train_rows[i] for i in outer_train_indices]
        outer_train_groups = [groups[i] for i in outer_train_indices]
        inner_fit_local, inner_stop_local, inner_assignment = train_validation_indices(
            outer_train_rows, outer_train_groups, seed=SEED + 1009 * (fold + 1)
        )
        fit_global = [outer_train_indices[i] for i in inner_fit_local]
        stop_global = [outer_train_indices[i] for i in inner_stop_local]
        group_sets = [
            {groups[i] for i in fit_global},
            {groups[i] for i in stop_global},
            {groups[i] for i in test_indices},
        ]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("nested group split overlap")
        if assigned[test_indices].any():
            raise ValueError("outer row scored more than once")
        fold_record: dict[str, Any] = {
            "fold": fold,
            "fit_rows": len(fit_global),
            "inner_stop_rows": len(stop_global),
            "outer_score_rows": len(test_indices),
            "fit_group_sha256": _hash_ids(sorted(group_sets[0])),
            "inner_stop_group_sha256": _hash_ids(sorted(group_sets[1])),
            "outer_score_group_sha256": _hash_ids(sorted(group_sets[2])),
            "target_count_fit": dict(sorted(Counter(
                str(train_rows[i]["options"][targets[i]]["id"]) for i in fit_global
            ).items())),
            "target_count_inner_stop": dict(sorted(Counter(
                str(train_rows[i]["options"][targets[i]]["id"]) for i in stop_global
            ).items())),
            "target_count_outer_score": dict(sorted(Counter(
                str(train_rows[i]["options"][targets[i]]["id"]) for i in test_indices
            ).items())),
            "inner_fold_group_counts": dict(sorted(Counter(inner_assignment.values()).items())),
            "inner_group_assignment_sha256": hashlib.sha256(
                json.dumps(inner_assignment, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for arm, feature_size, destination in (
            ("joint", IMAGE_WIDTH + 2 * TEXT_WIDTH, joint_oof),
            ("question_only", 2 * TEXT_WIDTH, question_oof),
        ):
            train_tasks = [tasks[i] for i in fit_global]
            stop_tasks = [tasks[i] for i in stop_global]
            if arm == "question_only":
                train_tasks = [value[:, IMAGE_WIDTH:] for value in train_tasks]
                stop_tasks = [value[:, IMAGE_WIDTH:] for value in stop_tasks]
            scorer, fit_summary = _fit(
                train_tasks,
                [targets[i] for i in fit_global],
                validation_tasks=stop_tasks,
                validation_targets=[targets[i] for i in stop_global],
                feature_size=feature_size,
                seed=SEED + fold,
                device=device,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
            )
            score_tasks = [tasks[i] for i in test_indices]
            if arm == "question_only":
                score_tasks = [value[:, IMAGE_WIDTH:] for value in score_tasks]
            with torch.inference_mode():
                logits = torch.cat([
                    _batch_logits(scorer, score_tasks, device,
                                  list(range(start, min(start + BATCH_SIZE, len(score_tasks)))))
                    for start in range(0, len(score_tasks), BATCH_SIZE)
                ])
            destination[test_indices] = logits
            fold_record[arm] = {
                "best_epoch": fit_summary["best_epoch"],
                "training_seconds": fit_summary["training_seconds"],
                "trainable_parameters": fit_summary["trainable_parameters"],
                "inner_stop_metrics": fit_summary["history"][
                    fit_summary["best_epoch"] - 1
                ]["validation"],
                "outer_metrics": _metrics_for(logits, [targets[i] for i in test_indices]),
                "training_history": fit_summary["history"],
            }
            checkpoint = output / f"fold-{fold:02d}-{arm}-selected.pt"
            torch.save(scorer.state_dict(), checkpoint)
            fold_record[arm]["checkpoint_sha256"] = sha256_file(checkpoint)
            if arm == "joint":
                zero_tasks = [value.clone() for value in score_tasks]
                for value in zero_tasks:
                    value[:, :IMAGE_WIDTH] = 0
                with torch.inference_mode():
                    zero_logits = torch.cat([
                        _batch_logits(scorer, zero_tasks, device,
                                      list(range(start, min(start + BATCH_SIZE, len(zero_tasks)))))
                        for start in range(0, len(zero_tasks), BATCH_SIZE)
                    ])
                image_zero_oof[test_indices] = zero_logits
        assigned[test_indices] = True
        folds.append(fold_record)
        _write_json(output / "fold-progress.json", {
            "completed_folds": fold + 1,
            "total_folds": FOLDS,
            "folds": folds,
            "elapsed_seconds": time.perf_counter() - started,
        })

    if not assigned.all():
        raise ValueError("some TRAIN rows lack an outer OOF prediction")
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    metrics = {
        "joint_image_question": _metrics_for(joint_oof, targets),
        "question_only": _metrics_for(question_oof, targets),
        "joint_with_image_zeroed": _metrics_for(image_zero_oof, targets),
        "image_zero_minus_joint_accuracy": float(
            (image_zero_oof.argmax(-1) != targets_tensor).float().mean()
            - (joint_oof.argmax(-1) != targets_tensor).float().mean()
        ),
    }
    cluster_bootstrap = _group_bootstrap(
        train_rows, targets, joint_oof, question_oof, seed=SEED
    )
    prediction_records = {
        "joint": _prediction_rows(train_rows, joint_oof, targets),
        "question_only": _prediction_rows(train_rows, question_oof, targets),
        "image_zero": _prediction_rows(train_rows, image_zero_oof, targets),
    }
    with (output / "oof-predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(train_rows):
            record = {
                "question_id": row["question_id"],
                "image_id": str(row["image_id"]),
                "group_key": groups[index],
                "target_answer": str(row["options"][targets[index]]["id"]),
                "target_option_index": targets[index],
                "fold": outer_assignment[groups[index]],
                "arms": {
                    name: record[index]
                    for name, record in prediction_records.items()
                },
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    result = {
        "status": "complete_train_only_nested_group_cv",
        "experiment": "tallyqa_vg_amt_four_choice_image_question_v1",
        "seed": SEED,
        "outer_folds": FOLDS,
        "inner_stop_folds": 5,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "sample_count": len(train_rows),
        "image_count": len({str(row["image_id"]) for row in train_rows}),
        "connected_group_count": len(set(groups)),
        "connected_grouping": "image id and normalized question string transitive closure",
        "ordered_question_ids_sha256": preflight["ordered_question_ids_sha256"],
        "outer_assignment_sha256": preflight["outer_assignment_sha256"],
        "source_commit": preflight["source_commit"],
        "source_code_sha256": preflight["source_code_sha256"],
        "train_only_manifest_sha256": sha256_file(task_path),
        "source_task_manifest_sha256": task["source_task_manifest_sha256"],
        "feature_cache_sha256": sha256_file(cache_path),
        "cache_report_sha256": sha256_file(cache_report_path),
        "config_sha256": sha256_file(args.config.resolve(strict=True)),
        "text_model_sha256": sha256_file(text_model_path / "model.safetensors"),
        "text_model_revision": TEXT_REVISION,
        "text_model_expected_sha256": TEXT_MODEL_SHA256,
        "parameters_prequant": {
            "videoprism_frozen": IMAGE_PARAMETERS,
            "minilm_frozen": TEXT_PARAMETERS,
            "joint_readout_trainable": folds[0]["joint"]["trainable_parameters"],
            "question_only_readout_trainable": folds[0]["question_only"]["trainable_parameters"],
            "joint_total": (
                IMAGE_PARAMETERS
                + TEXT_PARAMETERS
                + folds[0]["joint"]["trainable_parameters"]
            ),
        },
        "device": str(device),
        "runtime_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "metrics": metrics,
        "image_id_cluster_bootstrap_joint_minus_question_only": cluster_bootstrap,
        "folds": folds,
        "preflight_sha256": sha256_file(preflight_path),
        "elapsed_seconds": time.perf_counter() - started,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "sealed_audit_loaded": False,
        "checkpoints_selected_by": "inner grouped stop-fold NLL; outer fold scored once",
        "limitations": [
            "retrospective TRAIN-only cross-validation on a previously observed task source",
            "group CV evaluates the project-created four-choice adaptation, not original "
            "open-response TallyQA",
            "frozen encoder candidate and settings were informed by earlier development work",
            "does not establish product admission, broad image quality, or multimodal M-2 benefit",
        ],
    }
    _write_json(output / "result.json", result)
    indexed = []
    for artifact in sorted(output.iterdir(), key=lambda item: item.name):
        if artifact.is_file() and artifact.name != "artifact-index.json":
            indexed.append({
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            })
    _write_json(output / "artifact-index.json", {
        "schema_version": 1,
        "artifact_count": len(indexed),
        "total_bytes": sum(item["bytes"] for item in indexed),
        "artifacts": indexed,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
