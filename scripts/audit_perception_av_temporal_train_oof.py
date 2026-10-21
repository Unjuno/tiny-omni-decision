"""Train-only grouped out-of-fold probe for the frozen Perception AV task.

This diagnostic never scores or uses the existing validation split or its feature files.
It evaluates one fixed 32-update probe budget on deterministic video-group folds;
there is no checkpoint selection or hyperparameter search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.fit_perception_temporal_match_sequence_readouts import (
    AudioCandidateScorer,
    TemporalCandidateScorer,
    categorical_metrics,
)

SEED = 17
FOLDS = 5
UPDATES = 32
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.001


def assign_group_folds(
    rows: list[dict[str, Any]], folds: int = FOLDS, seed: int = SEED
) -> dict[str, int]:
    """Assign each video once, balancing source label and target position."""
    if folds < 2 or len(rows) < folds:
        raise ValueError("at least one unique video group per fold is required")
    by_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row["video_id"])
        if video_id in by_video:
            raise ValueError(f"duplicate training video group: {video_id}")
        target = int(row.get("target_index", row.get("target")))
        if target not in (0, 1):
            raise ValueError("target_index must be binary")
        by_video[video_id] = row

    fold_counts: list[Counter[tuple[int, int]]] = [Counter() for _ in range(folds)]
    total_counts = [0] * folds
    assignment: dict[str, int] = {}
    strata: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in by_video.values():
        strata[(int(row.get("target_index", row.get("target"))), int(row["label_id"]))].append(
            row
        )

    def rank(row: dict[str, Any]) -> str:
        token = f"{seed}\0{row['video_id']}".encode()
        return hashlib.sha256(token).hexdigest()

    for stratum in sorted(strata):
        ordered = sorted(strata[stratum], key=rank)
        for row in ordered:
            fold = min(
                range(folds),
                key=lambda candidate: (
                    fold_counts[candidate][stratum],
                    total_counts[candidate],
                    candidate,
                ),
            )
            video_id = str(row["video_id"])
            assignment[video_id] = fold
            fold_counts[fold][stratum] += 1
            total_counts[fold] += 1
    if set(assignment) != set(by_video) or set(assignment.values()) != set(range(folds)):
        raise RuntimeError("group fold assignment is incomplete")
    return assignment


def _load_feature(path: Path, expected_hash: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"invalid frozen feature shape or values: {path}")
    actual = hashlib.sha256(values.tobytes()).hexdigest()
    if actual != expected_hash:
        raise ValueError(f"frozen feature hash mismatch: {path}")
    return values


def load_train_only(manifest_path: Path, video_dir: Path, audio_dir: Path) -> list[dict[str, Any]]:
    """Load only manifest.train and feature sidecars under train directories."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("train")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest requires nonempty train rows")
    if not video_dir.name.lower() == "train" or not audio_dir.name.lower() == "train":
        raise ValueError("feature directories must be the train split directories")

    video_records: dict[str, tuple[Path, dict[str, Any]]] = {}
    for sidecar in video_dir.glob("*.json"):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("split") != "train":
            raise ValueError("non-train video feature found in train directory")
        sample_id = str(record["sample_id"])
        if sample_id in video_records:
            raise ValueError(f"duplicate video feature: {sample_id}")
        video_records[sample_id] = (sidecar.with_suffix(".npy"), record)

    audio_records: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for sidecar in audio_dir.glob("*.json"):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("split") != "train":
            raise ValueError("non-train audio feature found in train directory")
        key = (str(record["sample_id"]), int(record["event_id"]))
        if key in audio_records:
            raise ValueError(f"duplicate audio feature: {key}")
        audio_records[key] = (sidecar.with_suffix(".npy"), record)

    result = []
    seen_videos: set[str] = set()
    for row in rows:
        sample_id = str(row["sample_id"])
        video_id = str(row["video_id"])
        if video_id in seen_videos:
            raise ValueError(f"duplicate train video group: {video_id}")
        seen_videos.add(video_id)
        if sample_id not in video_records:
            raise ValueError(f"missing train video feature: {sample_id}")
        video_path, video_meta = video_records[sample_id]
        if str(video_meta["video_id"]) != video_id:
            raise ValueError(f"video identity mismatch for {sample_id}")
        video = _load_feature(video_path, video_meta["feature_sha256"], (8, 768))
        options = row["options"]
        if len(options) != 2:
            raise ValueError("expected exactly two audio options")
        audio = []
        for option in options:
            key = (sample_id, int(option["event_id"]))
            if key not in audio_records:
                raise ValueError(f"missing train audio feature: {key}")
            audio_path, audio_meta = audio_records[key]
            audio.append(_load_feature(audio_path, audio_meta["feature_sha256"], (384,)))
        result.append(
            {
                "sample_id": sample_id,
                "video_id": video_id,
                "label_id": int(row["label_id"]),
                "label_name": str(row["label_name"]),
                "target": int(row["target_index"]),
                "video": video,
                "audio": np.stack(audio),
            }
        )
    if set(video_records) != {str(row["sample_id"]) for row in rows}:
        raise ValueError("unexpected or missing train video feature rows")
    expected_audio = {
        (str(row["sample_id"]), int(option["event_id"]))
        for row in rows
        for option in row["options"]
    }
    if set(audio_records) != expected_audio:
        raise ValueError("unexpected or missing train audio feature rows")
    return result


def _prepare(rows: list[dict[str, Any]], mean_std: tuple[np.ndarray, ...]):
    video_mean, video_std, audio_mean, audio_std = mean_std
    video = np.stack([row["video"] for row in rows])
    audio = np.stack([row["audio"] for row in rows])
    target = np.asarray([row["target"] for row in rows], dtype=np.int64)
    return (
        torch.from_numpy(((video - video_mean) / video_std).astype(np.float32)),
        torch.from_numpy(((audio - audio_mean) / audio_std).astype(np.float32)),
        torch.from_numpy(target),
    )


def _normalizer(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    video = np.stack([row["video"] for row in rows])
    audio = np.stack([row["audio"] for row in rows])
    return (
        video.reshape(-1, 768).mean(axis=0),
        video.reshape(-1, 768).std(axis=0).clip(1e-5),
        audio.reshape(-1, 384).mean(axis=0),
        audio.reshape(-1, 384).std(axis=0).clip(1e-5),
    )


def _inventory_hash(directory: Path) -> str:
    entries = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        entries.append(
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def balance_candidate_order(
    rows: list[dict[str, Any]], seed: int = SEED
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Balance positive-option positions within source labels without duplicating rows."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[str(row["label_name"])].append(index)

    balanced = [dict(row) for row in rows]
    counts: dict[str, dict[str, int]] = {}
    for label, indices in sorted(by_label.items()):
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}\0{rows[index]['sample_id']}".encode()
            ).hexdigest(),
        )
        positions: Counter[int] = Counter()
        for rank, index in enumerate(ordered):
            row = rows[index]
            old_target = int(row["target"])
            new_target = rank % 2
            candidates = np.asarray(row["audio"])
            positive = candidates[old_target]
            negative = candidates[1 - old_target]
            balanced[index]["audio"] = (
                np.stack((positive, negative))
                if new_target == 0
                else np.stack((negative, positive))
            )
            balanced[index]["target"] = new_target
            positions[new_target] += 1
        counts[label] = {str(position): positions[position] for position in (0, 1)}
        if abs(positions[0] - positions[1]) > 1:
            raise RuntimeError(f"candidate order is not balanced within label: {label}")
    return balanced, counts


def _score(model: torch.nn.Module, arm: str, data) -> np.ndarray:
    video, audio, _ = data
    model.eval()
    with torch.inference_mode():
        scores = model(audio) if arm == "audio_only" else model(video, audio)
    values = scores.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite OOF scores")
    return values


def run(
    manifest_path: Path,
    video_dir: Path,
    audio_dir: Path,
    output: Path,
    *,
    balance_option_order: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    torch.set_num_threads(1)
    rows = load_train_only(manifest_path, video_dir, audio_dir)
    assignment = assign_group_folds(rows)
    balanced_target_counts = None
    if balance_option_order:
        rows, balanced_target_counts = balance_candidate_order(rows)
    labels = np.asarray([row["target"] for row in rows], dtype=np.int64)
    output.mkdir(parents=True)
    metadata = {
        "status": "RUNNING",
        "protocol": "train-only 5-fold video-group OOF; fixed 32 updates; no checkpoint selection",
        "candidate_order_mode": (
            "balanced within source label; deterministic seed 17"
            if balance_option_order
            else "frozen source manifest order"
        ),
        "option_order_sha256": hashlib.sha256(
            json.dumps(
                [(row["sample_id"], int(row["target"])) for row in rows],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "seed": SEED,
        "folds": FOLDS,
        "updates_per_fold_arm": UPDATES,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_rows_sha256": hashlib.sha256(
            json.dumps(
                json.loads(manifest_path.read_text(encoding="utf-8"))["train"],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "video_train_inventory_sha256": _inventory_hash(video_dir),
        "audio_train_inventory_sha256": _inventory_hash(audio_dir),
        "train_groups": len(rows),
        "optimizer_updates_total": FOLDS
        * len(("audio_only", "pooled_av", "sequence_av"))
        * UPDATES,
        "training_group_visits_total": UPDATES
        * sum(
            len(rows) - sum(assignment[row["video_id"]] == fold for row in rows)
            for fold in range(FOLDS)
        )
        * len(("audio_only", "pooled_av", "sequence_av")),
        "validation_rows_used": False,
        "gpu_used": False,
    }
    if balanced_target_counts is not None:
        metadata["balanced_source_label_target_counts"] = balanced_target_counts
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    arms = ("audio_only", "pooled_av", "sequence_av")
    oof_scores = {arm: np.full((len(rows), 2), np.nan, dtype=np.float64) for arm in arms}
    fold_records = []
    started = time.perf_counter()
    for fold in range(FOLDS):
        train_idx = [i for i, row in enumerate(rows) if assignment[row["video_id"]] != fold]
        test_idx = [i for i, row in enumerate(rows) if assignment[row["video_id"]] == fold]
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        stats = _normalizer(train_rows)
        train_data = _prepare(train_rows, stats)
        test_data = _prepare(test_rows, stats)
        fold_record = {
            "fold": fold,
            "train_groups": len(train_rows),
            "heldout_groups": len(test_rows),
            "heldout_target_counts": dict(Counter(str(row["target"]) for row in test_rows)),
            "heldout_source_label_counts": dict(Counter(row["label_name"] for row in test_rows)),
            "arms": {},
        }
        for arm in arms:
            torch.random.default_generator.manual_seed(SEED)
            model: torch.nn.Module = (
                AudioCandidateScorer()
                if arm == "audio_only"
                else TemporalCandidateScorer(pool_video=(arm == "pooled_av"))
            )
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise RuntimeError("train-only OOF probe must remain CPU-only")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            train_video, train_audio, train_targets = train_data
            for _ in range(UPDATES):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                scores = (
                    model(train_audio) if arm == "audio_only" else model(train_video, train_audio)
                )
                loss = F.cross_entropy(scores, train_targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss in fold {fold}/{arm}")
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise RuntimeError(f"non-finite gradient in fold {fold}/{arm}")
                optimizer.step()
            test_scores = _score(model, arm, test_data)
            oof_scores[arm][test_idx] = test_scores
            fold_record["arms"][arm] = {
                "train_nll": float(
                    F.cross_entropy(
                        model(train_audio)
                        if arm == "audio_only"
                        else model(train_video, train_audio),
                        train_targets,
                    ).detach()
                ),
                "heldout_metrics": {
                    key: value
                    for key, value in categorical_metrics(
                        test_scores, np.asarray([r["target"] for r in test_rows])
                    ).items()
                    if key != "probabilities"
                },
            }
        fold_records.append(fold_record)

    if any(not np.isfinite(scores).all() for scores in oof_scores.values()):
        raise RuntimeError("OOF prediction matrix is incomplete")
    summary = {
        "status": "PASS_TRAIN_ONLY_GROUPED_OOF",
        "protocol": metadata["protocol"],
        "candidate_order_mode": metadata["candidate_order_mode"],
        "option_order_sha256": metadata["option_order_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "train_rows_sha256": metadata["train_rows_sha256"],
        "video_train_inventory_sha256": metadata["video_train_inventory_sha256"],
        "audio_train_inventory_sha256": metadata["audio_train_inventory_sha256"],
        "unique_video_groups": len(rows),
        "target_counts": dict(Counter(str(value) for value in labels)),
        "source_label_target_counts": _source_label_target_counts(rows),
        "balanced_source_label_target_counts": balanced_target_counts,
        "target0_majority_baseline": float(max(Counter(labels).values()) / len(labels)),
        "label_only_oof_baseline": _label_only_baseline(rows, assignment),
        "fold_assignment_sha256": hashlib.sha256(
            json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "arms": {},
        "folds": fold_records,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_rows_used": False,
        "gpu_used": False,
    }
    for arm, scores in oof_scores.items():
        metrics = categorical_metrics(scores, labels)
        summary["arms"][arm] = {
            **{key: value for key, value in metrics.items() if key != "probabilities"},
            "bootstrap_95": _group_bootstrap(scores, labels),
            "per_source_label": _metrics_by_label(scores, rows),
            "predictions": [
                {
                    "sample_id": row["sample_id"],
                    "video_id": row["video_id"],
                    "fold": assignment[row["video_id"]],
                    "source_label": row["label_name"],
                    "target": int(row["target"]),
                    "prediction": int(scores[i].argmax()),
                    "scores": scores[i].tolist(),
                }
                for i, row in enumerate(rows)
            ],
        }
    (output / "oof_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata["status"] = "PASS_TRAIN_ONLY_GROUPED_OOF"
    metadata["elapsed_seconds"] = summary["elapsed_seconds"]
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary


def _label_only_baseline(rows: list[dict[str, Any]], assignment: dict[str, int]) -> dict[str, Any]:
    correct = total = 0
    for fold in range(FOLDS):
        counts: dict[str, Counter[int]] = defaultdict(Counter)
        heldout = []
        for row in rows:
            if assignment[row["video_id"]] == fold:
                heldout.append(row)
            else:
                counts[row["label_name"]][int(row["target"])] += 1
        for row in heldout:
            label_counts = counts[row["label_name"]]
            guess = min((0, 1), key=lambda target: (-label_counts[target], target))
            correct += int(guess == int(row["target"]))
            total += 1
    return {
        "accuracy": correct / total,
        "count": total,
        "input": "source label only; train-fold mode",
    }


def _metrics_by_label(scores: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["label_name"]].append(index)
    result = {}
    for label, indices in groups.items():
        target = np.asarray([rows[i]["target"] for i in indices], dtype=np.int64)
        metrics = categorical_metrics(scores[indices], target)
        result[label] = {key: value for key, value in metrics.items() if key != "probabilities"}
    return result


def _source_label_target_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        counts[row["label_name"]][int(row["target"])] += 1
    return {
        label: {str(target): int(counts_for_label[target]) for target in (0, 1)}
        for label, counts_for_label in sorted(counts.items())
    }


def _group_bootstrap(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Paired row bootstrap; each row is one unique video group in this task."""
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    correct = probabilities.argmax(axis=1) == targets
    nll = -np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-15))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(targets), size=(5_000, len(targets)))
    return {
        "replicates": 5_000,
        "seed": SEED,
        "unit": "video group; one row per group",
        "accuracy_ci_95": [
            float(np.quantile(correct[draws].mean(axis=1), 0.025)),
            float(np.quantile(correct[draws].mean(axis=1), 0.975)),
        ],
        "nll_ci_95": [
            float(np.quantile(nll[draws].mean(axis=1), 0.025)),
            float(np.quantile(nll[draws].mean(axis=1), 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-train-dir", type=Path, required=True)
    parser.add_argument("--audio-train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--balance-candidate-order",
        action="store_true",
        help="Balance positive option position within each source event label.",
    )
    args = parser.parse_args()
    run(
        args.manifest,
        args.video_train_dir,
        args.audio_train_dir,
        args.output,
        balance_option_order=args.balance_candidate_order,
    )


if __name__ == "__main__":
    main()
