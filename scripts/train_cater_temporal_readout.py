"""Fit frozen-encoder temporal Decision readouts on the fixed CATER split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_omni_decision.cater_temporal_probe import CaterTemporalDecisionHead  # noqa: E402
from tiny_omni_decision.clevr_vqa_readout import macro_by_kind, quality_metrics

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
TEXT_WEIGHTS_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
VIDEOPRISM_PARAMETERS = 114_365_184
MAX_EPOCHS = 12
PATIENCE = 4
BATCH_GROUPS = 4
BOOTSTRAP_SEED = 1701
BOOTSTRAP_REPETITIONS = 2000
FULL_TRAIN_QUERY_COUNT = 4360
FULL_VALIDATION_QUERY_COUNT = 1120


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def encode_texts(texts: list[str], tokenizer: Any, encoder: nn.Module) -> dict[str, Tensor]:
    unique = list(dict.fromkeys(texts))
    vectors = []
    with torch.no_grad():
        for start in range(0, len(unique), 64):
            batch = tokenizer(
                unique[start : start + 64],
                max_length=64,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            output = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
            vectors.extend(((output * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu())
    return {text: vector for text, vector in zip(unique, vectors, strict=True)}


def prepare_examples(
    rows: list[dict[str, Any]], embeddings: dict[str, Tensor]
) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        options = row["options"]
        target_ids = [
            index for index, option in enumerate(options) if option["id"] == row["target_option_id"]
        ]
        if len(target_ids) != 1:
            raise ValueError(f"target option is missing or duplicated: {row['id']}")
        start, end = map(int, row["frame_interval"])
        if start < 0 or end <= start:
            raise ValueError(f"invalid task interval: {row['id']}")
        prepared.append(
            {
                "row": row,
                "question": embeddings[row["question"]],
                "options": torch.stack([embeddings[option["text"]] for option in options]),
                "target": target_ids[0],
            }
        )
    return prepared


def group_by_video(examples: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[str(example["row"]["video_id"])].append(example)
    return list(groups.items())


def require_full_cohort(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    expected_train_count: int = FULL_TRAIN_QUERY_COUNT,
    expected_validation_count: int = FULL_VALIDATION_QUERY_COUNT,
) -> None:
    if len(train_rows) != expected_train_count or len(validation_rows) != expected_validation_count:
        raise ValueError(
            "full readout requires the frozen "
            f"{expected_train_count}/{expected_validation_count} query cohort; got "
            f"{len(train_rows)}/{len(validation_rows)}"
        )


def load_video_feature(
    path: Path, expected_media_sha256: str, expected_feature_sha256: str,
    expected_sample_id: str | None = None, expected_interval: list[int] | None = None,
) -> Tensor:
    with np.load(path, allow_pickle=False) as archive:
        values = archive["features"]
        metadata = json.loads(str(archive["metadata"].item()))
    if (
        metadata.get("media_sha256") != expected_media_sha256
        or (expected_sample_id is not None and metadata.get("sample_id") != expected_sample_id)
        or (expected_interval is not None and metadata.get("frame_interval") != expected_interval)
        or metadata.get("feature_sha256") != expected_feature_sha256
        or hashlib.sha256(values.tobytes()).hexdigest() != expected_feature_sha256
        or values.shape != (13, 256, 768)
        or values.dtype != np.float16
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"invalid frozen VideoPrism feature file: {path}")
    return torch.from_numpy(values.astype(np.float32))


def feature_record_for_example(
    feature_records: dict[tuple[str, str], dict[str, Any]],
    interval_records: dict[tuple[str, str, tuple[int, int]], dict[str, Any]],
    split: str,
    example: dict[str, Any],
    video_id: str | None = None,
) -> dict[str, Any]:
    row = example["row"]
    if video_id is None:
        return feature_records[(split, str(row["id"]))]
    interval = tuple(map(int, row["frame_interval"]))
    return interval_records[(split, video_id, interval)]


def _feature_for_mode(video: Tensor, mode: str, seed: int, interval: tuple[int, int]) -> Tensor:
    if mode == "video":
        return video
    if mode == "text_only":
        return torch.zeros_like(video)
    start, end = interval
    if mode == "static_first":
        return video[start : start + 1].expand_as(video)
    if mode == "shuffle_interior":
        changed = video.clone()
        interior = changed[start + 1 : end]
        if interior.shape[0] > 1:
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(interior.shape[0], generator=generator)
            changed[start + 1 : end] = interior[order]
        return changed
    if mode == "reverse_interval":
        changed = video.clone()
        changed[start : end + 1] = changed[start : end + 1].flip(0)
        return changed
    raise ValueError(f"unknown feature control: {mode}")


def _batch_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot score an empty prediction list")
    overall_probs = torch.zeros((len(records), max(len(row["probabilities"]) for row in records)))
    targets = []
    for index, row in enumerate(records):
        probabilities = torch.tensor(row["probabilities"], dtype=torch.float32)
        overall_probs[index, : len(probabilities)] = probabilities
        targets.append(int(row["target_index"]))
    overall = quality_metrics(overall_probs, torch.tensor(targets))
    by_interval: dict[str, dict[str, Any]] = {}
    for interval in sorted({str(row["interval"]) for row in records}):
        selected = [row for row in records if str(row["interval"]) == interval]
        probs = torch.zeros((len(selected), max(len(row["probabilities"]) for row in selected)))
        labels = []
        for index, row in enumerate(selected):
            values = torch.tensor(row["probabilities"], dtype=torch.float32)
            probs[index, : len(values)] = values
            labels.append(int(row["target_index"]))
        by_interval[interval] = quality_metrics(probs, torch.tensor(labels))
    return {
        "sample_weighted": overall,
        "by_interval": by_interval,
        "macro_by_interval": macro_by_kind(by_interval),
    }


def _predict(
    head: CaterTemporalDecisionHead,
    groups: list[tuple[str, list[dict[str, Any]]]],
    feature_records: dict[tuple[str, str], dict[str, Any]],
    interval_records: dict[tuple[str, str, tuple[int, int]], dict[str, Any]],
    feature_root: Path,
    split: str,
    *,
    feature_mode: str,
    seed: int,
    video_permutation: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    head.eval()
    results = []
    with torch.inference_mode():
        for video_id, examples in groups:
            current_id = (
                video_permutation.get(video_id, video_id) if video_permutation else video_id
            )
            for example in examples:
                row = example["row"]
                record = feature_record_for_example(
                    feature_records,
                    interval_records,
                    split,
                    example,
                    current_id if video_permutation else None,
                )
                interval = list(map(int, row["frame_interval"]))
                video = load_video_feature(
                    feature_root / str(record["feature_path"]),
                    str(record["media_sha256"]),
                    str(record["feature_sha256"]),
                    str(record["sample_id"]),
                    list(interval),
                )
                start, end = 0, video.shape[0] - 1
                values = _feature_for_mode(video, feature_mode, seed, (start, end))
                logits = head(
                    values,
                    example["question"],
                    example["options"],
                    start,
                    end,
                )
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
                prediction_index = int(np.argmax(probabilities))
                results.append(
                    {
                        "id": row["id"],
                        "video_id": video_id,
                        "interval": row["frame_interval"],
                        "options": [option["id"] for option in row["options"]],
                        "target_index": int(example["target"]),
                        "target": row["target_option_id"],
                        "prediction": row["options"][prediction_index]["id"],
                        "probabilities": probabilities,
                        "correct": prediction_index == int(example["target"]),
                    }
                )
    return results


def _train_condition(
    condition: str,
    train_groups: list[tuple[str, list[dict[str, Any]]]],
    validation_groups: list[tuple[str, list[dict[str, Any]]]],
    feature_records: dict[tuple[str, str], dict[str, Any]],
    interval_records: dict[tuple[str, str, tuple[int, int]], dict[str, Any]],
    feature_root: Path,
    *,
    seed: int,
    output_dir: Path,
    batch_groups: int = BATCH_GROUPS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    head = CaterTemporalDecisionHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = torch.randperm(len(train_groups), generator=generator).tolist()
        epoch_losses = []
        for offset in range(0, len(order), batch_groups):
            selected_groups = [
                train_groups[index] for index in order[offset : offset + batch_groups]
            ]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _video_id, examples in selected_groups:
                for example in examples:
                    row = example["row"]
                    record = feature_records[("train", str(row["id"]))]
                    video = load_video_feature(
                        feature_root / str(record["feature_path"]),
                        str(record["media_sha256"]),
                        str(record["feature_sha256"]),
                        str(record["sample_id"]),
                        list(map(int, row["frame_interval"])),
                    )
                    start, end = 0, video.shape[0] - 1
                    if condition == "text_only":
                        values = torch.zeros_like(video)
                    else:
                        values = video
                    logits = head(
                        values,
                        example["question"],
                        example["options"],
                        start,
                        end,
                    )
                    target = torch.tensor([example["target"]], dtype=torch.long)
                    losses.append(F.cross_entropy(logits[None, :], target))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {condition} training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        validation_predictions = _predict(
            head,
            validation_groups,
            feature_records,
            interval_records,
            feature_root,
            "validation",
            feature_mode=condition,
            seed=seed,
        )
        validation_metrics = _batch_metrics(validation_predictions)
        score = float(validation_metrics["macro_by_interval"]["nll"])
        history.append(
            {
                "epoch": epoch,
                "train_ce": float(np.mean(epoch_losses)),
                "validation": validation_metrics,
                "selection_score_macro_interval_nll": score,
            }
        )
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"validation failed to select a {condition} readout")
    head.load_state_dict(best_state)
    checkpoint_path = output_dir / f"best-{condition}-readout.pt"
    torch.save(best_state, checkpoint_path)
    selected_validation = _predict(
        head,
        validation_groups,
        feature_records,
        interval_records,
        feature_root,
        "validation",
        feature_mode=condition,
        seed=seed,
    )
    train_predictions = _predict(
        head,
        train_groups,
        feature_records,
        interval_records,
        feature_root,
        "train",
        feature_mode=condition,
        seed=seed,
    )
    return {
        "condition": condition,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_macro_interval_nll": best_score,
        "history": history,
        "train_metrics_at_selected_checkpoint": _batch_metrics(train_predictions),
        "validation_metrics_at_selected_checkpoint": _batch_metrics(selected_validation),
        "validation_predictions": selected_validation,
        "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_seconds": time.perf_counter() - started,
    }


def _cluster_bootstrap_accuracy_delta(
    video_predictions: list[dict[str, Any]],
    text_predictions: list[dict[str, Any]],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    text_by_id = {row["id"]: row for row in text_predictions}
    if {row["id"] for row in video_predictions} != set(text_by_id):
        raise ValueError(
            "paired video and text-only predictions must cover identical validation IDs"
        )
    groups = sorted({row["video_id"] for row in video_predictions})
    by_group: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for row in video_predictions:
        other = text_by_id[row["id"]]
        interval = int(row["interval"][0]) // 75
        by_group[row["video_id"]].append((interval, float(row["correct"]), float(other["correct"])))
    group_values = np.full((len(groups), 4, 2), np.nan, dtype=np.float64)
    for index, group in enumerate(groups):
        for interval in range(4):
            values = [row for row in by_group[group] if row[0] == interval]
            if values:
                group_values[index, interval, 0] = np.mean([row[1] for row in values])
                group_values[index, interval, 1] = np.mean([row[2] for row in values])
    observed_intervals = np.flatnonzero(~np.isnan(group_values[:, :, 0]).all(axis=0))
    if not len(observed_intervals):
        raise ValueError("paired bootstrap has no observed validation intervals")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(repetitions):
        sample = rng.integers(0, len(groups), size=len(groups))
        means = np.nanmean(group_values[sample][:, observed_intervals], axis=0)
        deltas.append(float(np.mean(means[:, 0] - means[:, 1])))
    point = float(
        _batch_metrics(video_predictions)["macro_by_interval"]["accuracy"]
        - _batch_metrics(text_predictions)["macro_by_interval"]["accuracy"]
    )
    return {
        "macro_accuracy_delta_video_minus_text_only": point,
        "scene_group_paired_bootstrap_95pct": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "scene_groups": len(groups),
        "repetitions": repetitions,
        "seed": BOOTSTRAP_SEED,
        "method": "paired percentile bootstrap, resample video groups, macro-average intervals",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-groups", type=int, default=BATCH_GROUPS)
    parser.add_argument("--max-video-groups", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    manifest_dir = args.manifest_dir.resolve(strict=True)
    feature_dir = args.feature_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    text_model_path = args.text_model.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite readout output: {output}")
    if sha256_file(text_model_path / "model.safetensors") != TEXT_WEIGHTS_SHA256:
        raise ValueError("MiniLM checkpoint SHA-256 mismatch")
    result_path = feature_dir / "result.json"
    feature_result = json.loads(result_path.read_text(encoding="utf-8"))
    if feature_result.get("status") != "PASS_CATER_FROZEN_VIDEOPRISM_FEATURES":
        raise ValueError("complete verified CATER VideoPrism features are required")
    train_rows = read_jsonl(manifest_dir / "train.jsonl")
    validation_rows = read_jsonl(manifest_dir / "validation.jsonl")
    if feature_result["manifest_sha256"]["train"] != sha256_file(manifest_dir / "train.jsonl"):
        raise ValueError("train manifest differs from the feature-extraction run")
    if feature_result["manifest_sha256"]["validation"] != sha256_file(
        manifest_dir / "validation.jsonl"
    ):
        raise ValueError("validation manifest differs from the feature-extraction run")
    if args.max_video_groups is None:
        require_full_cohort(train_rows, validation_rows)
    train_video_ids = {str(row["video_id"]) for row in train_rows}
    validation_video_ids = {str(row["video_id"]) for row in validation_rows}
    if train_video_ids & validation_video_ids:
        raise ValueError("train and validation videos overlap")
    feature_records = {
        (str(row["split"]), str(row["sample_id"])): row for row in feature_result["records"]
    }
    required_sample_keys = {
        (str(row["split"]), str(row["id"])) for row in train_rows + validation_rows
    }
    if set(feature_records) != required_sample_keys:
        raise ValueError("feature run does not match all train/validation sample identities")
    interval_records = {
        (str(row["split"]), str(row["video_id"]), tuple(map(int, row["frame_interval"]))): row
        for row in feature_result["records"]
    }
    if len(interval_records) != len(feature_records):
        raise ValueError("each video must have at most one feature per task interval")

    torch.set_num_threads(args.cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
    text_encoder = AutoModel.from_pretrained(text_model_path, local_files_only=True).eval().cpu()
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_parameter_count = sum(parameter.numel() for parameter in text_encoder.parameters())
    if text_parameter_count != 17_389_824:
        raise ValueError(f"unexpected MiniLM parameter count: {text_parameter_count}")
    all_rows = train_rows + validation_rows
    texts = [row["question"] for row in all_rows]
    texts.extend(option["text"] for row in all_rows for option in row["options"])
    embeddings = encode_texts(texts, tokenizer, text_encoder)
    del tokenizer, text_encoder
    train_examples = prepare_examples(train_rows, embeddings)
    validation_examples = prepare_examples(validation_rows, embeddings)
    train_groups = group_by_video(train_examples)
    validation_groups = group_by_video(validation_examples)
    if args.max_video_groups is not None:
        train_groups = train_groups[: args.max_video_groups]
        validation_groups = validation_groups[: args.max_video_groups]
    output.mkdir(parents=True)
    run_started = datetime.now(UTC).isoformat()
    video_result = _train_condition(
        "video",
        train_groups,
        validation_groups,
        feature_records,
        interval_records,
        feature_dir / "features",
        seed=args.seed,
        output_dir=output,
        batch_groups=args.batch_groups,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    text_result = _train_condition(
        "text_only",
        train_groups,
        validation_groups,
        feature_records,
        interval_records,
        feature_dir / "features",
        seed=args.seed,
        output_dir=output,
        batch_groups=args.batch_groups,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    # Controls are post-selection diagnostics on the fixed development validation set.
    video_head = CaterTemporalDecisionHead()
    video_head.load_state_dict(
        torch.load(video_result["checkpoint_path"], map_location="cpu", weights_only=True)
    )
    control_predictions = {}
    for mode in ("text_only", "static_first", "shuffle_interior", "reverse_interval"):
        control_predictions[mode] = _predict(
            video_head,
            validation_groups,
            feature_records,
            interval_records,
            feature_dir / "features",
            "validation",
            feature_mode=mode,
            seed=args.seed,
        )
    validation_video_ids_sorted = sorted(validation_video_ids)
    permutation = list(validation_video_ids_sorted)
    random.Random(args.seed).shuffle(permutation)
    if len(permutation) > 1 and permutation == validation_video_ids_sorted:
        permutation = permutation[1:] + permutation[:1]
    control_predictions["permuted_video"] = _predict(
        video_head,
        validation_groups,
        feature_records,
        interval_records,
        feature_dir / "features",
        "validation",
        feature_mode="video",
        seed=args.seed,
        video_permutation=dict(zip(validation_video_ids_sorted, permutation, strict=True)),
    )
    bootstrap = _cluster_bootstrap_accuracy_delta(
        video_result["validation_predictions"], text_result["validation_predictions"]
    )

    heads = CaterTemporalDecisionHead()
    trainable = sum(parameter.numel() for parameter in heads.parameters())
    repo_root = Path(__file__).resolve().parents[1]
    project_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "status": "PASS_CATER_TEMPORAL_FROZEN_PATH_A_READOUT",
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": project_commit,
        "working_tree_had_changes": bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "code_sha256": {
            "readout_script": sha256_file(Path(__file__).resolve()),
            "head_module": sha256_file(
                repo_root / "src" / "tiny_omni_decision" / "cater_temporal_probe.py"
            ),
        },
        "seed": args.seed,
        "task": (
            "select the object with greatest annotated net displacement over the "
            "supplied 75-frame interval"
        ),
        "training_queries": sum(len(rows) for _, rows in train_groups),
        "validation_queries": sum(len(rows) for _, rows in validation_groups),
        "unique_training_video_groups": len(train_groups),
        "unique_validation_video_groups": len(validation_groups),
        "train_manifest_sha256": sha256_file(manifest_dir / "train.jsonl"),
        "validation_manifest_sha256": sha256_file(manifest_dir / "validation.jsonl"),
        "feature_result_sha256": sha256_file(result_path),
        "text_model_revision": TEXT_REVISION,
        "text_model_weights_sha256": TEXT_WEIGHTS_SHA256,
        "text_model_files": artifact_file_manifest(text_model_path),
        "text_model_parameters": text_parameter_count,
        "videoprism_parameters": VIDEOPRISM_PARAMETERS,
        "head_parameters": trainable,
        "total_unquantized_resident_parameters": VIDEOPRISM_PARAMETERS
        + text_parameter_count
        + trainable,
        "training_setup": {
            "encoder_weights_frozen": True,
            "trainable_module": (
                "question/option-conditioned spatial attention + 1-layer GRU + candidate scorer"
            ),
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "batch_groups": args.batch_groups,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "selection_rule": "lowest macro-over-interval validation NLL; min_delta=1e-4",
        },
        "video_condition": {
            key: value for key, value in video_result.items() if key != "validation_predictions"
        },
        "text_only_condition": {
            key: value for key, value in text_result.items() if key != "validation_predictions"
        },
        "controls_validation_metrics": {
            mode: _batch_metrics(predictions) for mode, predictions in control_predictions.items()
        },
        "video_vs_text_only_paired_scene_bootstrap": bootstrap,
        "training_device": "CPU",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__,
        },
        "validation_scope": (
            "fixed development selector; checkpoint selected on this validation; not blind evidence"
        ),
    }
    for condition, values in (("video", video_result), ("text_only", text_result)):
        path = output / f"{condition}-validation-predictions.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in values["validation_predictions"]
            ),
            encoding="utf-8",
        )
        report[f"{condition}_validation_predictions_sha256"] = sha256_file(path)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "video": video_result["validation_metrics_at_selected_checkpoint"],
                "text_only": text_result["validation_metrics_at_selected_checkpoint"],
                "bootstrap": bootstrap,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
