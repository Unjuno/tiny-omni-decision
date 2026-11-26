"""Run nested room-held-out AV matching on frozen dev-train feature pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.run_dcase2025_nested_room_cv import room_folds, sha256_file
from tiny_omni_decision.dcase2025_av_match import AVMatchReadout

ARMS = ("joint", "audio_missing", "video_missing", "video_shuffled", "uniform")


def _standardize(train: np.ndarray, values: np.ndarray):
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    return (values - mean) / scale, mean, scale


def _metrics(probabilities: np.ndarray, targets: np.ndarray):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(targets), 4)
        or not len(targets)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(1), 1.0, atol=1e-7)
        or np.any(targets < 0)
        or np.any(targets > 3)
    ):
        raise ValueError("expected normalized four-choice probabilities and valid targets")
    predicted = probabilities.argmax(1)
    confidence = probabilities.max(1)
    ece = 0.0
    for bucket in range(15):
        lower = bucket / 15
        upper = (bucket + 1) / 15
        mask = (confidence >= lower) & (confidence <= upper if bucket == 14 else confidence < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float((predicted[mask] == targets[mask]).mean()) - float(confidence[mask].mean())
            )
    onehot = np.eye(4, dtype=np.float64)[targets]
    return {
        "count": len(targets),
        "accuracy": float((predicted == targets).mean()),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-15, 1)).mean()
        ),
        "brier": float(np.square(probabilities - onehot).sum(1).mean()),
        "ece_15": ece,
        "mean_confidence": float(confidence.mean()),
    }


def _shuffle_options(videos: np.ndarray, query_ids: list[str], *, seed: int):
    shuffled = videos.copy()
    for index, query_id in enumerate(query_ids):
        offset = 1 + hashlib.sha256(f"{seed}\0{query_id}".encode()).digest()[0] % 3
        shuffled[index] = np.roll(videos[index], int(offset), axis=0)
    return shuffled


def fit_fold(
    audio_train: np.ndarray,
    video_train: np.ndarray,
    targets_train: np.ndarray,
    audio_valid: np.ndarray,
    video_valid: np.ndarray,
    targets_valid: np.ndarray,
    *,
    seed: int,
    rank: int,
    max_epochs: int,
    patience: int,
):
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = AVMatchReadout(rank=rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    train_audio = torch.from_numpy(audio_train.astype(np.float32, copy=False))
    train_video = torch.from_numpy(video_train.astype(np.float32, copy=False))
    train_target = torch.as_tensor(targets_train, dtype=torch.long)
    valid_audio = torch.from_numpy(audio_valid.astype(np.float32, copy=False))
    valid_video = torch.from_numpy(video_valid.astype(np.float32, copy=False))
    valid_target = torch.as_tensor(targets_valid, dtype=torch.long)
    best_state = None
    best_nll = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train_target), generator=generator)
        batch_losses = []
        for indexes in order.split(32):
            logits = model(train_audio[indexes], train_video[indexes])
            loss = F.cross_entropy(logits, train_target[indexes])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite train CE at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_logits = model(valid_audio, valid_video)
            val_nll = float(F.cross_entropy(val_logits, valid_target))
            train_logits = model(train_audio, train_video)
            train_nll = float(F.cross_entropy(train_logits, train_target))
        if not np.isfinite(val_nll) or not np.isfinite(train_nll):
            raise FloatingPointError(f"non-finite evaluation CE at epoch {epoch + 1}")
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_batch_ce": float(np.mean(batch_losses)),
                "train_nll": train_nll,
                "validation_nll": val_nll,
            }
        )
        if val_nll < best_nll - 0.0001:
            best_nll = val_nll
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise AssertionError("no finite validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        train_probs = model(train_audio, train_video).softmax(-1).numpy()
        valid_probs = model(valid_audio, valid_video).softmax(-1).numpy()
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "train_metrics": _metrics(train_probs, targets_train),
        "validation_metrics": _metrics(valid_probs, targets_valid),
        "train_validation_nll_gap": float(
            _metrics(valid_probs, targets_valid)["nll"]
            - _metrics(train_probs, targets_train)["nll"]
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def run(
    features_dir: Path,
    task_dir: Path,
    output_dir: Path,
    *,
    seed: int = 17,
    rank: int = 8,
    max_epochs: int = 30,
    patience: int = 5,
):
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run output: {output_dir}")
    task_preflight = json.loads((task_dir / "task-preflight.json").read_text(encoding="utf-8"))
    task_config_path = task_dir / "effective-config.json"
    task_config = json.loads(task_config_path.read_text(encoding="utf-8"))
    task_manifest = task_dir / "task-manifest.jsonl"
    if (
        task_preflight.get("status") != "PASS"
        or task_preflight.get("positive_matches_query_parent") is not True
        or task_preflight.get("candidate_parent_unique_per_query") is not True
        or task_config.get("devtest_accessed") is not False
        or task_config.get("sealed_audit_accessed") is not False
        or sha256_file(task_manifest) != task_preflight.get("task_manifest_sha256")
    ):
        raise ValueError("frozen AV match task preflight failed")
    if (
        sha256_file(features_dir / "preflight.json") != task_config.get("source_preflight_sha256")
        or sha256_file(features_dir / "query-manifest.jsonl")
        != task_config.get("source_manifest_sha256")
        or sha256_file(features_dir / "features_audio.npy")
        != task_config.get("audio_features_sha256")
        or sha256_file(features_dir / "features_video.npy")
        != task_config.get("video_features_sha256")
    ):
        raise ValueError("source feature corpus hashes differ from the frozen task config")
    feature_progress = json.loads((features_dir / "progress.json").read_text(encoding="utf-8"))
    if feature_progress.get("identity") != task_config.get("feature_identity"):
        raise ValueError("task manifest was built from different feature identity")
    examples = [json.loads(line) for line in task_manifest.read_text(encoding="utf-8").splitlines()]
    if len(examples) != task_preflight.get("examples") or len(
        {row["query_id"] for row in examples}
    ) != len(examples):
        raise ValueError("task manifest row count or query IDs are invalid")
    source_rows = [
        json.loads(line)
        for line in (features_dir / "query-manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for example in examples:
        if len(example.get("options", [])) != 4 or not 0 <= int(example["target_index"]) < 4:
            raise ValueError("each task query must have exactly four choices and a valid target")
        query_row = source_rows[int(example["query_row"])]
        if (
            query_row.get("sample_id") != example.get("query_sample_id")
            or query_row.get("parent_recording") != example.get("query_parent")
            or query_row.get("room_id") != example.get("room_id")
            or query_row.get("event_class") != example.get("event_class")
        ):
            raise ValueError("task query identity does not match its source feature row")
        options = example["options"]
        target_index = int(example["target_index"])
        if (
            len({option["parent_recording"] for option in options}) != 4
            or options[target_index]["parent_recording"] != example["query_parent"]
            or options[target_index]["sample_id"] != example["query_sample_id"]
        ):
            raise ValueError("task positive/negative parent identities are invalid")
        for option in options:
            option_row = source_rows[int(option["feature_row"])]
            if (
                option_row.get("sample_id") != option.get("sample_id")
                or option_row.get("parent_recording") != option.get("parent_recording")
                or option_row.get("room_id") != example.get("room_id")
                or option_row.get("event_class") != example.get("event_class")
            ):
                raise ValueError("candidate option does not match its same-room/event source row")
    rows_for_split = [
        {
            "sample_id": row["query_id"],
            "room_id": row["room_id"],
            "parent_recording": row["query_parent"],
        }
        for row in examples
    ]
    folds = room_folds(rows_for_split, seed=seed)
    raw_audio = np.load(features_dir / "features_audio.npy", mmap_mode="r")
    raw_video = np.load(features_dir / "features_video.npy", mmap_mode="r")
    targets = np.asarray([row["target_index"] for row in examples], dtype=np.int64)
    if raw_audio.shape != raw_video.shape or raw_audio.shape[1] != 768:
        raise ValueError("source feature bank shape mismatch")
    query_rows = np.asarray([row["query_row"] for row in examples], dtype=np.int64)
    candidate_rows = np.asarray(
        [[option["feature_row"] for option in row["options"]] for row in examples], dtype=np.int64
    )
    query_ids = [str(row["query_id"]) for row in examples]
    probabilities = {name: np.full((len(examples), 4), np.nan, dtype=np.float64) for name in ARMS}
    output_dir.mkdir(parents=True)
    config = {
        "task_manifest_sha256": sha256_file(task_manifest),
        "task_config_sha256": sha256_file(task_config_path),
        "feature_identity": feature_progress["identity"],
        "audio_features_sha256": sha256_file(features_dir / "features_audio.npy"),
        "video_features_sha256": sha256_file(features_dir / "features_video.npy"),
        "task": (
            "4-way audio query to matching same-room/same-event video; parent identity is target"
        ),
        "selection": (
            "nested held-out room; minimum inner-validation NLL; min_delta=0.0001; "
            "patience=5; max_epochs=30"
        ),
        "architecture": (
            "rank-8 shared bilinear audio/video projections with normalized latent dot product"
        ),
        "rank": rank,
        "seed": seed,
        "optimizer": "AdamW lr=0.003 weight_decay=0.0001, batch_size=32, CPU",
        "candidate_count": 4,
        "target_position_counts": task_config["target_position_counts"],
        "outer_protocol": "leave-one-room-out across 9 rooms; every query tested once",
        "sealed_audit_accessed": False,
        "development_test_accessed": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    (output_dir / "effective-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fold_history = []
    started = time.perf_counter()
    for fold in folds:
        train_idx = fold["train_indices"]
        valid_idx = fold["validation_indices"]
        test_idx = fold["test_indices"]
        train_query_feature_rows = np.unique(query_rows[train_idx])
        train_video_feature_rows = np.unique(candidate_rows[train_idx].reshape(-1))
        audio_train_raw = np.asarray(raw_audio[train_query_feature_rows], dtype=np.float32)
        video_train_raw = np.asarray(raw_video[train_video_feature_rows], dtype=np.float32)
        _, audio_mean, audio_scale = _standardize(audio_train_raw, audio_train_raw)
        _, video_mean, video_scale = _standardize(video_train_raw, video_train_raw)
        audio_queries = (
            np.asarray(raw_audio[query_rows], dtype=np.float32) - audio_mean
        ) / audio_scale
        video_options = (
            np.asarray(raw_video[candidate_rows], dtype=np.float32) - video_mean
        ) / video_scale
        model, fit = fit_fold(
            audio_queries[train_idx],
            video_options[train_idx],
            targets[train_idx],
            audio_queries[valid_idx],
            video_options[valid_idx],
            targets[valid_idx],
            seed=seed,
            rank=rank,
            max_epochs=max_epochs,
            patience=patience,
        )
        record = {
            "test_room": fold["test_room"],
            "validation_room": fold["validation_room"],
            "train_queries": len(train_idx),
            "validation_queries": len(valid_idx),
            "test_queries": len(test_idx),
            **fit,
        }
        fold_history.append(record)
        test_audio = torch.from_numpy(audio_queries[test_idx])
        test_video = torch.from_numpy(video_options[test_idx])
        with torch.inference_mode():
            joint = model(test_audio, test_video).softmax(-1).numpy()
            audio_missing = model(torch.zeros_like(test_audio), test_video).softmax(-1).numpy()
            video_missing = model(test_audio, torch.zeros_like(test_video)).softmax(-1).numpy()
            shuffled_video = _shuffle_options(
                video_options[test_idx], [query_ids[index] for index in test_idx], seed=seed
            )
            video_shuffled = model(test_audio, torch.from_numpy(shuffled_video)).softmax(-1).numpy()
        probabilities["joint"][test_idx] = joint
        probabilities["audio_missing"][test_idx] = audio_missing
        probabilities["video_missing"][test_idx] = video_missing
        probabilities["video_shuffled"][test_idx] = video_shuffled
        probabilities["uniform"][test_idx] = 0.25
        record["outer_metrics"] = {
            name: _metrics(probabilities[name][test_idx], targets[test_idx]) for name in ARMS
        }
        (output_dir / "fold-history.json").write_text(
            json.dumps(fold_history, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if any(not np.isfinite(value).all() for value in probabilities.values()):
        raise AssertionError("incomplete or non-finite out-of-fold predictions")
    prediction_path = output_dir / "oof-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(examples):
            stream.write(
                json.dumps(
                    {
                        "query_id": row["query_id"],
                        "room_id": row["room_id"],
                        "event_class": row["event_class"],
                        "query_parent": row["query_parent"],
                        "target_index": int(targets[index]),
                        "probabilities": {
                            name: value[index].tolist() for name, value in probabilities.items()
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    room_metrics = {}
    event_metrics = {}
    for group_key, field in (("room", "room_id"), ("event_class", "event_class")):
        groups = sorted({row[field] for row in examples})
        target_metrics = room_metrics if group_key == "room" else event_metrics
        for group in groups:
            mask = np.asarray([row[field] == group for row in examples], dtype=bool)
            target_metrics[str(group)] = {
                name: _metrics(value[mask], targets[mask]) for name, value in probabilities.items()
            }
    result = {
        "status": "PASS",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.perf_counter() - started,
        "queries": len(examples),
        "rooms": len(room_metrics),
        "unique_query_parents": len({row["query_parent"] for row in examples}),
        "overall_metrics": {
            name: _metrics(value, targets) for name, value in probabilities.items()
        },
        "per_room_metrics": room_metrics,
        "per_event_metrics": event_metrics,
        "fold_results": fold_history,
        "trainable_parameters": sum(
            parameter.numel() for parameter in AVMatchReadout(rank=rank).parameters()
        ),
        "predictions_sha256": sha256_file(prediction_path),
        "development_test_accessed": False,
        "sealed_audit_accessed": False,
        "selection_used_outer_test": False,
        "interpretation_limit": (
            "One task generated from 78 source recordings and 9 rooms; development evidence only."
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    result = run(
        args.features,
        args.task,
        args.output,
        seed=args.seed,
        rank=args.rank,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "queries": result["queries"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
