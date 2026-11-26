"""Fit one fixed, small AV-match readout and evaluate frozen controls once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tiny_omni_decision.robomnist_av_match_evaluation import (  # noqa: E402
    classification_metrics,
    paired_bootstrap_delta,
    validate_examples,
)
from tiny_omni_decision.robomnist_av_match_readout import (  # noqa: E402
    BilinearAVMatchScorer,
    SingleModalityMatchScorer,
)

MANIFEST_SHA256 = "3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f"
SEED = 17
RANK = 32
TRAIN_STEPS = 400
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
BOOTSTRAP_REPLICATES = 2000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _load_group_features(
    cache: Path, group_ids: set[str]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    features: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for group_id in sorted(group_ids):
        group_dir = cache / "groups" / group_id
        receipt_path = group_dir / "receipt.json"
        feature_path = group_dir / "features.npz"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("group_id") != group_id
            or sha256_file(feature_path) != receipt.get("feature_sha256")
        ):
            raise ValueError(f"feature receipt/hash invalid for {group_id}")
        with np.load(feature_path, allow_pickle=False) as data:
            if str(data["group_id"].item()) != group_id:
                raise ValueError(f"feature payload group mismatch for {group_id}")
            video = np.asarray(data["video_features"], dtype=np.float32)
            audio = np.asarray(data["audio_features"], dtype=np.float32)
        if (
            video.shape != (8, 768)
            or audio.shape != (8, 384)
            or not np.isfinite(video).all()
            or not np.isfinite(audio).all()
        ):
            raise ValueError(f"feature payload shape/value mismatch for {group_id}")
        features[group_id] = (audio, video)
        hashes[group_id] = receipt["feature_sha256"]
    return features, hashes


def _pool_features(features: dict[str, np.ndarray], rows: list[dict[str, Any]]):
    unique_train_audio = sorted({row["audio_group"] for row in rows})
    unique_train_video = sorted({row["video_group"] for row in rows})
    audio_values = np.stack(
        [
            np.concatenate((features[group][0].mean(0), features[group][0].std(0)))
            for group in unique_train_audio
        ]
    )
    video_values = np.stack(
        [
            np.concatenate((features[group][1].mean(0), features[group][1].std(0)))
            for group in unique_train_video
        ]
    )
    audio_mean, audio_scale = audio_values.mean(0), audio_values.std(0)
    video_mean, video_scale = video_values.mean(0), video_values.std(0)
    audio_scale[audio_scale < 1e-6] = 1.0
    video_scale[video_scale < 1e-6] = 1.0
    summaries = {}
    for group, (audio, video) in features.items():
        audio_summary = np.concatenate((audio.mean(0), audio.std(0)))
        video_summary = np.concatenate((video.mean(0), video.std(0)))
        summaries[group] = (
            ((audio_summary - audio_mean) / audio_scale).astype(np.float32),
            ((video_summary - video_mean) / video_scale).astype(np.float32),
        )
    normalizers = {
        "audio_mean_sha256": hashlib.sha256(audio_mean.tobytes()).hexdigest(),
        "audio_scale_sha256": hashlib.sha256(audio_scale.tobytes()).hexdigest(),
        "video_mean_sha256": hashlib.sha256(video_mean.tobytes()).hexdigest(),
        "video_scale_sha256": hashlib.sha256(video_scale.tobytes()).hexdigest(),
        "fit_group_counts": {"audio": len(unique_train_audio), "video": len(unique_train_video)},
    }
    return summaries, normalizers


def _arrays(rows: list[dict[str, Any]], summaries: dict[str, tuple[np.ndarray, np.ndarray]]):
    audio = np.stack([summaries[row["audio_group"]][0] for row in rows])
    video = np.stack([summaries[row["video_group"]][1] for row in rows])
    target = np.asarray([row["target"] for row in rows], dtype=np.int64)
    pair_ids = np.asarray([row["pair_unit_id"] for row in rows])
    return (
        torch.from_numpy(audio),
        torch.from_numpy(video),
        torch.from_numpy(target),
        pair_ids,
    )


def _fit(model, train_inputs, targets: torch.Tensor, *, steps: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, BilinearAVMatchScorer):
            logits = model(*train_inputs)
        else:
            logits = model(train_inputs[0])
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            history.append({"step": step, "train_ce": float(loss.detach())})
    model.eval()
    return history


def _predict(model, audio: torch.Tensor, video: torch.Tensor, *, mode: str) -> np.ndarray:
    with torch.inference_mode():
        if isinstance(model, BilinearAVMatchScorer):
            logits = model(audio, video)
        elif mode == "audio":
            logits = model(audio)
        elif mode == "video":
            logits = model(video)
        else:
            raise ValueError(f"unknown prediction mode {mode}")
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()


def _counterfactual_rows(
    rows: list[dict[str, Any]], summaries: dict[str, tuple[np.ndarray, np.ndarray]], *, swap: str
):
    audio_values = []
    video_values = []
    for row in rows:
        pair_rows = [item for item in rows if item["pair_unit_id"] == row["pair_unit_id"]]
        groups = sorted(
            {item["audio_group"] if swap == "audio" else item["video_group"] for item in pair_rows}
        )
        current = row["audio_group"] if swap == "audio" else row["video_group"]
        replacement = next((group for group in groups if group != current), None)
        if replacement is None:
            raise ValueError("counterfactual pair unit does not contain two source groups")
        audio_group = replacement if swap == "audio" else row["audio_group"]
        video_group = replacement if swap == "video" else row["video_group"]
        audio_values.append(summaries[audio_group][0])
        video_values.append(summaries[video_group][1])
    return torch.from_numpy(np.stack(audio_values)), torch.from_numpy(np.stack(video_values))


def _load_frozen_fit(
    frozen_fit: Path,
    *,
    manifest_sha: str,
    train_feature_hashes_sha256: str,
    normalizer_hashes: dict[str, Any],
    steps: int,
):
    fit_meta = json.loads((frozen_fit / "run.json").read_text(encoding="utf-8"))
    checkpoint_path = frozen_fit / "readouts.pt"
    config_path = frozen_fit / "config.json"
    implementation_paths = {
        "train_only_script": REPO_ROOT / "scripts" / "fit_robomnist_av_match_train_only.py",
        "readout_fit_script": Path(__file__).resolve(),
        "readout_module": (
            REPO_ROOT / "src" / "tiny_omni_decision" / "robomnist_av_match_readout.py"
        ),
        "feature_extractor": REPO_ROOT / "scripts" / "extract_robomnist_av_match_feature_cache.py",
    }
    expected_implementation = {
        name: sha256_file(path) for name, path in implementation_paths.items()
    }
    if (
        fit_meta.get("status") != "TRAIN_COMPLETE_VALIDATION_PENDING"
        or fit_meta.get("manifest_sha256") != manifest_sha
        or fit_meta.get("train_feature_hashes_sha256") != train_feature_hashes_sha256
        or fit_meta.get("normalizer_hashes") != normalizer_hashes
        or fit_meta.get("implementation_sha256") != expected_implementation
        or fit_meta.get("training", {}).get("steps") != steps
        or fit_meta.get("training", {}).get("seed") != SEED
        or fit_meta.get("training", {}).get("validation_used") is not False
        or fit_meta.get("validation_features_read") is not False
        or not config_path.is_file()
        or sha256_file(config_path) != fit_meta.get("config_sha256")
        or not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != fit_meta.get("readout_checkpoint_sha256")
    ):
        raise ValueError("frozen train-only fit does not match this evaluation contract")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {"joint", "audio_only", "video_only"}
    if set(state) != expected:
        raise ValueError("frozen checkpoint does not contain the three declared readouts")
    return fit_meta, state


def run(
    manifest_path: Path,
    cache: Path,
    output: Path,
    *,
    steps: int = TRAIN_STEPS,
    frozen_fit: Path | None = None,
):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing readout output: {output}")
    if steps < 1:
        raise ValueError("training steps must be positive")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != MANIFEST_SHA256:
        raise ValueError("frozen RoboMNIST manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_meta = json.loads((cache / "run.json").read_text(encoding="utf-8"))
    if (
        run_meta.get("status") != "COMPLETE"
        or run_meta.get("effective", {}).get("manifest_sha256") != manifest_sha
        or run_meta.get("effective", {}).get("group_count") != 1780
        or run_meta.get("training") is not False
        or run_meta.get("sealed_audit_opened") is not False
    ):
        raise ValueError("feature cache is incomplete or does not match the frozen split")
    rows = validate_examples(manifest)
    group_ids = {
        row[key]
        for split_rows in rows.values()
        for row in split_rows
        for key in ("audio_group", "video_group")
    }
    features, feature_hashes = _load_group_features(cache, group_ids)
    summaries, normalizer_hashes = _pool_features(features, rows["train"])
    train_audio, train_video, train_target, _ = _arrays(rows["train"], summaries)
    val_audio, val_video, val_target, val_pair_ids = _arrays(rows["validation"], summaries)
    train_group_ids = {
        row[key] for row in rows["train"] for key in ("audio_group", "video_group")
    }
    train_feature_hashes_sha256 = hashlib.sha256(
        "\n".join(
            f"{key}:{feature_hashes[key]}" for key in sorted(train_group_ids)
        ).encode()
    ).hexdigest()
    frozen_fit_meta = None
    frozen_state = None
    if frozen_fit is not None:
        frozen_fit_meta, frozen_state = _load_frozen_fit(
            frozen_fit,
            manifest_sha=manifest_sha,
            train_feature_hashes_sha256=train_feature_hashes_sha256,
            normalizer_hashes=normalizer_hashes,
            steps=steps,
        )

    output.mkdir(parents=True)
    started = time.perf_counter()
    meta = {
        "status": "RUNNING",
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "implementation_sha256": {
            "fit_script": sha256_file(Path(__file__).resolve()),
            "readout_module": sha256_file(
                REPO_ROOT / "src" / "tiny_omni_decision" / "robomnist_av_match_readout.py"
            ),
        },
        "manifest_sha256": manifest_sha,
        "feature_run": str(cache),
        "feature_group_count": len(features),
        "feature_hashes_sha256": hashlib.sha256(
            "\n".join(f"{key}:{feature_hashes[key]}" for key in sorted(feature_hashes)).encode()
        ).hexdigest(),
        "train_feature_hashes_sha256": train_feature_hashes_sha256,
        "normalizer_hashes": normalizer_hashes,
        "seed": SEED,
        "training": {
            "steps": steps,
            "batch": "full train set each update; deterministic order",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": 5.0,
            "validation_used_for_training_or_selection": False,
        },
        "validation_count": len(rows["validation"]),
        "validation_pair_units": len(set(val_pair_ids.tolist())),
        "sealed_audit_opened": False,
        "gpu_used": False,
    }
    if frozen_fit is not None:
        meta["frozen_fit_run"] = str(frozen_fit)
        meta["frozen_fit_checkpoint_sha256"] = frozen_fit_meta["readout_checkpoint_sha256"]
    _write_json(output / "run.json", meta)

    try:
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        config = {
            "seed": SEED,
            "rank": RANK,
            "steps": steps,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "batch": "full train set each optimizer update",
            "validation_selection": False,
            "temperature": 1.0,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
        config_path = output / "config.json"
        _write_json(config_path, config)
        meta["config_sha256"] = sha256_file(config_path)
        _write_json(output / "run.json", meta)
        models = {
            "joint": BilinearAVMatchScorer(rank=RANK),
            "audio_only": SingleModalityMatchScorer(train_audio.shape[1]),
            "video_only": SingleModalityMatchScorer(train_video.shape[1]),
        }
        histories = {}
        training_seconds = {}
        if frozen_state is None:
            for name, model in models.items():
                torch.manual_seed(SEED)
                inputs = (
                    (train_audio, train_video)
                    if name == "joint"
                    else (train_audio if name == "audio_only" else train_video,)
                )
                fit_started = time.perf_counter()
                histories[name] = _fit(model, inputs, train_target, steps=steps)
                training_seconds[name] = time.perf_counter() - fit_started
        else:
            for name, model in models.items():
                model.load_state_dict(frozen_state[name], strict=True)
            histories = frozen_fit_meta["training_history"]
            training_seconds = frozen_fit_meta["training_seconds_by_readout"]
        inference_started = time.perf_counter()
        joint_prob = _predict(models["joint"], val_audio, val_video, mode="joint")
        audio_prob = _predict(models["audio_only"], val_audio, val_video, mode="audio")
        video_prob = _predict(models["video_only"], val_audio, val_video, mode="video")
        joint_correct = joint_prob.argmax(-1) == val_target.numpy()
        audio_correct = audio_prob.argmax(-1) == val_target.numpy()
        video_correct = video_prob.argmax(-1) == val_target.numpy()
        shuffled_audio, shuffled_audio_video = _counterfactual_rows(
            rows["validation"], summaries, swap="audio"
        )
        shuffled_video_audio, shuffled_video = _counterfactual_rows(
            rows["validation"], summaries, swap="video"
        )
        missing_audio = _predict(
            models["joint"], torch.zeros_like(val_audio), val_video, mode="joint"
        )
        missing_video = _predict(
            models["joint"], val_audio, torch.zeros_like(val_video), mode="joint"
        )
        shuffled_audio_prob = _predict(
            models["joint"], shuffled_audio, shuffled_audio_video, mode="joint"
        )
        shuffled_video_prob = _predict(
            models["joint"], shuffled_video_audio, shuffled_video, mode="joint"
        )
        inference_seconds = time.perf_counter() - inference_started
        predictions = {
            "joint": joint_prob,
            "audio_only": audio_prob,
            "video_only": video_prob,
            "missing_audio": missing_audio,
            "missing_video": missing_video,
            "shuffle_audio_within_pair": shuffled_audio_prob,
            "shuffle_video_within_pair": shuffled_video_prob,
        }
        targets = val_target.numpy()
        metrics = {
            name: classification_metrics(probability, targets)
            for name, probability in predictions.items()
        }
        train_predictions = {
            "joint": _predict(models["joint"], train_audio, train_video, mode="joint"),
            "audio_only": _predict(models["audio_only"], train_audio, train_video, mode="audio"),
            "video_only": _predict(models["video_only"], train_audio, train_video, mode="video"),
        }
        train_metrics = {
            name: classification_metrics(probability, train_target.numpy())
            for name, probability in train_predictions.items()
        }
        metrics["paired_bootstrap_joint_vs_audio_only"] = paired_bootstrap_delta(
            joint_correct, audio_correct, val_pair_ids
        )
        metrics["paired_bootstrap_joint_vs_video_only"] = paired_bootstrap_delta(
            joint_correct, video_correct, val_pair_ids
        )
        model_hashes = {}
        checkpoint_path = output / "readouts.pt"
        torch.save({name: model.state_dict() for name, model in models.items()}, checkpoint_path)
        model_hashes[checkpoint_path.name] = sha256_file(checkpoint_path)
        reloaded = {
            "joint": BilinearAVMatchScorer(rank=RANK),
            "audio_only": SingleModalityMatchScorer(train_audio.shape[1]),
            "video_only": SingleModalityMatchScorer(train_video.shape[1]),
        }
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for name, model in reloaded.items():
            model.load_state_dict(state[name], strict=True)
        reload_prob = _predict(reloaded["joint"], val_audio, val_video, mode="joint")
        if not np.array_equal(reload_prob, joint_prob):
            raise RuntimeError("save/reload changed joint validation probabilities")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-only readout unexpectedly initialized CUDA")
        prediction_rows = []
        for index, row in enumerate(rows["validation"]):
            prediction_rows.append(
                {
                    "example_index": index,
                    "pair_unit_id": row["pair_unit_id"],
                    "audio_group": row["audio_group"],
                    "video_group": row["video_group"],
                    "target_option_id": "same" if row["target"] else "different",
                    "predictions": {
                        name: {
                            "option_ids": ["different", "same"],
                            "predicted_option_id": ["different", "same"][
                                int(probability[index].argmax())
                            ],
                            "probabilities": probability[index].tolist(),
                        }
                        for name, probability in predictions.items()
                    },
                }
            )
        prediction_path = output / "validation-predictions.jsonl"
        with prediction_path.open("x", encoding="utf-8", newline="\n") as target:
            for row in prediction_rows:
                target.write(json.dumps(row, sort_keys=True) + "\n")
            target.flush()
            os.fsync(target.fileno())
        result = {
            **meta,
            "status": "COMPLETE",
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "training_history": histories,
            "training_seconds_by_readout": training_seconds,
            "readout_fit_reused_without_refit": frozen_fit is not None,
            "inference_seconds_all_validation_controls": inference_seconds,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": "CPU",
                "cuda_initialized": torch.cuda.is_initialized(),
            },
            "training_metrics_seen_examples": train_metrics,
            "parameters": {name: model.trainable_parameters for name, model in models.items()},
            "resident_encoder_parameters": 116_326_423,
            "readout_checkpoint_sha256": model_hashes[checkpoint_path.name],
            "readout_checkpoint_bytes": checkpoint_path.stat().st_size,
            "reload_exact_probability_match": True,
            "metrics": metrics,
            "prediction_file_sha256": sha256_file(prediction_path),
            "quality_claim_scope": "RoboMNIST episode-level AV repetition match only",
            "limitations": [
                "one fixed seed and one train-only fit; no hyperparameter selection",
                "validation is development data used only for final reported comparisons",
                "groups are episode-disjoint, not room/device/scene held out",
                "not broad multimodal or general-purpose Omni quality evidence",
            ],
        }
        _write_json(output / "training-history.json", histories)
        _write_json(output / "result.json", result)
        _write_json(output / "run.json", result)
        return result
    except BaseException as exc:
        meta.update(
            {
                "status": "INCOMPLETE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _write_json(output / "run.json", meta)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument(
        "--frozen-fit",
        type=Path,
        help="evaluate a matching train-only checkpoint without fitting it again",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.manifest.resolve(strict=True),
                args.cache.resolve(strict=True),
                args.output.resolve(),
                steps=args.steps,
                frozen_fit=args.frozen_fit.resolve(strict=True) if args.frozen_fit else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
