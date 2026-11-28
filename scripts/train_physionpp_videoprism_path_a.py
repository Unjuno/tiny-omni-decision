"""Fit one task-scoped Decision readout over frozen Physion++ VideoPrism features."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

if __package__:
    from .audit_physionpp_split_integrity import audit_train_validation_splits
else:
    from audit_physionpp_split_integrity import audit_train_validation_splits
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

OPTIONS = ("No", "Yes")
VIDEO_DIMENSION = 768
CANDIDATE_DIMENSION = 2
HIDDEN_SIZE = 128
EXPECTED_TRAIN_ROWS = 652
EXPECTED_VALIDATION_ROWS = 148
EXPECTED_TRAIN_HASH = "0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b"
EXPECTED_VALIDATION_HASH = "1254f01256197f1c2e23f8fa13dc03f87789c328f32bf768300134dd6b70d8dc"
TRAIN_FEATURES = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-videoprism-features-20261128-attempt-02"
)


def _read_jsonl(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = [row.get("id") for row in rows]
    if not rows or any(row.get("split") != expected_split for row in rows):
        raise ValueError(f"expected non-empty {expected_split} manifest")
    if len(set(ids)) != len(rows):
        raise ValueError(f"duplicate IDs in {expected_split} manifest")
    if any(row.get("options") != list(OPTIONS) for row in rows):
        raise ValueError("experiment requires the fixed ordered [No, Yes] options")
    if any(row.get("target") not in OPTIONS for row in rows):
        raise ValueError("all targets must be one of the supplied [No, Yes] options")
    return rows


def load_feature_cache(
    cache_path: Path,
    report_path: Path,
    rows: list[dict[str, Any]],
    *,
    expected_split: str,
    expected_manifest_sha256: str,
) -> np.ndarray:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_status = (
        "complete_train_feature_cache"
        if expected_split == "train"
        else "complete_development_validation_feature_cache"
    )
    if report.get("status") != expected_status:
        raise ValueError(f"incomplete {expected_split} feature cache")
    if (
        report.get("model_revision") != "be719a406d563b66f0ac969e7c94bab8e997c81a"
        or report.get("checkpoint_sha256")
        != "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
        or report.get("frame_count") != 8
        or report.get("frame_size") != 288
        or report.get("feature_dimension") != VIDEO_DIMENSION
        or report.get("device_policy") != "JAX_PLATFORMS=cpu"
    ):
        raise ValueError(f"{expected_split} cache does not match the pinned VideoPrism profile")
    manifest_key = (
        "train_manifest_sha256" if expected_split == "train" else "validation_manifest_sha256"
    )
    ids_key = (
        "feature_ids_in_manifest_order"
        if expected_split == "train"
        else "validation_ids_in_manifest_order"
    )
    media_key = (
        "feature_media_sha256_in_manifest_order"
        if expected_split == "train"
        else "validation_media_sha256_in_manifest_order"
    )
    if report.get(manifest_key) != expected_manifest_sha256:
        raise ValueError(f"{expected_split} cache manifest hash mismatch")
    if report.get("feature_cache_sha256") != sha256_file(cache_path):
        raise ValueError(f"{expected_split} cache file hash mismatch")
    expected_ids = [row["id"] for row in rows]
    expected_media = [row["media"][0]["sha256"] for row in rows]
    if report.get(ids_key) != expected_ids or report.get(media_key) != expected_media:
        raise ValueError(f"{expected_split} feature report ID/media order mismatch")
    with np.load(cache_path, allow_pickle=False) as payload:
        ids = payload["sample_ids"].astype(str).tolist()
        media_hashes = payload["media_sha256"].astype(str).tolist()
        features = np.asarray(payload["features"], dtype=np.float32)
    if ids != expected_ids or media_hashes != expected_media:
        raise ValueError(f"{expected_split} feature cache ID/media order mismatch")
    if features.shape != (len(rows), VIDEO_DIMENSION) or not np.isfinite(features).all():
        raise ValueError(f"invalid {expected_split} feature matrix: {features.shape}")
    return features


def candidate_features(video: np.ndarray, options: list[str]) -> np.ndarray:
    if video.shape != (VIDEO_DIMENSION,) or not np.isfinite(video).all():
        raise ValueError("video feature must be a finite 768-D vector")
    if len(options) != 2 or set(options) != set(OPTIONS):
        raise ValueError("options must be a permutation of the fixed [No, Yes] pair")
    matrix = np.zeros((len(options), VIDEO_DIMENSION + CANDIDATE_DIMENSION), dtype=np.float32)
    for index, option in enumerate(options):
        matrix[index, :VIDEO_DIMENSION] = video
        matrix[index, VIDEO_DIMENSION + OPTIONS.index(option)] = 1.0
    return matrix


def metric_values(logits: Tensor, targets: Tensor, bins: int = 15) -> dict[str, float]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = (predictions == targets).float()
    nll = F.nll_loss(probabilities.clamp_min(1e-12).log(), targets)
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[-1]).float()
    brier = (probabilities - one_hot).square().sum(dim=-1).mean()
    ece = torch.zeros((), dtype=torch.float32)
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs()
    return {
        "accuracy": float(correct.mean()),
        "nll": float(nll),
        "brier": float(brier),
        f"ece_{bins}": float(ece),
        "mean_confidence": float(confidence.mean()),
        "count": int(targets.numel()),
    }


def _batch_logits(
    scorer: FrozenFeatureCandidateScorer, features: Tensor, batch_size: int = 256
) -> Tensor:
    scorer.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            outputs.append(scorer(features[start : start + batch_size]))
    return torch.cat(outputs)


def _build_features(videos: np.ndarray, rows: list[dict[str, Any]]) -> Tensor:
    result = np.stack(
        [candidate_features(video, row["options"]) for video, row in zip(videos, rows, strict=True)]
    )
    return torch.from_numpy(result)


def _targets(rows: list[dict[str, Any]]) -> Tensor:
    return torch.tensor([row["options"].index(row["target"]) for row in rows], dtype=torch.long)


def validate_config(config: dict[str, Any]) -> None:
    expected_source = {
        "dataset_id": "physionpp/readout",
        "revision": "0c53b983ca138962ba636c0936e735490ad4c326",
        "train_rows": EXPECTED_TRAIN_ROWS,
        "validation_rows": EXPECTED_VALIDATION_ROWS,
        "train_manifest_sha256": EXPECTED_TRAIN_HASH,
        "validation_manifest_sha256": EXPECTED_VALIDATION_HASH,
        "split_group": "scenario_plus_scene_seed",
        "sealed_audit_loaded": False,
    }
    if any(config.get("source", {}).get(key) != value for key, value in expected_source.items()):
        raise ValueError("config source/data contract differs from the fixed corpus")
    if config.get("task", {}).get("options") != list(OPTIONS):
        raise ValueError("config task options differ from the frozen binary task")
    encoder = config.get("encoder", {})
    if (
        encoder.get("repo_id") != "google/videoprism-base-f16r288"
        or encoder.get("revision") != "be719a406d563b66f0ac969e7c94bab8e997c81a"
        or encoder.get("checkpoint_sha256")
        != "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
        or encoder.get("frozen") is not True
        or encoder.get("frames") != 8
        or encoder.get("resolution") != 288
        or encoder.get("feature_dimension") != VIDEO_DIMENSION
    ):
        raise ValueError("config encoder profile differs from the pinned frozen VideoPrism")
    readout = config.get("readout", {})
    expected_readout = {
        "architecture": "shared_FrozenFeatureCandidateScorer",
        "input_dimension": 770,
        "hidden_size": HIDDEN_SIZE,
        "trainable_parameters_expected": 98817,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "loss": "cross_entropy",
        "seed": 17,
        "epochs": 200,
        "early_stopping": False,
        "batch_questions": 32,
        "candidate_order_randomization": "per_question_per_epoch",
        "checkpoint_selection": "minimum_development_validation_nll",
        "preserve_best_and_final_checkpoint_separately": True,
    }
    if any(readout.get(key) != value for key, value in expected_readout.items()):
        raise ValueError("config readout/training policy differs from the frozen candidate")


def train_and_evaluate(
    *,
    train_path: Path,
    validation_path: Path,
    train_cache: Path,
    train_cache_report: Path,
    validation_cache: Path,
    validation_cache_report: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if sha256_file(train_path) != EXPECTED_TRAIN_HASH:
        raise ValueError("frozen train manifest hash mismatch")
    if sha256_file(validation_path) != EXPECTED_VALIDATION_HASH:
        raise ValueError("frozen development validation manifest hash mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "path-a-video-videoprism-physionpp-contact-v0-seed17":
        raise ValueError("unexpected experiment ID")
    validate_config(config)
    train_rows = _read_jsonl(train_path, "train")
    validation_rows = _read_jsonl(validation_path, "validation")
    if len(train_rows) != EXPECTED_TRAIN_ROWS or len(validation_rows) != EXPECTED_VALIDATION_ROWS:
        raise ValueError("frozen corpus row counts differ from the registered experiment")
    split_audit = audit_train_validation_splits(train_path, validation_path)
    if split_audit.get("status") != "disjoint":
        raise ValueError("train/validation split integrity check failed")

    train_videos = load_feature_cache(
        train_cache,
        train_cache_report,
        train_rows,
        expected_split="train",
        expected_manifest_sha256=EXPECTED_TRAIN_HASH,
    )
    validation_videos = load_feature_cache(
        validation_cache,
        validation_cache_report,
        validation_rows,
        expected_split="validation",
        expected_manifest_sha256=EXPECTED_VALIDATION_HASH,
    )
    train_features = _build_features(train_videos, train_rows)
    validation_features = _build_features(validation_videos, validation_rows)
    train_targets = _targets(train_rows)
    validation_targets = _targets(validation_rows)

    seed = int(config["readout"]["seed"])
    epochs = int(config["readout"]["epochs"])
    batch_questions = int(config["readout"]["batch_questions"])
    learning_rate = float(config["readout"]["learning_rate"])
    weight_decay = float(config["readout"]["weight_decay"])
    hidden_size = int(config["readout"]["hidden_size"])
    if seed != 17 or epochs != 200 or config["readout"]["early_stopping"] is not False:
        raise ValueError("this run requires the frozen seed-17/200-epoch/no-early-stop schedule")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    scorer = FrozenFeatureCandidateScorer(VIDEO_DIMENSION + CANDIDATE_DIMENSION, hidden_size)
    parameter_count = sum(parameter.numel() for parameter in scorer.parameters())
    if parameter_count != int(config["readout"]["trainable_parameters_expected"]):
        raise ValueError(f"readout parameter count mismatch: {parameter_count}")
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_nll = float("inf")
    started_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        scorer.train()
        order = torch.randperm(len(train_features), generator=generator)
        losses = []
        for start in range(0, len(order), batch_questions):
            indices = order[start : start + batch_questions]
            batch = train_features[indices]
            targets = train_targets[indices]
            swaps = torch.rand(len(indices), generator=generator) < 0.5
            first = torch.where(swaps, 1, 0)
            second = 1 - first
            permutations = torch.stack([first, second], dim=1)
            gather_index = permutations.unsqueeze(-1).expand(-1, -1, batch.shape[-1])
            batch = batch.gather(1, gather_index)
            targets = torch.where(swaps, 1 - targets, targets)
            logits = scorer(batch)
            loss = F.cross_entropy(logits.float(), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            losses.append(float(loss.detach()))

        train_logits = _batch_logits(scorer, train_features)
        validation_logits = _batch_logits(scorer, validation_features)
        train_metrics = metric_values(train_logits, train_targets)
        validation_metrics = metric_values(validation_logits, validation_targets)
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": float(np.mean(losses)),
                "train": train_metrics,
                "development_validation": validation_metrics,
            }
        )
        if validation_metrics["nll"] < best_nll:
            best_nll = validation_metrics["nll"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in scorer.state_dict().items()
            }

    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("validation did not select a checkpoint")
    final_state = {
        name: tensor.detach().cpu().clone() for name, tensor in scorer.state_dict().items()
    }
    scorer.load_state_dict(best_state)
    selected_logits = _batch_logits(scorer, validation_features)
    selected_metrics = metric_values(selected_logits, validation_targets)
    selected_train_metrics = metric_values(_batch_logits(scorer, train_features), train_targets)

    # Input-necessity controls are inference-only and use the selected checkpoint.
    zero_video = validation_videos.copy()
    zero_video[:] = 0.0
    zero_features = _build_features(zero_video, validation_rows)
    zero_logits = _batch_logits(scorer, zero_features)
    permutation = torch.randperm(
        len(validation_videos), generator=torch.Generator().manual_seed(17017)
    ).numpy()
    permuted_features = _build_features(validation_videos[permutation], validation_rows)
    permuted_logits = _batch_logits(scorer, permuted_features)

    canonical = validation_features[0]
    reversed_logits = scorer(canonical.flip(0)).flip(0)
    option_order_max_abs_logit_delta = float((reversed_logits - selected_logits[0]).abs().max())
    if option_order_max_abs_logit_delta > 1e-6:
        raise AssertionError("candidate order permutation changed semantic option scores")

    prior = train_targets.bincount(minlength=2).float() / len(train_targets)
    prior_logits = prior.clamp_min(1e-12).log().expand(len(validation_rows), -1)
    prior_metrics = metric_values(prior_logits, validation_targets)
    zero_metrics = metric_values(zero_logits, validation_targets)
    permuted_metrics = metric_values(permuted_logits, validation_targets)

    output.mkdir(parents=True, exist_ok=False)
    best_path = output / "best-readout.pt"
    final_path = output / "final-readout.pt"
    torch.save(
        {
            "state_dict": best_state,
            "feature_size": VIDEO_DIMENSION + CANDIDATE_DIMENSION,
            "hidden_size": hidden_size,
            "seed": seed,
            "best_epoch": best_epoch,
            "selection_metric": "minimum development validation NLL",
        },
        best_path,
    )
    torch.save(
        {
            "state_dict": final_state,
            "feature_size": VIDEO_DIMENSION + CANDIDATE_DIMENSION,
            "hidden_size": hidden_size,
            "seed": seed,
            "final_epoch": epochs,
        },
        final_path,
    )
    history_path = output / "learning-history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    predictions = []
    for index, row in enumerate(validation_rows):
        probabilities = torch.softmax(selected_logits[index].float(), dim=-1).tolist()
        predictions.append(
            {
                "id": row["id"],
                "source_asset_group_id": row["source_asset_group_id"],
                "options": row["options"],
                "target": row["target"],
                "prediction": OPTIONS[int(np.argmax(probabilities))],
                "probabilities": probabilities,
            }
        )
    prediction_path = output / "validation-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "complete_task_specific_path_a_video_readout",
        "claim_scope": (
            "one simulated Physion++ binary contact task; exposed development validation; "
            "not general video/world understanding"
        ),
        "run_started_utc": started_utc,
        "run_finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess_head(),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "train_manifest_sha256": sha256_file(train_path),
        "validation_manifest_sha256": sha256_file(validation_path),
        "train_feature_cache_sha256": sha256_file(train_cache),
        "train_feature_report_sha256": sha256_file(train_cache_report),
        "validation_feature_cache_sha256": sha256_file(validation_cache),
        "validation_feature_report_sha256": sha256_file(validation_cache_report),
        "split_integrity": split_audit,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "unique_train_groups": len({row["source_asset_group_id"] for row in train_rows}),
        "unique_validation_groups": len({row["source_asset_group_id"] for row in validation_rows}),
        "train_labels": {
            name: int((train_targets == index).sum()) for index, name in enumerate(OPTIONS)
        },
        "validation_labels": {
            name: int((validation_targets == index).sum()) for index, name in enumerate(OPTIONS)
        },
        "validation_is_exposed_development_data": True,
        "sealed_or_publisher_test_loaded": False,
        "human_blinded_observability_review_complete": False,
        "model": {
            "encoder_revision": "be719a406d563b66f0ac969e7c94bab8e997c81a",
            "encoder_checkpoint_sha256": (
                "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
            ),
            "encoder_parameters_frozen": 114365184,
            "candidate_scorer_trainable_parameters": parameter_count,
            "resident_dense_parameter_estimate": 114365184 + parameter_count,
            "candidate_order_max_abs_logit_delta": option_order_max_abs_logit_delta,
        },
        "effective_training": {
            "seed": seed,
            "epochs_completed": epochs,
            "early_stopping": False,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_questions": batch_questions,
            "candidate_order_randomization": "per question on every epoch",
            "checkpoint_selection": "minimum development validation NLL",
            "best_epoch": best_epoch,
            "final_checkpoint_epoch": epochs,
            "trainable_parameters": parameter_count,
        },
        "selected_checkpoint": {
            "development_validation": selected_metrics,
            "train_fit": selected_train_metrics,
        },
        "controls": {
            "training_label_prior": prior_metrics,
            "selected_checkpoint_zero_video": zero_metrics,
            "selected_checkpoint_permuted_validation_video": permuted_metrics,
            "permutation_seed": 17017,
        },
        "artifacts": {
            "best_checkpoint_sha256": sha256_file(best_path),
            "best_checkpoint_bytes": best_path.stat().st_size,
            "final_checkpoint_sha256": sha256_file(final_path),
            "learning_history_sha256": sha256_file(history_path),
            "validation_predictions_sha256": sha256_file(prediction_path),
        },
        "training_seconds": training_seconds,
        "device": "cpu",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pyyaml": importlib.metadata.version("PyYAML"),
        },
        "limitations": [
            "one seed and one task-specific binary simulator benchmark",
            "validation was used for per-epoch checkpoint selection and is not blind",
            "blinded human review of label observability is incomplete",
            "no audio, image, text variation, or joint multimodal evidence is tested",
            "no quantization, deployment, or product promotion was performed",
        ],
    }
    report_path = output / "run-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def subprocess_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-feature-cache", type=Path, required=True)
    parser.add_argument("--train-feature-report", type=Path, required=True)
    parser.add_argument("--validation-feature-cache", type=Path, required=True)
    parser.add_argument("--validation-feature-report", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/pretrained_reuse/path_a_video_videoprism_physionpp_contact_seed17.yaml"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train_and_evaluate(
        train_path=args.train.resolve(strict=True),
        validation_path=args.validation.resolve(strict=True),
        train_cache=args.train_feature_cache.resolve(strict=True),
        train_cache_report=args.train_feature_report.resolve(strict=True),
        validation_cache=args.validation_feature_cache.resolve(strict=True),
        validation_cache_report=args.validation_feature_report.resolve(strict=True),
        config_path=args.config.resolve(strict=True),
        output=args.output.resolve(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
