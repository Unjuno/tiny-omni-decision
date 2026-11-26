"""Fit fixed CPU readouts and controls on frozen AV features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SEED = 17
PCA_DIMENSIONS = 32
L2_WEIGHT = 0.1
MAX_LBFGS_ITERATIONS = 200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(
    manifest: dict[str, Any], video_result: dict[str, Any], audio_result: dict[str, Any]
):
    import torch

    torch.set_num_threads(1)
    video_rows = {row["sample_id"]: row for row in video_result["records"]}
    audio_rows = {(row["sample_id"], int(row["event_id"])): row for row in audio_result["records"]}
    samples = {}
    for split in ("train", "validation"):
        for row in manifest[split]:
            video_record = video_rows[row["sample_id"]]
            video = np.load(video_record["feature_path"], allow_pickle=False).astype(np.float32)
            if video.shape != (768,) or not np.isfinite(video).all():
                raise ValueError(f"invalid video feature: {row['sample_id']}")
            audio_vectors = []
            for option in row["options"]:
                audio_record = audio_rows[(row["sample_id"], int(option["event_id"]))]
                audio = np.load(audio_record["feature_path"], allow_pickle=False).astype(np.float32)
                if audio.shape != (384,) or not np.isfinite(audio).all():
                    raise ValueError(
                        f"invalid audio feature: {row['sample_id']}:{option['event_id']}"
                    )
                audio_vectors.append(audio)
            samples[row["sample_id"]] = {
                "row": row,
                "video": video,
                "audio": np.stack(audio_vectors),
                "split": split,
            }
    return samples


class PCAProjector:
    def __init__(self, torch: Any, dimensions: int) -> None:
        self.torch = torch
        self.dimensions = dimensions

    def fit(self, values: np.ndarray) -> PCAProjector:
        torch = self.torch
        tensor = torch.as_tensor(values, dtype=torch.float64, device="cpu")
        self.mean = tensor.mean(dim=0)
        centered = tensor - self.mean
        _, _, right = torch.linalg.svd(centered, full_matrices=False)
        self.components = right[: min(self.dimensions, right.shape[0])].T.contiguous()
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        tensor = self.torch.as_tensor(values, dtype=self.torch.float64, device="cpu")
        return ((tensor - self.mean) @ self.components).numpy().astype(np.float32)


class BinaryLogistic:
    def __init__(self, torch: Any, *, l2_weight: float, max_iter: int) -> None:
        self.torch = torch
        self.l2_weight = l2_weight
        self.max_iter = max_iter

    def fit(self, features: np.ndarray, labels: np.ndarray) -> BinaryLogistic:
        torch = self.torch
        x = torch.as_tensor(features, dtype=torch.float64, device="cpu")
        y = torch.as_tensor(labels, dtype=torch.float64, device="cpu")
        self.weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [self.weight, self.bias],
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            logits = x @ self.weight + self.bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss = loss + 0.5 * self.l2_weight * self.weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            logits = x @ self.weight + self.bias
            final_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            final_loss = final_loss + 0.5 * self.l2_weight * self.weight.square().sum()
            self.training_loss = float(final_loss)
        self.weight = self.weight.detach()
        self.bias = self.bias.detach()
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        x = self.torch.as_tensor(features, dtype=self.torch.float64, device="cpu")
        return (x @ self.weight + self.bias).detach().numpy()


def categorical_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == targets
    one_hot = np.eye(2, dtype=np.float64)[targets]
    nll = -np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-15))
    brier = np.square(probabilities - one_hot).sum(axis=1) / 2.0
    ece = 0.0
    bins = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (confidence >= low) & (confidence < high if index < 9 else confidence <= high)
        if not mask.any():
            bins.append({"count": 0, "mean_confidence": None, "accuracy": None})
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += int(mask.sum()) / len(targets) * abs(mean_confidence - accuracy)
        bins.append(
            {"count": int(mask.sum()), "mean_confidence": mean_confidence, "accuracy": accuracy}
        )
    return {
        "count": len(targets),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_10_equal_width": float(ece),
        "mean_confidence": float(confidence.mean()),
        "choice_accuracy": float(
            np.mean(
                scores[np.arange(len(targets)), targets]
                > scores[np.arange(len(targets)), 1 - targets]
            )
            + 0.5
            * np.mean(
                scores[np.arange(len(targets)), targets]
                == scores[np.arange(len(targets)), 1 - targets]
            )
        ),
        "calibration_bins": bins,
        "probabilities": probabilities,
        "predictions": predicted,
    }


def paired_bootstrap_delta(
    correct_a: np.ndarray, correct_b: np.ndarray, *, seed: int, repetitions: int = 2000
) -> dict[str, Any]:
    if len(correct_a) != len(correct_b) or len(correct_a) == 0:
        raise ValueError("paired bootstrap needs equally sized non-empty per-video arrays")
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, len(correct_a), size=len(correct_a))
        deltas[index] = np.mean(correct_a[sample]) - np.mean(correct_b[sample])
    return {
        "group_unit": "video_id; one sample per video",
        "seed": seed,
        "repetitions": repetitions,
        "mean_delta": float(np.mean(correct_a) - np.mean(correct_b)),
        "percentile_95_interval": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    manifest_path = args.manifest.resolve(strict=True)
    video_result_path = args.video_features.resolve(strict=True)
    audio_result_path = args.audio_features.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_result = json.loads(video_result_path.read_text(encoding="utf-8"))
    audio_result = json.loads(audio_result_path.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite readout output: {output}")
    output.mkdir(parents=True)
    if video_result["status"] != "PASS_TEMPORAL_MATCH_VIDEOPRISM_FEATURES":
        raise ValueError("VideoPrism features are incomplete")
    if audio_result["status"] != "PASS_TEMPORAL_MATCH_EFFICIENTAT_FEATURES":
        raise ValueError("EfficientAT features are incomplete")
    if video_result["manifest_sha256"] != sha256_file(manifest_path) or audio_result[
        "manifest_sha256"
    ] != sha256_file(manifest_path):
        raise ValueError("feature results do not belong to the frozen task manifest")
    train_ids = {row["sample_id"] for row in manifest["train"]}
    val_ids = {row["sample_id"] for row in manifest["validation"]}
    train_videos = {row["video_id"] for row in manifest["train"]}
    val_videos = {row["video_id"] for row in manifest["validation"]}
    if train_ids & val_ids or train_videos & val_videos:
        raise ValueError("train/validation sample or video identity overlap")
    if len(train_ids) != len(manifest["train"]) or len(val_ids) != len(manifest["validation"]):
        raise ValueError("duplicate split sample IDs")

    samples = load_features(manifest, video_result, audio_result)
    train_rows = [samples[row["sample_id"]] for row in manifest["train"]]
    val_rows = [samples[row["sample_id"]] for row in manifest["validation"]]

    def stack(rows: list[dict[str, Any]]):
        video = np.repeat(np.stack([row["video"] for row in rows]), 2, axis=0)
        audio = np.concatenate([row["audio"] for row in rows], axis=0)
        target = np.asarray([int(row["row"]["target_index"]) for row in rows], dtype=np.int64)
        option_labels = np.asarray(
            [int(option == row["row"]["target_index"]) for row in rows for option in range(2)],
            dtype=np.int64,
        )
        return video, audio, option_labels, target

    train_video, train_audio, train_option_labels, train_targets = stack(train_rows)
    val_video, val_audio, val_option_labels, val_targets = stack(val_rows)
    video_projector = PCAProjector(torch, PCA_DIMENSIONS).fit(train_video)
    audio_projector = PCAProjector(torch, PCA_DIMENSIONS).fit(train_audio)
    train_video_pc = video_projector.transform(train_video)
    train_audio_pc = audio_projector.transform(train_audio)
    val_video_pc = video_projector.transform(val_video)
    val_audio_pc = audio_projector.transform(val_audio)
    train_joint = np.concatenate(
        [train_audio_pc, train_video_pc, train_audio_pc * train_video_pc], axis=1
    )
    val_joint = np.concatenate([val_audio_pc, val_video_pc, val_audio_pc * val_video_pc], axis=1)

    audio_model = BinaryLogistic(torch, l2_weight=L2_WEIGHT, max_iter=MAX_LBFGS_ITERATIONS).fit(
        train_audio_pc, train_option_labels
    )
    video_model = BinaryLogistic(torch, l2_weight=L2_WEIGHT, max_iter=MAX_LBFGS_ITERATIONS).fit(
        train_video_pc, train_option_labels
    )
    joint_model = BinaryLogistic(torch, l2_weight=L2_WEIGHT, max_iter=MAX_LBFGS_ITERATIONS).fit(
        train_joint, train_option_labels
    )
    checkpoint_path = output / "readout-checkpoints.pt"
    torch.save(
        {
            "seed": SEED,
            "pca_dimensions": PCA_DIMENSIONS,
            "l2_weight": L2_WEIGHT,
            "max_lbfgs_iterations": MAX_LBFGS_ITERATIONS,
            "video_projector": {
                "mean": video_projector.mean.cpu(),
                "components": video_projector.components.cpu(),
            },
            "audio_projector": {
                "mean": audio_projector.mean.cpu(),
                "components": audio_projector.components.cpu(),
            },
            "models": {
                "audio_only": {"weight": audio_model.weight.cpu(), "bias": audio_model.bias.cpu()},
                "video_only": {"weight": video_model.weight.cpu(), "bias": video_model.bias.cpu()},
                "joint": {"weight": joint_model.weight.cpu(), "bias": joint_model.bias.cpu()},
            },
        },
        checkpoint_path,
    )

    def pair_scores(model: BinaryLogistic, features: np.ndarray, count: int) -> np.ndarray:
        return model.score(features).reshape(count, 2)

    val_scores = {
        "audio_only": pair_scores(audio_model, val_audio_pc, len(val_rows)),
        "video_only": pair_scores(video_model, val_video_pc, len(val_rows)),
        "joint": pair_scores(joint_model, val_joint, len(val_rows)),
    }
    train_scores = {
        "audio_only": pair_scores(audio_model, train_audio_pc, len(train_rows)),
        "video_only": pair_scores(video_model, train_video_pc, len(train_rows)),
        "joint": pair_scores(joint_model, train_joint, len(train_rows)),
    }
    metrics = {
        "train": {
            name: categorical_metrics(scores, train_targets)
            for name, scores in train_scores.items()
        },
        "validation": {
            name: categorical_metrics(scores, val_targets) for name, scores in val_scores.items()
        },
    }

    rng = np.random.default_rng(SEED)
    label_ids = np.asarray([int(item["row"]["label_id"]) for item in val_rows])
    video_perm = np.arange(len(val_rows))
    audio_pair_perm = np.arange(len(val_rows))
    singleton_label_groups = 0
    for label_id in np.unique(label_ids):
        indices = np.flatnonzero(label_ids == label_id)
        if len(indices) < 2:
            singleton_label_groups += len(indices)
            continue
        offset = int(rng.integers(1, len(indices)))
        video_perm[indices] = np.roll(indices, offset)
        offset = int(rng.integers(1, len(indices)))
        audio_pair_perm[indices] = np.roll(indices, offset)
    shuffled_video_pc = val_video_pc.reshape(len(val_rows), 2, -1)[video_perm].reshape(
        -1, PCA_DIMENSIONS
    )
    shuffled_video_joint = np.concatenate(
        [val_audio_pc, shuffled_video_pc, val_audio_pc * shuffled_video_pc], axis=1
    )
    video_missing = np.zeros_like(val_video_pc)
    missing_video_joint = np.concatenate(
        [val_audio_pc, video_missing, val_audio_pc * video_missing], axis=1
    )
    shuffled_audio_pc = val_audio_pc.reshape(len(val_rows), 2, -1)[audio_pair_perm].reshape(
        -1, PCA_DIMENSIONS
    )
    shuffled_audio_joint = np.concatenate(
        [shuffled_audio_pc, val_video_pc, shuffled_audio_pc * val_video_pc], axis=1
    )
    controls = {
        "joint_video_shuffled_across_videos": pair_scores(
            joint_model, shuffled_video_joint, len(val_rows)
        ),
        "joint_audio_shuffled_across_candidates": pair_scores(
            joint_model, shuffled_audio_joint, len(val_rows)
        ),
        "joint_video_missing_mean_imputed": pair_scores(
            joint_model, missing_video_joint, len(val_rows)
        ),
    }
    control_metrics = {
        name: categorical_metrics(scores, val_targets) for name, scores in controls.items()
    }

    joint_correct = val_scores["joint"].argmax(axis=1) == val_targets
    audio_correct = val_scores["audio_only"].argmax(axis=1) == val_targets
    paired = paired_bootstrap_delta(joint_correct, audio_correct, seed=SEED)

    predictions = []
    for index, item in enumerate(val_rows):
        row = item["row"]
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "label_id": row["label_id"],
                "target_index": int(val_targets[index]),
                "options": row["options"],
                "joint_scores": val_scores["joint"][index].tolist(),
                "joint_probabilities": metrics["validation"]["joint"]["probabilities"][
                    index
                ].tolist(),
                "audio_only_scores": val_scores["audio_only"][index].tolist(),
                "video_only_scores": val_scores["video_only"][index].tolist(),
            }
        )

    result = {
        "status": "PASS_TEMPORAL_MATCH_READOUTS",
        "manifest_sha256": sha256_file(manifest_path),
        "video_features_sha256": sha256_file(video_result_path),
        "audio_features_sha256": sha256_file(audio_result_path),
        "seed": SEED,
        "train_videos": len(train_rows),
        "validation_videos": len(val_rows),
        "train_candidate_rows": len(train_option_labels),
        "validation_candidate_rows": len(val_option_labels),
        "pca_dimensions_per_modality": PCA_DIMENSIONS,
        "joint_feature_definition": "[audio_PCA32, video_PCA32, elementwise_product]",
        "l2_weight": L2_WEIGHT,
        "max_lbfgs_iterations": MAX_LBFGS_ITERATIONS,
        "readout_checkpoint_sha256": sha256_file(checkpoint_path),
        "training_device": "CPU",
        "train_metrics": {
            name: {
                key: value
                for key, value in metric.items()
                if key not in ("probabilities", "predictions")
            }
            for name, metric in metrics["train"].items()
        },
        "validation_metrics": {
            name: {
                key: value
                for key, value in metric.items()
                if key not in ("probabilities", "predictions")
            }
            for name, metric in metrics["validation"].items()
        },
        "controls_validation_metrics": {
            name: {
                key: value
                for key, value in metric.items()
                if key not in ("probabilities", "predictions")
            }
            for name, metric in control_metrics.items()
        },
        "paired_joint_minus_audio_accuracy": paired,
        "shuffled_control_policy": (
            "permute paired video/audio features across validation videos within each sound label; "
            "singleton-label groups remain unshuffled"
        ),
        "singleton_label_video_count_not_shufflable": singleton_label_groups,
        "validation_scope": (
            "development-only; task source was previously inspected; no sealed audit"
        ),
        "predictions": predictions,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "predictions"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
