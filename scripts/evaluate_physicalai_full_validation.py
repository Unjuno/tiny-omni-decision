"""Score the frozen PhysicalAI pilot probes over all project validation episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.physicalai_pilot_metrics import expected_calibration_error
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from physicalai_pilot_metrics import expected_calibration_error


def metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    one_hot = np.eye(probabilities.shape[1])[targets]
    present = np.unique(targets)
    per_class_accuracy = {
        str(int(label)): float(correct[targets == label].mean()) for label in present
    }
    return {
        "accuracy": float(correct.mean()),
        "macro_accuracy_present_classes": float(np.mean(list(per_class_accuracy.values()))),
        "per_class_accuracy": per_class_accuracy,
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-8, 1)).mean()
        ),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_10": expected_calibration_error(probabilities, targets, bins=10),
        "mean_confidence": float(confidence.mean()),
        "n": int(len(targets)),
    }


def evaluate(
    pilot_dir: Path, validation_dir: Path, *, bootstrap_replicates: int = 2000
) -> dict[str, Any]:
    features_path = validation_dir / "features.npz"
    features = np.load(features_path, allow_pickle=False)
    if not np.all(features["split"] == "validation"):
        raise ValueError("full-validation feature archive contains non-validation examples")
    if len(np.unique(features["episode"])) < 200:
        raise ValueError("full validation must include at least 200 episode groups")

    # Torch is an optional model-scoring dependency; metric and fail-closed
    # validation tests should remain runnable in the CPU-only CI environment.
    import torch
    from torch import nn

    checkpoint_path = pilot_dir / "probe-checkpoints.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    action = features["action"].astype(np.float64)
    scaled_action = (action - checkpoint["action_mean_train"].numpy()) / checkpoint[
        "action_std_train"
    ].numpy()
    prototypes = checkpoint["action_prototypes_train"].numpy()
    targets = np.argmin(
        ((scaled_action[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    video = features["video"].astype(np.float32)
    state = features["state"].astype(np.float32)
    last_frame = features["video_last"].astype(np.float32)
    val_count = len(targets)
    joint_inputs = np.concatenate([video, state], axis=1)
    modality_inputs = {
        "proprio_history": state,
        "video_8frame_mean": video,
        "video_last_frame": last_frame,
        "joint_video_proprio": joint_inputs,
    }
    probabilities: dict[str, np.ndarray] = {}
    models: dict[str, tuple[nn.Linear, torch.Tensor, torch.Tensor]] = {}
    for name, values in modality_inputs.items():
        arm = checkpoint["arms"][name]
        model = nn.Linear(values.shape[1], 8)
        model.load_state_dict(arm["state_dict"])
        mean = arm["feature_mean_train"]
        std = arm["feature_std_train"]
        with torch.inference_mode():
            logits = model((torch.tensor(values) - mean) / std)
            probabilities[name] = logits.softmax(1).numpy()
        models[name] = (model, mean, std)

    joint_model, joint_mean, joint_std = models["joint_video_proprio"]
    width = video.shape[1]
    controls = {
        "joint_video_missing": joint_inputs.copy(),
        "joint_proprio_missing": joint_inputs.copy(),
    }
    controls["joint_video_missing"][:, :width] = joint_mean[:width].numpy()
    controls["joint_proprio_missing"][:, width:] = joint_mean[width:].numpy()
    for name, values in controls.items():
        with torch.inference_mode():
            probabilities[name] = (
                joint_model((torch.tensor(values) - joint_mean) / joint_std).softmax(1).numpy()
            )

    episodes = features["episode"].astype(np.int64)
    windows = features["window"].astype(np.int64)
    unique_episodes = np.unique(episodes)
    next_episode = {
        int(episode): int(unique_episodes[(index + 1) % len(unique_episodes)])
        for index, episode in enumerate(unique_episodes)
    }
    feature_lookup = {
        (int(episode), int(window)): video[index]
        for index, (episode, window) in enumerate(zip(episodes, windows, strict=True))
    }
    permuted_video = np.stack(
        [
            feature_lookup[(next_episode[int(episode)], int(window))]
            for episode, window in zip(episodes, windows, strict=True)
        ]
    )
    misaligned = np.concatenate([permuted_video, state], axis=1)
    with torch.inference_mode():
        probabilities["joint_video_episode_misaligned"] = (
            joint_model((torch.tensor(misaligned) - joint_mean) / joint_std).softmax(1).numpy()
        )

    prior_metrics = json.loads((pilot_dir / "pilot-metrics.json").read_text())
    class_prior = np.asarray(prior_metrics["train_class_counts"], dtype=np.float64)
    class_prior /= class_prior.sum()
    probabilities["train_class_prior"] = np.tile(class_prior, (val_count, 1))
    results = {name: metrics(prob, targets) for name, prob in probabilities.items()}
    train_cache = np.load(pilot_dir / "features.npz", allow_pickle=False)
    train_mask = train_cache["split"] == "train"
    train_actions = train_cache["action"][train_mask].astype(np.float64)
    train_scaled_actions = (train_actions - checkpoint["action_mean_train"].numpy()) / checkpoint[
        "action_std_train"
    ].numpy()
    train_targets = np.argmin(
        ((train_scaled_actions[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2),
        axis=1,
    )
    train_inputs = {
        "proprio_history": train_cache["state"][train_mask].astype(np.float32),
        "video_8frame_mean": train_cache["video"][train_mask].astype(np.float32),
        "video_last_frame": train_cache["video_last"][train_mask].astype(np.float32),
    }
    train_inputs["joint_video_proprio"] = np.concatenate(
        [train_inputs["video_8frame_mean"], train_inputs["proprio_history"]], axis=1
    )
    fixed_train_metrics = {}
    for name, values in train_inputs.items():
        model, mean, std = models[name]
        with torch.inference_mode():
            train_prob = model((torch.tensor(values) - mean) / std).softmax(1).numpy()
        fixed_train_metrics[name] = metrics(train_prob, train_targets)
    results["fixed_train_metrics"] = fixed_train_metrics
    results["action_codebook"] = {
        "train_prototype_count": int(len(prototypes)),
        "validation_normalized_rmse": float(
            np.sqrt(np.mean((scaled_action - prototypes[targets]) ** 2))
        ),
        "validation_target_support": np.bincount(targets, minlength=len(prototypes)).tolist(),
        "validation_episode_count": int(len(unique_episodes)),
    }

    rng = np.random.default_rng(17)
    group_rows = {int(episode): np.flatnonzero(episodes == episode) for episode in unique_episodes}
    bootstrap = {}
    for baseline_name in (
        "proprio_history",
        "video_8frame_mean",
        "joint_video_missing",
        "joint_proprio_missing",
        "joint_video_episode_misaligned",
    ):
        differences = []
        for _ in range(bootstrap_replicates):
            sampled_groups = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)
            indices = np.concatenate([group_rows[int(group)] for group in sampled_groups])
            joint_accuracy = (
                probabilities["joint_video_proprio"][indices].argmax(1) == targets[indices]
            ).mean()
            baseline_accuracy = (
                probabilities[baseline_name][indices].argmax(1) == targets[indices]
            ).mean()
            differences.append(float(joint_accuracy - baseline_accuracy))
        bootstrap[baseline_name] = {
            "joint_minus_baseline_accuracy_mean": float(np.mean(differences)),
            "episode_cluster_bootstrap_95pct": np.quantile(differences, [0.025, 0.975]).tolist(),
            "resamples": bootstrap_replicates,
            "seed": 17,
        }
    results["episode_cluster_bootstrap_joint_accuracy"] = bootstrap
    results["protocol"] = {
        "fit_checkpoint": str(checkpoint_path),
        "fit_seed": 17,
        "fit_train_episodes": 64,
        "scored_validation_episodes": int(len(unique_episodes)),
        "validation_queries": int(val_count),
        "windows_per_episode": 4,
        "fit_checkpoint_changed": False,
        "checkpoint_selection_on_this_full_validation": False,
        "reserved_holdout_payload_used": False,
        "validation_is_blind": False,
        "bootstrap_group": "episode_index",
    }
    predictions = []
    for index in range(val_count):
        row = {
            "episode_index": int(episodes[index]),
            "window_index": int(windows[index]),
            "frame_index": int(features["t"][index]),
            "target_prototype": int(targets[index]),
        }
        for name, prob in probabilities.items():
            row[name] = {
                "prediction": int(prob[index].argmax()),
                "probabilities": prob[index].tolist(),
            }
        predictions.append(row)
    predictions_path = validation_dir / "full-validation-predictions.json"
    predictions_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    report = {
        "source_manifest_sha256": json.loads(
            (validation_dir / "download-receipt.json").read_text()
        )["manifest_sha256"],
        "feature_file_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "probe_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "results": results,
    }
    (validation_dir / "full-validation-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-artifact-dir", type=Path, required=True)
    parser.add_argument("--validation-artifact-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result = evaluate(
        args.pilot_artifact_dir,
        args.validation_artifact_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
