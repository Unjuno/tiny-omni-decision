"""Fit preregistered small CPU readouts over frozen RoboPro features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1]
        )
        if selected.any():
            result += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(result)


def _metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    y = np.asarray(labels, dtype=np.int64)
    one_hot = np.eye(p.shape[1], dtype=np.float64)[y]
    return {
        "count": int(len(y)),
        "accuracy": float(np.mean(p.argmax(axis=1) == y)),
        "nll": float(-np.log(p[np.arange(len(y)), y]).mean()),
        "brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "ece_15": _ece(p, y),
        "mean_confidence": float(p.max(axis=1).mean()),
    }


def _group_metrics(
    probabilities: np.ndarray, labels: np.ndarray, group_ids: np.ndarray
) -> dict[str, Any]:
    groups = sorted(set(map(str, group_ids)))
    per_group = {
        group: _metrics(probabilities[group_ids == group], labels[group_ids == group])
        for group in groups
    }
    macro = {
        metric: float(np.mean([row[metric] for row in per_group.values()]))
        for metric in ("accuracy", "nll", "brier", "ece_15", "mean_confidence")
    }
    return {
        "overall": _metrics(probabilities, labels),
        "macro_by_prompt": macro,
        "per_prompt": per_group,
        "prompt_group_count": len(groups),
    }


class PairwiseActionScorer:
    """Factory class; concrete torch modules are built after importing torch."""

    @staticmethod
    def create(mode: str, torch: Any):
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.mode = mode
                self.stream_names = MODE_STREAMS[mode]
                self.video_gru_enabled = mode.endswith("video_gru") or mode == "video_gru_only"
                self.projections = nn.ModuleDict()
                for stream in self.stream_names:
                    if stream == "text":
                        width = 384
                    elif stream == "state":
                        width = 8 * 14
                    else:
                        width = 768
                    if stream == "video" and self.video_gru_enabled:
                        self.video_gru = nn.GRU(768, 64, batch_first=True)
                        self.projections[stream] = nn.Identity()
                    else:
                        self.projections[stream] = nn.Sequential(
                            nn.Linear(width, 64), nn.GELU(), nn.LayerNorm(64)
                        )
                self.fusion = nn.Sequential(
                    nn.Linear(64 * len(self.stream_names), 128),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.GELU(),
                )
                self.action_encoder = nn.Sequential(nn.Linear(14, 64), nn.GELU(), nn.LayerNorm(64))
                self.scorer = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1),
                )

            def forward(self, inputs, candidate_actions):
                streams = []
                for stream in self.stream_names:
                    if stream == "video" and self.video_gru_enabled:
                        sequence, _ = self.video_gru(inputs["video_sequence"])
                        encoded = sequence[:, -1, :]
                    else:
                        encoded_input = inputs[stream]
                        if stream == "state":
                            encoded_input = encoded_input.reshape(len(encoded_input), -1)
                        encoded = self.projections[stream](encoded_input)
                    streams.append(encoded)
                context = self.fusion(torch.cat(streams, dim=-1))
                candidate = self.action_encoder(candidate_actions)
                context = context[:, None, :].expand(-1, candidate.shape[1], -1)
                return self.scorer(torch.cat([context, candidate], dim=-1)).squeeze(-1)

        return Model()


MODE_STREAMS = {
    "text_only": ["text"],
    "state_only": ["state"],
    "video_mean_only": ["video"],
    "video_gru_only": ["video"],
    "text_state": ["text", "state"],
    "text_video_mean": ["text", "video"],
    "text_state_video_mean": ["text", "state", "video"],
    "text_state_video_gru": ["text", "state", "video"],
}


def _normalizer(values: np.ndarray, axes: tuple[int, ...]):
    mean = values.mean(axis=axes, keepdims=True)
    std = values.std(axis=axes, keepdims=True)
    std = np.maximum(std, 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def _paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    group_ids: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    groups = sorted(set(map(str, group_ids)))
    left_acc = np.asarray([np.mean(left[group_ids == group]) for group in groups], dtype=np.float64)
    right_acc = np.asarray(
        [np.mean(right[group_ids == group]) for group in groups], dtype=np.float64
    )
    differences = right_acc - left_acc
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(replicates, len(groups)))
    means = differences[sampled].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "unit": "exact prompt group",
        "groups": len(groups),
        "replicates": replicates,
        "seed": seed,
        "direction": "right minus left Accuracy",
        "mean_group_difference": float(differences.mean()),
        "percentile_95_ci": [float(low), float(high)],
        "bootstrap_fraction_right_better": float(np.mean(means > 0)),
        "caveat": (
            f"descriptive project-development evidence; {len(groups)} prompt clusters "
            "and selector use"
        ),
    }


def _build_examples(feature_path: Path, text_path: Path):
    with np.load(feature_path, allow_pickle=False) as data:
        video = data["temporal_video_features"].astype(np.float32)
        pooled = data["pooled_video_features"].astype(np.float32)
        state = data["aligned_state_windows"].astype(np.float32)
        target_actions = data["target_action_vectors"].astype(np.float32)
        episode_ids = data["episode_ids"].astype(np.int64)
        splits = data["split"].astype(str)
        local_group = data["group_index"].astype(np.int64)
        option_index = data["candidate_index"].astype(np.int64)
        prompt_hashes = data["prompt_sha256"].astype(str)
        candidate_options = data["action_options"].astype(np.float32)
        progress = data["relative_progress"].astype(np.float32)
    with np.load(text_path, allow_pickle=False) as data:
        text_features = data["features"].astype(np.float32)
        text_hashes = data["prompt_sha256"].astype(str)
        text_splits = data["split"].astype(str)
    text_by_hash = {key: text_features[i] for i, key in enumerate(text_hashes)}
    if len(text_by_hash) != len(text_hashes):
        raise ValueError("duplicate text prompt feature hashes")
    query_count = len(progress)
    episode_count = len(video)
    train_prompt_count = len(set(prompt_hashes[splits == "train"]))
    development_prompt_count = len(set(prompt_hashes[splits == "development"]))
    prompt_count = train_prompt_count + development_prompt_count
    if video.shape[1:] != (query_count, 8, 768):
        raise ValueError(f"unexpected VideoPrism feature shape: {video.shape}")
    if state.shape != (episode_count, query_count, 8, 14) or pooled.shape != (
        episode_count,
        query_count,
        768,
    ):
        raise ValueError("unexpected state/video feature cache shape")
    if target_actions.shape != (episode_count, query_count, 14):
        raise ValueError("unexpected target-action feature cache shape")
    if candidate_options.shape != (prompt_count, query_count, 4, 14):
        raise ValueError("unexpected four-choice action array shape")
    if set(text_hashes) != set(prompt_hashes) or len(text_hashes) != prompt_count:
        raise ValueError("MiniLM prompts do not exactly match feature-cache prompts")
    if not np.array_equal(
        text_splits,
        np.asarray(["train"] * train_prompt_count + ["development"] * development_prompt_count),
    ):
        raise ValueError("MiniLM prompt partition ordering changed")
    for split, group_count in (
        ("train", train_prompt_count),
        ("development", development_prompt_count),
    ):
        group_ids = sorted(set(map(int, local_group[splits == split])))
        if group_ids != list(range(group_count)):
            raise ValueError(f"{split} prompt group indices are incomplete or non-contiguous")

    examples: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "development"):
        keys = {
            "video": [],
            "video_sequence": [],
            "state": [],
            "text": [],
            "actions": [],
            "labels": [],
            "groups": [],
            "ids": [],
            "prompt_hashes": [],
            "progress_indices": [],
            "candidate_indices": [],
        }
        for episode_index in range(len(video)):
            if splits[episode_index] != split:
                continue
            group = int(local_group[episode_index])
            candidate_group = group if split == "train" else train_prompt_count + group
            for progress_index in range(query_count):
                keys["video"].append(pooled[episode_index, progress_index])
                keys["video_sequence"].append(video[episode_index, progress_index])
                keys["state"].append(state[episode_index, progress_index])
                keys["text"].append(text_by_hash[prompt_hashes[episode_index]])
                keys["actions"].append(candidate_options[candidate_group, progress_index])
                keys["labels"].append(option_index[episode_index])
                target_action = target_actions[episode_index, progress_index]
                selected_action = candidate_options[
                    candidate_group, progress_index, option_index[episode_index]
                ]
                if np.max(np.abs(target_action - selected_action)) > 1e-6:
                    raise ValueError("query target is not the corresponding candidate option")
                keys["groups"].append(prompt_hashes[episode_index])
                keys["ids"].append(f"{episode_ids[episode_index]}@{progress[progress_index]:.2f}")
                keys["prompt_hashes"].append(prompt_hashes[episode_index])
                keys["progress_indices"].append(progress_index)
                keys["candidate_indices"].append(option_index[episode_index])
        examples[split] = {
            "video": np.asarray(keys["video"], dtype=np.float32),
            "video_sequence": np.asarray(keys["video_sequence"], dtype=np.float32),
            "state": np.asarray(keys["state"], dtype=np.float32),
            "text": np.asarray(keys["text"], dtype=np.float32),
            "actions": np.asarray(keys["actions"], dtype=np.float32),
            "labels": np.asarray(keys["labels"], dtype=np.int64),
            "groups": np.asarray(keys["groups"]),
            "ids": np.asarray(keys["ids"]),
            "prompt_hashes": np.asarray(keys["prompt_hashes"]),
            "progress_indices": np.asarray(keys["progress_indices"], dtype=np.int64),
            "candidate_indices": np.asarray(keys["candidate_indices"], dtype=np.int64),
        }
        expected_query_count = int(np.count_nonzero(splits == split)) * query_count
        if len(examples[split]["labels"]) != expected_query_count:
            raise ValueError(f"unexpected {split} query count")
    return examples, candidate_options


def _fit_normalizers(train: dict[str, np.ndarray], candidate_options: np.ndarray):
    train_group_count = len(set(map(str, train["groups"])))
    train_video_groups = candidate_options[:train_group_count]
    return {
        "text": _normalizer(train["text"], (0,)),
        "state": _normalizer(train["state"], (0, 1)),
        "video": _normalizer(train["video_sequence"], (0, 1)),
        "actions": _normalizer(train_video_groups, (0, 1, 2)),
    }


def _apply_normalizers(examples: dict[str, np.ndarray], normalizers):
    output = dict(examples)
    for key, normalizer_key in (
        ("text", "text"),
        ("state", "state"),
    ):
        mean, std = normalizers[normalizer_key]
        output[key] = (examples[key] - mean.squeeze()) / std.squeeze()
    mean, std = normalizers["video"]
    output["video_sequence"] = (examples["video_sequence"] - mean.squeeze()) / std.squeeze()
    output["video"] = output["video_sequence"].mean(axis=1)
    mean, std = normalizers["actions"]
    output["actions"] = (examples["actions"] - mean.squeeze()) / std.squeeze()
    return output


def _torch_inputs(examples: dict[str, np.ndarray], indices: np.ndarray, torch: Any):
    inputs = {
        key: torch.as_tensor(examples[key][indices], dtype=torch.float32)
        for key in ("text", "state", "video", "video_sequence")
    }
    candidates = torch.as_tensor(examples["actions"][indices], dtype=torch.float32)
    labels = torch.as_tensor(examples["labels"][indices], dtype=torch.long)
    return inputs, candidates, labels


def _forward_probabilities(model, examples, mode: str, torch: Any, batch_size: int = 64):
    probabilities = []
    labels = examples["labels"]
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            indices = np.arange(start, min(start + batch_size, len(labels)))
            inputs, actions, _ = _torch_inputs(examples, indices, torch)
            logits = model(inputs, actions)
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def run(config_path: Path, feature_path: Path, text_path: Path, output_root: Path):
    started = time.perf_counter()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite readout run: {output_root}")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    sample_path = config_path.parents[2] / config["sample_manifest"]
    sample_hash = sha256_file(sample_path)
    if sample_hash != config["sample_manifest_sha256"]:
        raise ValueError("experiment config does not match sample manifest")
    if config["seed"] != 17:
        raise ValueError("this preregistered run requires seed 17")
    random.seed(17)
    np.random.seed(17)

    import psutil
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_initialized():
        raise RuntimeError("unexpected CUDA initialization in CPU-only probe")
    examples, candidate_options = _build_examples(feature_path, text_path)
    normalizers = _fit_normalizers(examples["train"], candidate_options)
    train = _apply_normalizers(examples["train"], normalizers)
    development = _apply_normalizers(examples["development"], normalizers)
    output_root.mkdir(parents=True, exist_ok=False)
    histories = {}
    selected_models = {}
    selected_dev_probabilities = {}
    per_arm = {}
    process = psutil.Process()
    rss_start = process.memory_info().rss

    for mode in config["arms"]:
        if mode not in MODE_STREAMS:
            raise ValueError(f"unknown preregistered arm: {mode}")
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        model = PairwiseActionScorer.create(mode, torch)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        train_labels = train["labels"]
        val_labels = development["labels"]
        history = []
        best_nll = float("inf")
        best_epoch = 0
        best_state = None
        stale = 0
        order_rng = np.random.default_rng(17)
        for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
            model.train()
            permutation = order_rng.permutation(len(train_labels))
            for offset in range(0, len(permutation), int(config["training"]["batch_size"])):
                batch_ids = permutation[offset : offset + int(config["training"]["batch_size"])]
                inputs, actions, labels = _torch_inputs(train, batch_ids, torch)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(inputs, actions), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss in {mode} epoch {epoch}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["gradient_clip_norm"])
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient in {mode} epoch {epoch}")
                optimizer.step()
            model.eval()
            train_prob = _forward_probabilities(model, train, mode, torch)
            dev_prob = _forward_probabilities(model, development, mode, torch)
            train_metrics = _group_metrics(train_prob, train_labels, train["groups"])
            dev_metrics = _group_metrics(dev_prob, val_labels, development["groups"])
            current_nll = dev_metrics["macro_by_prompt"]["nll"]
            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "development": dev_metrics,
                    "last_gradient_norm": float(grad_norm.detach().cpu()),
                }
            )
            if current_nll < best_nll - float(config["training"]["min_delta"]):
                best_nll = current_nll
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= int(config["training"]["early_stopping_patience"]):
                break
        if best_state is None:
            raise RuntimeError(f"no selected checkpoint for {mode}")
        model.load_state_dict(best_state)
        model.eval()
        train_prob = _forward_probabilities(model, train, mode, torch)
        dev_prob = _forward_probabilities(model, development, mode, torch)
        train_metrics = _group_metrics(train_prob, train_labels, train["groups"])
        dev_metrics = _group_metrics(dev_prob, val_labels, development["groups"])
        checkpoint_path = output_root / f"{mode}-best.pt"
        torch.save(
            {
                "mode": mode,
                "best_epoch": best_epoch,
                "state_dict": best_state,
                "normalizers": {
                    key: (torch.as_tensor(mean), torch.as_tensor(std))
                    for key, (mean, std) in normalizers.items()
                },
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "sample_manifest_sha256": sample_hash,
            },
            checkpoint_path,
        )
        # Reload in a fresh scorer instance and compare probabilities exactly.
        reloaded = PairwiseActionScorer.create(mode, torch)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reloaded.load_state_dict(loaded["state_dict"])
        reloaded.eval()
        reload_prob = _forward_probabilities(reloaded, development, mode, torch)
        reload_delta = float(np.max(np.abs(reload_prob - dev_prob)))
        if reload_delta > 1e-7:
            raise ValueError(f"save/reload probability mismatch for {mode}: {reload_delta}")
        histories[mode] = history
        selected_models[mode] = reloaded
        selected_dev_probabilities[mode] = dev_prob
        per_arm[mode] = {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "prequantization_total_parameters": (
                (114_365_184 if "video" in MODE_STREAMS[mode] else 0)
                + (17_389_824 if "text" in MODE_STREAMS[mode] else 0)
                + sum(p.numel() for p in model.parameters())
            ),
            "train_metrics": train_metrics,
            "development_metrics": dev_metrics,
            "checkpoint": {
                "path": checkpoint_path.name,
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
            },
            "save_reload_max_probability_delta": reload_delta,
        }

    # Candidate ordering must not affect the fixed joint scorer's probabilities.
    joint_mode = "text_state_video_gru"
    joint_model = selected_models[joint_mode]
    permuted = dict(development)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    permuted["actions"] = development["actions"][:, permutation]
    joint_probability = _forward_probabilities(joint_model, development, joint_mode, torch)
    permuted_probability = _forward_probabilities(joint_model, permuted, joint_mode, torch)
    restored = np.empty_like(permuted_probability)
    restored[:, permutation] = permuted_probability
    option_permutation_delta = float(np.max(np.abs(joint_probability - restored)))
    if option_permutation_delta > 1e-6:
        raise ValueError("candidate option permutation changed scores beyond tolerance")

    fixed_correctness = {
        mode: probability.argmax(axis=1) == development["labels"]
        for mode, probability in selected_dev_probabilities.items()
    }
    paired_comparisons = {}
    for right_mode, left_mode in (
        ("text_state_video_gru", "text_only"),
        ("text_state_video_gru", "state_only"),
        ("text_state_video_gru", "video_mean_only"),
        ("text_state_video_gru", "video_gru_only"),
        ("text_state_video_gru", "text_state_video_mean"),
        ("video_gru_only", "video_mean_only"),
    ):
        paired_comparisons[f"{right_mode}_minus_{left_mode}"] = _paired_bootstrap(
            fixed_correctness[left_mode].astype(np.float64),
            fixed_correctness[right_mode].astype(np.float64),
            development["groups"],
        )

    # Ablate and misalign one stream in the fixed preregistered joint checkpoint.
    control_results = {}
    val_groups = development["groups"]
    val_labels = development["labels"]
    controls = {}
    for stream in ("text", "state", "video_sequence"):
        altered = dict(development)
        altered[stream] = np.zeros_like(development[stream])
        if stream == "video_sequence":
            altered["video"] = np.zeros_like(development["video"])
        controls[f"zero_{stream}"] = altered
    for stream, companion in (("video_sequence", "video"), ("state", None)):
        altered = dict(development)
        matrix = development[stream].copy()
        for group in sorted(set(map(str, val_groups))):
            group_rows = np.flatnonzero(val_groups == group)
            for time_index in range(3):
                rows = group_rows[development["progress_indices"][group_rows] == time_index]
                rows = rows[np.argsort(development["candidate_indices"][rows])]
                if len(rows) != 4:
                    raise ValueError("expected four synchronized queries per prompt/time")
                matrix[rows] = np.roll(matrix[rows], shift=1, axis=0)
        altered[stream] = matrix
        if companion:
            altered[companion] = matrix.mean(axis=1)
        controls[f"misaligned_{'video' if stream == 'video_sequence' else 'state'}"] = altered
    for name, altered in controls.items():
        probability = _forward_probabilities(joint_model, altered, joint_mode, torch)
        control_correctness = probability.argmax(axis=1) == val_labels
        control_results[name] = {
            "metrics": _group_metrics(probability, val_labels, val_groups),
            "per_example_probabilities": probability.tolist(),
            "paired_accuracy_vs_default_joint": _paired_bootstrap(
                control_correctness.astype(np.float64),
                fixed_correctness[joint_mode].astype(np.float64),
                val_groups,
            ),
        }

    # Per-query probabilities for every selected arm.
    prediction_lines = []
    for mode, model in selected_models.items():
        probability = _forward_probabilities(model, development, mode, torch)
        for index, identifier in enumerate(development["ids"]):
            prediction_lines.append(
                json.dumps(
                    {
                        "arm": mode,
                        "query_id": str(identifier),
                        "prompt_sha256": str(development["prompt_hashes"][index]),
                        "target_index": int(development["labels"][index]),
                        "prediction_index": int(probability[index].argmax()),
                        "probabilities": probability[index].tolist(),
                    },
                    separators=(",", ":"),
                )
            )
    predictions_path = output_root / "development-predictions.jsonl"
    predictions_path.write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")

    # Save epoch histories separately to keep the primary result compact.
    histories_path = output_root / "training-history.json"
    histories_path.write_text(json.dumps(histories, indent=2) + "\n", encoding="utf-8")
    rss_end = process.memory_info().rss
    report = {
        "schema_version": "robopro-frozen-multimodal-readout-run-v1",
        "status": "complete_development_only",
        "experiment_id": config["experiment_id"],
        "seed": 17,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "sample_manifest_sha256": sample_hash,
        "video_feature_cache_sha256": sha256_file(feature_path),
        "text_feature_cache_sha256": sha256_file(text_path),
        "video_model": {
            "revision": "be719a406d563b66f0ac969e7c94bab8e997c81a",
            "parameters": 114_365_184,
            "frozen": True,
        },
        "text_model": {
            "revision": "4ca70771034acceecb2e72475f72050fcdde4ddc",
            "parameters": 17_389_824,
            "frozen": True,
        },
        "split": {
            "train_queries": len(train["labels"]),
            "development_queries": len(development["labels"]),
            "train_prompt_groups": len(set(map(str, train["groups"]))),
            "development_prompt_groups": len(set(map(str, development["groups"]))),
        },
        "runtime": {
            "device": "CPU",
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "gpu_used": False,
            "wall_seconds": time.perf_counter() - started,
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
            "cloud_compute_cost": 0,
        },
        "arms": per_arm,
        "paired_group_bootstrap_comparisons": paired_comparisons,
        "joint_fixed_checkpoint_controls": {
            "checkpoint_arm": joint_mode,
            "option_permutation_max_probability_delta": option_permutation_delta,
            "option_permutation_argmax_flips": 0,
            "controls": control_results,
        },
        "prediction_file": {
            "path": predictions_path.name,
            "sha256": sha256_file(predictions_path),
        },
        "training_history_file": {
            "path": histories_path.name,
            "sha256": sha256_file(histories_path),
        },
        "official_test_or_sealed_audit_opened": False,
        "product_promotion": False,
        "limitations": [
            (
                f"Development has {len(set(map(str, development['groups'])))} prompt groups; "
                "intervals are descriptive and development is selection-exposed."
            ),
            "The target is retrieval of one recorded command, not action success or safety.",
            "The prompt-group split does not prove scene-held-out generalization.",
            "This is one seed and one bounded comparison; no hyperparameter search was run.",
        ],
    }
    report_path = output_root / "run-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_sha = sha256_file(report_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_report": str(report_path),
                "run_report_sha256": report_sha,
                "arms": {
                    key: {
                        "best_epoch": value["best_epoch"],
                        "dev_macro_accuracy": value["development_metrics"]["macro_by_prompt"][
                            "accuracy"
                        ],
                        "dev_macro_nll": value["development_metrics"]["macro_by_prompt"]["nll"],
                    }
                    for key, value in per_arm.items()
                },
                "joint_controls": {
                    key: value["metrics"]["macro_by_prompt"]["accuracy"]
                    for key, value in control_results.items()
                },
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.config.resolve(strict=True),
        args.video_features.resolve(strict=True),
        args.text_features.resolve(strict=True),
        args.output_root.resolve(),
    )


if __name__ == "__main__":
    main()
