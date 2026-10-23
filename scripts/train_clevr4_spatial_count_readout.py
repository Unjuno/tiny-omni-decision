"""Matched CPU/GPU readout comparison on frozen CLEVR-4 spatial features."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file  # noqa: E402
from tiny_omni_decision.spatial_count import CandidateConditionedSpatialScorer  # noqa: E402


class MeanPooledCandidateScorer(nn.Module):
    def __init__(self, visual_size: int = 768, option_size: int = 384, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(visual_size + option_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, visual: Tensor, options: Tensor) -> Tensor:
        pooled = visual.mean(dim=1)
        expanded = pooled[:, None, :].expand(-1, options.shape[1], -1)
        return self.network(torch.cat((expanded, options), dim=-1)).squeeze(-1)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--option-embeddings", type=Path, required=True)
    parser.add_argument("--feature-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    rows = np.arange(len(targets))
    correct = probs.argmax(axis=1) == targets
    confidence = probs.max(axis=1)
    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        mask = (
            confidence >= boundaries[index] if index == 0 else confidence > boundaries[index]
        ) & (confidence <= boundaries[index + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    one_hot = np.eye(logits.shape[1], dtype=np.float64)[targets]
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(probs[rows, targets], 1e-12, 1.0)).mean()),
        "brier": float(((probs - one_hot) ** 2).sum(axis=1).mean()),
        "ece_15_equal_width": ece,
    }


def _macro(metrics: dict[str, dict[str, float | int]]) -> dict[str, float]:
    keys = ("accuracy", "nll", "brier", "ece_15_equal_width")
    return {key: float(np.mean([metrics[name][key] for name in CLEVR4_TAXONOMIES])) for key in keys}


def _questions(rows: list[dict[str, Any]]) -> list[tuple[int, int, int, str]]:
    result = []
    for image_index, row in enumerate(rows):
        for taxonomy_index, (taxonomy, options) in enumerate(CLEVR4_TAXONOMIES.items()):
            target = str(row[taxonomy])
            if target not in options:
                raise ValueError(f"unknown {taxonomy} label: {target}")
            result.append((image_index, taxonomy_index, options.index(target), taxonomy))
    return result


def _load_option_tensor(path: Path) -> Tensor:
    option_map = np.load(path, allow_pickle=False)
    matrix = np.stack(
        [
            option_map[f"{taxonomy}__{option}"]
            for taxonomy, options in CLEVR4_TAXONOMIES.items()
            for option in options
        ]
    ).reshape(len(CLEVR4_TAXONOMIES), 10, 384)
    if not np.isfinite(matrix).all():
        raise ValueError("option embedding cache contains NaN or Inf")
    return torch.from_numpy(matrix.astype(np.float32))


def _evaluate(
    model: nn.Module,
    feature_array: np.ndarray,
    questions: list[tuple[int, int, int, str]],
    option_tensor: Tensor,
    device: torch.device,
    batch_size: int,
    image_ids: list[str],
) -> tuple[dict[str, dict[str, float | int]], list[dict[str, Any]]]:
    model.eval()
    collected_logits: dict[str, list[np.ndarray]] = {name: [] for name in CLEVR4_TAXONOMIES}
    collected_targets: dict[str, list[int]] = {name: [] for name in CLEVR4_TAXONOMIES}
    prediction_rows = []
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            rows = questions[start : start + batch_size]
            ids = np.asarray([item[0] for item in rows], dtype=np.int64)
            visual_np = np.asarray(feature_array[ids], dtype=np.float32)
            taxonomy_ids = torch.tensor([item[1] for item in rows], dtype=torch.long)
            choices = option_tensor[taxonomy_ids].to(device)
            visual = torch.from_numpy(visual_np).to(device)
            logits = model(visual, choices).float().cpu().numpy()
            for row, values in zip(rows, logits, strict=True):
                image_index, taxonomy_index, target, taxonomy = row
                shifted = values - values.max()
                probabilities = np.exp(shifted)
                probabilities /= probabilities.sum()
                option_names = list(CLEVR4_TAXONOMIES.values())[taxonomy_index]
                collected_logits[taxonomy].append(values)
                collected_targets[taxonomy].append(target)
                prediction_rows.append(
                    {
                        "image_id": image_ids[image_index],
                        "taxonomy": taxonomy,
                        "options": option_names,
                        "target": option_names[target],
                        "prediction": option_names[int(values.argmax())],
                        "probabilities": probabilities.tolist(),
                    }
                )
    by_taxonomy = {
        name: _metrics(
            np.stack(collected_logits[name]), np.asarray(collected_targets[name], dtype=np.int64)
        )
        for name in CLEVR4_TAXONOMIES
    }
    return {"by_taxonomy": by_taxonomy, "macro": _macro(by_taxonomy)}, prediction_rows


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fit_arm(
    arm: str,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_questions: list[tuple[int, int, int, str]],
    validation_questions: list[tuple[int, int, int, str]],
    train_diagnostic_questions: list[tuple[int, int, int, str]],
    option_tensor: Tensor,
    device: torch.device,
    output_dir: Path,
    config: dict[str, Any],
    train_image_ids: list[str],
    validation_image_ids: list[str],
) -> dict[str, Any]:
    _seed(17)
    model: nn.Module = (
        MeanPooledCandidateScorer().to(device)
        if arm == "mean_pooled_mlp"
        else CandidateConditionedSpatialScorer().to(device)
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = {"mean_pooled_mlp": 147713, "option_conditioned_spatial": 311553}
    if parameter_count != expected_parameters[arm]:
        raise ValueError(f"unexpected trainable parameter count for {arm}: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    rng = random.Random(17)
    history = []
    best_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    started = time.perf_counter()
    batch_size = int(config["comparison"]["batch_questions"])

    for epoch in range(1, int(config["comparison"]["epochs"]) + 1):
        order = list(range(len(train_questions)))
        rng.shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_questions = [train_questions[index] for index in indices]
            image_indices = np.asarray([row[0] for row in batch_questions], dtype=np.int64)
            visuals = torch.from_numpy(
                np.asarray(train_features[image_indices], dtype=np.float32)
            ).to(device)
            taxonomy_indices = torch.tensor([row[1] for row in batch_questions], dtype=torch.long)
            options = option_tensor[taxonomy_indices].to(device)
            targets = torch.tensor(
                [row[2] for row in batch_questions], dtype=torch.long, device=device
            )
            logits = model(visuals, options)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_result, _ = _evaluate(
            model,
            validation_features,
            validation_questions,
            option_tensor,
            device,
            batch_size,
            validation_image_ids,
        )
        train_diagnostic_result, _ = _evaluate(
            model,
            train_features,
            train_diagnostic_questions,
            option_tensor,
            device,
            batch_size,
            train_image_ids,
        )
        selector = validation_result["macro"]["nll"]
        history.append(
            {
                "epoch": epoch,
                "rolling_train_ce": float(np.mean(losses)),
                "fixed_train_diagnostic": train_diagnostic_result,
                "validation": validation_result,
                "train_validation_macro_accuracy_gap": (
                    validation_result["macro"]["accuracy"]
                    - train_diagnostic_result["macro"]["accuracy"]
                ),
                "train_validation_macro_nll_gap": (
                    validation_result["macro"]["nll"]
                    - train_diagnostic_result["macro"]["nll"]
                ),
                "validation_selector_macro_nll": selector,
            }
        )
        if selector < best_nll:
            best_nll = selector
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            torch.save(best_state, output_dir / f"{arm}-best.pt")
    if best_state is None:
        raise AssertionError("training produced no selected checkpoint")
    torch.save(model.state_dict(), output_dir / f"{arm}-final.pt")
    model.load_state_dict(
        torch.load(output_dir / f"{arm}-best.pt", map_location="cpu", weights_only=True)
    )
    model.to(device)
    selected_result, predictions = _evaluate(
        model,
        validation_features,
        validation_questions,
        option_tensor,
        device,
        batch_size,
        validation_image_ids,
    )
    if not np.isclose(
        selected_result["macro"]["nll"], best_nll, rtol=1e-7, atol=1e-7
    ):
        raise ValueError("best checkpoint reload does not reproduce the selected validation NLL")
    (output_dir / f"{arm}-validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    return {
        "arm": arm,
        "trainable_parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_macro_nll": best_nll,
        "selected_validation": selected_result,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "best_checkpoint_sha256": sha256_file(output_dir / f"{arm}-best.pt"),
        "final_checkpoint_sha256": sha256_file(output_dir / f"{arm}-final.pt"),
        "prediction_sha256": sha256_file(output_dir / f"{arm}-validation-predictions.jsonl"),
        "rolling_train_metric_note": (
            "rolling minibatch CE; fixed-train diagnostic metrics are recorded separately"
        ),
    }


def main() -> None:
    args = _args()
    split_dir = args.split_dir.resolve(strict=True)
    feature_dir = args.feature_dir.resolve(strict=True)
    metadata_path = args.feature_metadata.resolve(strict=True)
    option_path = args.option_embeddings.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("seed") != 17 or config.get("comparison", {}).get("checkpoint_selection") != (
        "minimum_macro_validation_nll_across_four_taxonomies"
    ):
        raise ValueError("unexpected experiment configuration")
    feature_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        feature_metadata.get("status") != "complete"
        or feature_metadata.get("full_corpus") is not True
    ):
        raise ValueError("feature cache is not marked complete")
    expected_checkpoint = config["encoder"]["checkpoint_sha256"]
    if feature_metadata.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError("feature cache checkpoint differs from the frozen config")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes < 8 * 1024**3:
            raise RuntimeError("at least 8 GiB free VRAM is required for this readout run")
    train_features = np.load(feature_dir / "train-spatial-features.npy", mmap_mode="r")
    validation_features = np.load(feature_dir / "validation-spatial-features.npy", mmap_mode="r")
    train_rows = _jsonl(split_dir / "train-labels.jsonl")
    validation_rows = _jsonl(split_dir / "validation-labels.jsonl")
    if train_features.shape != (len(train_rows), 64, 768) or validation_features.shape != (
        len(validation_rows),
        64,
        768,
    ):
        raise ValueError("feature arrays do not align with frozen split labels")
    train_media = {
        row["id"]: row["sha256"] for row in _jsonl(split_dir / "train-image-manifest.jsonl")
    }
    validation_media = {
        row["id"]: row["sha256"] for row in _jsonl(split_dir / "validation-image-manifest.jsonl")
    }
    if set(train_media) & set(validation_media):
        raise ValueError("training and validation media identity overlaps")
    if len(train_media) != len(train_rows) or len(validation_media) != len(validation_rows):
        raise ValueError("media manifests contain duplicate IDs or differ from labels")
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        ordered_ids = "\n".join(str(row["id"]) for row in rows)
        ordered_hash = hashlib.sha256(ordered_ids.encode()).hexdigest()
        if feature_metadata.get(f"{split}_id_order_sha256") != ordered_hash:
            raise ValueError(f"{split} labels do not match the frozen feature ID order")
        manifest_path = split_dir / f"{split}-image-manifest.jsonl"
        if feature_metadata.get(f"{split}_manifest_sha256") != sha256_file(manifest_path):
            raise ValueError(f"{split} media manifest changed after feature extraction")
        feature_path = feature_dir / f"{split}-spatial-features.npy"
        if feature_metadata.get("feature_hashes", {}).get(split) != sha256_file(feature_path):
            raise ValueError(f"{split} frozen feature file hash mismatch")
    option_tensor = _load_option_tensor(option_path)
    train_questions = _questions(train_rows)
    validation_questions = _questions(validation_rows)
    diagnostic_count = int(config["reporting"]["fixed_train_diagnostic_images"])
    diagnostic_rows = train_rows[: min(diagnostic_count, len(train_rows))]
    train_diagnostic_questions = _questions(diagnostic_rows)
    train_image_ids = [str(row["id"]) for row in train_rows]
    validation_image_ids = [str(row["id"]) for row in validation_rows]
    results = {}
    for arm in ("mean_pooled_mlp", "option_conditioned_spatial"):
        results[arm] = _fit_arm(
            arm,
            train_features,
            validation_features,
            train_questions,
            validation_questions,
            train_diagnostic_questions,
            option_tensor,
            device,
            output,
            config,
            train_image_ids,
            validation_image_ids,
        )
    metadata = {
        "status": "complete",
        "experiment_id": config["experiment_id"],
        "seed": 17,
        "config_sha256": sha256_file(config_path),
        "effective_config": config,
        "trainer_script_sha256": sha256_file(Path(__file__).resolve()),
        "feature_metadata_sha256": sha256_file(metadata_path),
        "split_summary_sha256": sha256_file(split_dir / "split-summary.json"),
        "option_embedding_sha256": sha256_file(option_path),
        "train_images": len(train_rows),
        "validation_images": len(validation_rows),
        "unique_train_media": len(set(train_media.values())),
        "unique_validation_media": len(set(validation_media.values())),
        "train_questions": len(train_questions),
        "validation_questions": len(validation_questions),
        "device": str(device),
        "base_encoder_parameters": int(config["encoder"]["parameters"]),
        "total_parameters_before_quantization": {
            arm: int(config["encoder"]["parameters"] + result["trainable_parameters"])
            for arm, result in results.items()
        },
        "results": results,
        "sealed_audit_loaded": False,
        "prior_observed_validation_loaded": False,
        "validation_use": "new train-derived development split; used for model selection",
    }
    (output / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output / "effective-config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "training-history.json").write_text(
        json.dumps({key: value["history"] for key, value in results.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
