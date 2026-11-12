"""Fit a frozen DINOv2 screen-action Path A readout on pinned MiniWoB rows.

All model execution is local and CPU-only. The collector audit is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoModel

from scripts.miniwob_screen_action_data import (
    model_option_positions,
    output_option_ids,
    validate_screen_action_row,
)
from tiny_omni_decision.miniwob_dino_readout import (
    MiniWoBOptionScorer,
    classification_metrics,
    finite_json_metrics,
    permute_candidate_axis,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = ROOT / "configs/pretrained_reuse/miniwob_dino_path_a_v1.json"
PIXEL_FEATURE_WIDTH = 1152
FEATURE_WIDTH = 1154
MODEL_SIZE = 224
PATCH_SIZE = 14
PATCH_GRID = MODEL_SIZE // PATCH_SIZE
MODEL_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
MODEL_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_corpus(
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    corpus = config["corpus"]
    root = Path(corpus["path"]).resolve(strict=True)
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != corpus["manifest_sha256"]:
        raise ValueError("frozen MiniWoB manifest hash mismatch")
    manifest = read_json(manifest_path)
    if (
        manifest.get("status") != "FROZEN_DEVELOPMENT_CORPUS_NO_MODEL_EVALUATION"
        or manifest.get("task") != corpus["task"]
        or manifest.get("quality_metrics_reported") is not False
        or manifest.get("sealed_audit_created") is not False
    ):
        raise ValueError("unexpected corpus identity or evaluation state")

    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        jsonl = root / f"{split}.jsonl"
        expected_hash = corpus[f"{split}_jsonl_sha256"]
        if manifest["splits"][split]["examples_sha256"] != expected_hash:
            raise ValueError(f"{split} manifest and experiment config hashes disagree")
        if sha256_file(jsonl) != expected_hash:
            raise ValueError(f"{split} JSONL hash mismatch")
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
        if len(rows) != corpus["expected_examples_per_split"]:
            raise ValueError(f"{split} row count differs from frozen configuration")
        groups: set[str] = set()
        screenshots: set[str] = set()
        for row in rows:
            validate_screen_action_row(row)
            if row["task_template"] != corpus["task"] or len(row["options"]) != 4:
                raise ValueError("task template or four-option contract mismatch")
            if row["sample_id"] in groups or row["episode_group_id"] in groups:
                raise ValueError(f"duplicate {split} sample/group")
            groups.add(row["sample_id"])
            groups.add(row["episode_group_id"])
            rel = Path(row["screenshot_path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("screenshot path escapes frozen corpus root")
            image_path = (root / rel).resolve(strict=True)
            if root not in image_path.parents:
                raise ValueError("screenshot path resolves outside frozen corpus root")
            if sha256_file(image_path) != row["screenshot_sha256"]:
                raise ValueError(f"{split} screenshot hash mismatch: {row['sample_id']}")
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                if [rgb.height, rgb.width, 3] != row["screenshot_shape"]:
                    raise ValueError("screenshot dimensions differ from frozen row")
                pixels_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
            if pixels_hash in screenshots:
                raise ValueError(f"duplicate screenshot pixels within {split}")
            screenshots.add(pixels_hash)
        splits[split] = rows

    train, validation = splits["train"], splits["validation"]
    train_ids = {row["sample_id"] for row in train}
    val_ids = {row["sample_id"] for row in validation}
    train_groups = {row["episode_group_id"] for row in train}
    val_groups = {row["episode_group_id"] for row in validation}
    train_media = {row["screenshot_sha256"] for row in train}
    val_media = {row["screenshot_sha256"] for row in validation}
    train_pixels = {
        hashlib.sha256(
            np.asarray(Image.open(root / row["screenshot_path"]).convert("RGB")).tobytes()
        ).hexdigest()
        for row in train
    }
    val_pixels = {
        hashlib.sha256(
            np.asarray(Image.open(root / row["screenshot_path"]).convert("RGB")).tobytes()
        ).hexdigest()
        for row in validation
    }
    if (
        train_ids & val_ids
        or train_groups & val_groups
        or train_media & val_media
        or train_pixels & val_pixels
    ):
        raise ValueError("cross-split ID, episode, screenshot-file or screenshot-pixel overlap")
    return root, manifest, splits


def load_encoder(config: dict[str, Any]) -> tuple[torch.nn.Module, Path, dict[str, str]]:
    spec = config["encoder"]
    model_dir = Path(spec["artifact_path"]).resolve(strict=True)
    expected_files = {
        "README.md": "4c20dca454a8e5c670e8de5c7e6040f512aeca5438516f7623eedc4e3b00599c",
        "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
        "preprocessor_config.json": spec["processor_config_sha256"],
        "model.safetensors": spec["weight_sha256"],
    }
    actual_hashes = {name: sha256_file(model_dir / name) for name in expected_files}
    for name, expected in expected_files.items():
        if actual_hashes[name] != expected:
            raise ValueError(f"pinned DINOv2 file hash mismatch: {name}")
    upstream = read_json(model_dir / "config.json")
    if upstream.get("model_type") != "dinov2" or upstream.get("hidden_size") != 384:
        raise ValueError("local checkpoint config is not the pinned DINOv2-small architecture")
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device="cpu", dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != spec["parameters"] or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("loaded DINOv2 parameter count or frozen state mismatch")
    return model, model_dir, actual_hashes


def preprocess_screens(paths: list[Path]) -> torch.Tensor:
    batch: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(
                (MODEL_SIZE, MODEL_SIZE), resample=Image.Resampling.BICUBIC
            )
            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        batch.append(((pixels - MODEL_MEAN) / MODEL_STD).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(batch, axis=0).copy())


def encode_screen_features(
    rows: list[dict[str, Any]], root: Path, model: torch.nn.Module
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    batch_size = 8
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            images = preprocess_screens([root / row["screenshot_path"] for row in batch_rows])
            tokens = model(pixel_values=images).last_hidden_state
            if tokens.shape != (len(batch_rows), PATCH_GRID * PATCH_GRID + 1, 384):
                raise ValueError(f"unexpected DINOv2 token shape: {tuple(tokens.shape)}")
            cls = tokens[:, 0]
            patches = tokens[:, 1:].reshape(len(batch_rows), PATCH_GRID, PATCH_GRID, 384)
            global_mean = patches.mean(dim=(1, 2))
            local: list[torch.Tensor] = []
            coordinates: list[list[tuple[float, float]]] = []
            for row_index, row in enumerate(batch_rows):
                row_features: list[torch.Tensor] = []
                row_coordinates: list[tuple[float, float]] = []
                for x_norm, y_norm in model_option_positions(row):
                    column = min(int(x_norm * PATCH_GRID), PATCH_GRID - 1)
                    row_index_patch = min(int(y_norm * PATCH_GRID), PATCH_GRID - 1)
                    x0, x1 = max(0, column - 1), min(PATCH_GRID, column + 2)
                    y0, y1 = max(0, row_index_patch - 1), min(PATCH_GRID, row_index_patch + 2)
                    row_features.append(patches[row_index, y0:y1, x0:x1].mean(dim=(0, 1)))
                    row_coordinates.append((x_norm, y_norm))
                local.append(torch.stack(row_features))
                coordinates.append(row_coordinates)
            local_tokens = torch.stack(local)
            xy = torch.tensor(coordinates, dtype=torch.float32)
            shared = torch.cat((cls, global_mean), dim=1)[:, None, :].expand(-1, 4, -1)
            chunks.append(torch.cat((shared, local_tokens, xy), dim=-1))
    features = torch.cat(chunks)
    if features.shape != (len(rows), 4, FEATURE_WIDTH) or not torch.isfinite(features).all():
        raise ValueError("invalid or non-finite frozen option features")
    return features


def black_screen_features(rows: list[dict[str, Any]], model: torch.nn.Module) -> torch.Tensor:
    with torch.inference_mode():
        image = Image.new("RGB", (160, 210), color=(0, 0, 0))
        pixels = preprocess_screens_from_image(image).unsqueeze(0)
        tokens = model(pixel_values=pixels).last_hidden_state
        patches = tokens[:, 1:].reshape(1, PATCH_GRID, PATCH_GRID, 384)
        shared = torch.cat((tokens[:, 0], patches.mean(dim=(1, 2))), dim=1)[0]
        blank_local = patches[0].mean(dim=(0, 1)).expand(4, -1)
    xy = torch.tensor([model_option_positions(row) for row in rows], dtype=torch.float32)
    return torch.cat(
        (shared.expand(len(rows), 4, -1), blank_local.expand(len(rows), 4, -1), xy), dim=-1
    )


def preprocess_screens_from_image(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB").resize((MODEL_SIZE, MODEL_SIZE), resample=Image.Resampling.BICUBIC)
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    normalized = ((pixels - MODEL_MEAN) / MODEL_STD).transpose(2, 0, 1).copy()
    return torch.from_numpy(normalized).unsqueeze(0)


def targets_for(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([row["target_index"] for row in rows], dtype=torch.long)


def permute_screens(features: torch.Tensor, seed: int) -> tuple[torch.Tensor, list[int]]:
    """Cyclically map every row to a different example within its own split."""
    if features.shape[0] < 2:
        raise ValueError("at least two examples are required for screen permutation")
    order = np.random.default_rng(seed).permutation(features.shape[0]).tolist()
    donor_indices = order[1:] + order[:1]
    if any(receiver == donor for receiver, donor in enumerate(donor_indices)):
        raise ValueError("screen permutation unexpectedly retained a self-pair")
    visual = features[donor_indices, :, :PIXEL_FEATURE_WIDTH]
    result = features.clone()
    result[:, :, :PIXEL_FEATURE_WIDTH] = visual
    return result, donor_indices


def metric_dict(
    model: torch.nn.Module, features: torch.Tensor, targets: torch.Tensor
) -> dict[str, Any]:
    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=1)
    return finite_json_metrics(classification_metrics(probabilities, targets))


def predict_rows(
    model: torch.nn.Module, features: torch.Tensor, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=1)
    records = []
    for row, distribution in zip(rows, probabilities.tolist(), strict=True):
        ids = output_option_ids(row)
        scored = {
            option_id: float(value) for option_id, value in zip(ids, distribution, strict=True)
        }
        selected = max(ids, key=scored.__getitem__)
        records.append(
            {
                "sample_id": row["sample_id"],
                "episode_group_id": row["episode_group_id"],
                "target_index": row["target_index"],
                "target_option_id": ids[row["target_index"]],
                "selected_option_id": selected,
                "probabilities_by_option_id": scored,
            }
        )
    return records, probabilities


def fit_arm(
    name: str,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    output: Path,
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = MiniWoBOptionScorer().cpu()
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    output.mkdir(parents=True, exist_ok=False)
    best_nll = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    best_val_probabilities: torch.Tensor | None = None
    best_path = output / "checkpoint-best.safetensors"
    final_path = output / "checkpoint-final.safetensors"
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for epoch in range(1, settings["max_epochs"] + 1):
        start = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_targets), generator=generator)
        for batch_start in range(0, len(order), settings["batch_size"]):
            indices = order[batch_start : batch_start + settings["batch_size"]]
            logits = model(train_features[indices])
            loss = torch.nn.functional.cross_entropy(logits, train_targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"non-finite gradient in {name} at epoch {epoch}")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError(f"non-finite parameter in {name} at epoch {epoch}")
        model.eval()
        train_metrics = metric_dict(model, train_features, train_targets)
        val_metrics = metric_dict(model, val_features, val_targets)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "seconds": time.perf_counter() - start,
        }
        history.append(record)
        nll = float(val_metrics["nll"])
        if nll < best_nll - settings["min_delta"]:
            best_nll = nll
            best_epoch = epoch
            stale = 0
            with torch.inference_mode():
                best_val_probabilities = torch.softmax(model(val_features), dim=1).clone()
            save_file(
                {key: value.detach().contiguous() for key, value in model.state_dict().items()},
                best_path,
            )
        else:
            stale += 1
        if stale >= settings["early_stopping_patience"]:
            break

    save_file(
        {key: value.detach().contiguous() for key, value in model.state_dict().items()}, final_path
    )
    selected = MiniWoBOptionScorer().cpu()
    selected.load_state_dict(load_file(best_path, device="cpu"), strict=True)
    selected.eval()
    final_model = MiniWoBOptionScorer().cpu()
    final_model.load_state_dict(load_file(final_path, device="cpu"), strict=True)
    final_model.eval()
    selected_train_records, selected_train_probs = predict_rows(
        selected, train_features, train_rows
    )
    selected_val_records, selected_val_probs = predict_rows(selected, val_features, val_rows)
    final_val_records, final_val_probs = predict_rows(final_model, val_features, val_rows)
    if best_val_probabilities is None:
        raise RuntimeError("training failed to produce a selected validation checkpoint")
    best_reload_delta = float((selected_val_probs - best_val_probabilities).abs().max())
    if best_reload_delta > 0:
        raise RuntimeError("best checkpoint save/reload changed validation probabilities")
    metrics = {
        "selected_checkpoint": {
            "epoch": best_epoch,
            "train": classification_metrics(selected_train_probs, train_targets),
            "validation": classification_metrics(selected_val_probs, val_targets),
            "best_reload_max_probability_abs_difference": best_reload_delta,
            "checkpoint_sha256": sha256_file(best_path),
            "final_checkpoint_sha256": sha256_file(final_path),
            "trainable_readout_parameters": trainable,
        },
        "final_checkpoint": {
            "epoch": history[-1]["epoch"],
            "validation": classification_metrics(final_val_probs, val_targets),
        },
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "training-history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history), encoding="utf-8"
    )
    for filename, records in (
        ("train-predictions.jsonl", selected_train_records),
        ("validation-predictions.jsonl", selected_val_records),
        ("validation-final-predictions.jsonl", final_val_records),
    ):
        (output / filename).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
        )
    return metrics


def position_prior(
    train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: Counter[tuple[float, float]] = Counter()
    for row in train_rows:
        counts[model_option_positions(row)[row["target_index"]]] += 1
    rates = {position: counts[position] / len(train_rows) for position in counts}
    logits = {position: float(np.log(max(rate, 1e-6))) for position, rate in rates.items()}
    distributions: list[torch.Tensor] = []
    for row in val_rows:
        scores = torch.tensor(
            [logits.get(position, -20.0) for position in model_option_positions(row)]
        )
        distributions.append(torch.softmax(scores, dim=0))
    probabilities = torch.stack(distributions)
    metrics = classification_metrics(probabilities, targets_for(val_rows))
    records = []
    for row, distribution in zip(val_rows, probabilities.tolist(), strict=True):
        ids = output_option_ids(row)
        scores = dict(zip(ids, distribution, strict=True))
        records.append(
            {
                "sample_id": row["sample_id"],
                "target_option_id": ids[row["target_index"]],
                "selected_option_id": max(ids, key=scores.__getitem__),
                "probabilities_by_option_id": scores,
            }
        )
    return {
        "position_rates": {str(key): value for key, value in rates.items()},
        "validation": metrics,
    }, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.smoke:
        parser.error("--preflight-only and --smoke are separate execution modes")
    config_path = args.config.resolve(strict=True)
    config = read_json(config_path)
    settings = dict(config["readout"])
    output = Path(config["output_path"])
    run_mode = "smoke" if args.smoke else "full"
    if args.smoke:
        output = output.with_name(output.name + "-smoke")
        settings["max_epochs"] = 2
        settings["early_stopping_patience"] = 2
    if output.exists() and not args.preflight_only:
        raise FileExistsError(f"refusing to overwrite existing experiment output: {output}")
    if config["device"] != "cpu" or torch.cuda.is_initialized():
        raise ValueError("this experiment is pinned to CPU and must not initialize CUDA")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")
    if visible_devices not in ("", "-1", "none", "None"):
        raise ValueError("CUDA visibility must be disabled for CPU-only execution")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    root, corpus_manifest, splits = read_corpus(config)
    train_rows, val_rows = splits["train"], splits["validation"]
    if args.smoke:
        train_rows, val_rows = train_rows[:16], val_rows[:16]
    train_targets, val_targets = targets_for(train_rows), targets_for(val_rows)
    model, model_dir, model_hashes = load_encoder(config)
    encoder_count = sum(parameter.numel() for parameter in model.parameters())
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.preflight_only:
        smoke_train = encode_screen_features(train_rows[:2], root, model)
        smoke_val = encode_screen_features(val_rows[:2], root, model)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("frozen DINOv2 encoder unexpectedly became trainable")
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_PASS",
                    "device": "cpu",
                    "corpus_rows_validated": {
                        "train": len(train_rows),
                        "validation": len(val_rows),
                    },
                    "smoke_feature_shapes": {
                        "train": list(smoke_train.shape),
                        "validation": list(smoke_val.shape),
                    },
                    "encoder_parameters": encoder_count,
                    "encoder_hashes": model_hashes,
                    "finite": bool(
                        torch.isfinite(smoke_train).all() and torch.isfinite(smoke_val).all()
                    ),
                    "cuda_used": False,
                },
                indent=2,
            )
        )
        return
    extraction_start = time.perf_counter()
    train_features = encode_screen_features(train_rows, root, model)
    val_features = encode_screen_features(val_rows, root, model)
    extraction_seconds = time.perf_counter() - extraction_start
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen DINOv2 encoder unexpectedly became trainable")
    if config["encoder"]["weight_sha256"] != sha256_file(model_dir / "model.safetensors"):
        raise RuntimeError("frozen encoder checkpoint bytes changed during extraction")
    output.mkdir(parents=True, exist_ok=False)
    (output / "effective-config.json").write_text(
        json.dumps(
            {
                **config,
                "output_path": str(output),
                "run_mode": run_mode,
                "effective_readout": settings,
                "config_sha256": sha256_file(config_path),
                "source_commit": subprocess.run(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "source_tree_sha256": sha256_json(
                    {
                        path.relative_to(ROOT).as_posix(): sha256_file(path)
                        for path in (
                            ROOT / "scripts/train_miniwob_dino_path_a.py",
                            ROOT / "src/tiny_omni_decision/miniwob_dino_readout.py",
                            ROOT / "configs/pretrained_reuse/miniwob_dino_path_a_v1.json",
                        )
                    }
                ),
                "corpus_root": str(root),
                "corpus_manifest": corpus_manifest,
                "encoder_file_hashes": model_hashes,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "numpy": np.__version__,
                    "pillow": __import__("PIL").__version__,
                    "device": "cpu",
                    "threads": torch.get_num_threads(),
                },
                "started_at": started_at,
                "feature_extraction_seconds": extraction_seconds,
                "feature_shapes": {
                    "train": list(train_features.shape),
                    "validation": list(val_features.shape),
                },
                "train_sample_order_sha256": sha256_json([row["sample_id"] for row in train_rows]),
                "validation_sample_order_sha256": sha256_json(
                    [row["sample_id"] for row in val_rows]
                ),
                "unique_group_counts": {
                    "train": len({row["episode_group_id"] for row in train_rows}),
                    "validation": len({row["episode_group_id"] for row in val_rows}),
                },
                "cuda_used": False,
                "cloud_cost_usd": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "frozen-features.npz",
        train=train_features.numpy(),
        validation=val_features.numpy(),
        train_targets=train_targets.numpy(),
        validation_targets=val_targets.numpy(),
    )

    baseline, prior_predictions = position_prior(train_rows, val_rows)
    (output / "position-prior-metrics.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "position-prior-validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prior_predictions),
        encoding="utf-8",
    )

    blank_start = time.perf_counter()
    train_blank = black_screen_features(train_rows, model)
    val_blank = black_screen_features(val_rows, model)
    blank_seconds = time.perf_counter() - blank_start
    train_shuffled, train_donors = permute_screens(train_features, config["seed"] + 101)
    val_shuffled, val_donors = permute_screens(val_features, config["seed"] + 202)
    arms = {
        "matched_screen": (train_features, val_features),
        "zeroed_screen": (train_blank, val_blank),
        "permuted_screen": (train_shuffled, val_shuffled),
    }
    results: dict[str, Any] = {
        "position_prior": baseline,
        "arms": {},
        "controls": {
            "train_screen_donor_indices_sha256": sha256_json(train_donors),
            "validation_screen_donor_indices_sha256": sha256_json(val_donors),
            "all_train_screens_permuted_off_diagonal": len(set(train_donors)) == len(train_rows)
            and all(index != donor for index, donor in enumerate(train_donors)),
            "all_validation_screens_permuted_off_diagonal": len(set(val_donors)) == len(val_rows)
            and all(index != donor for index, donor in enumerate(val_donors)),
            "zeroed_feature_extraction_seconds": blank_seconds,
        },
    }
    for name, (train_arm, val_arm) in arms.items():
        results["arms"][name] = fit_arm(
            name,
            train_arm,
            train_targets,
            val_arm,
            val_targets,
            train_rows,
            val_rows,
            output / name,
            settings,
            config["seed"],
        )
    matched_path = output / "matched_screen" / "checkpoint-best.safetensors"
    matched_model = MiniWoBOptionScorer().cpu()
    matched_model.load_state_dict(load_file(matched_path, device="cpu"), strict=True)
    matched_model.eval()
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted_features, inverse = permute_candidate_axis(val_features, permutation)
    with torch.inference_mode():
        original_probabilities = torch.softmax(matched_model(val_features), dim=1)
        permuted_probabilities = torch.softmax(
            matched_model(permuted_features), dim=1
        ).index_select(1, inverse)
    option_order_delta = float((original_probabilities - permuted_probabilities).abs().max())
    if option_order_delta > 1e-7:
        raise RuntimeError("candidate-order permutation changed option-aligned probabilities")
    ordered_records, _ = predict_rows(matched_model, val_features, val_rows)
    results["controls"]["candidate_order_permutation"] = {
        "input_permutation": permutation.tolist(),
        "max_option_aligned_probability_abs_difference": option_order_delta,
        "accuracy_after_id_remap": float(
            classification_metrics(permuted_probabilities, val_targets)["accuracy"]
        ),
        "matched_accuracy": float(
            classification_metrics(original_probabilities, val_targets)["accuracy"]
        ),
    }
    results["summary"] = {
        "task_scope": "synthetic MiniWoB++ click-color screen-only development task",
        "validation_is_development_only": True,
        "not_multimodal_product_evidence": True,
        "selected_rule": settings["selector"],
        "candidate_order_probability_mapping_verified": option_order_delta <= 1e-7,
        "readout_parameters": sum(parameter.numel() for parameter in matched_model.parameters()),
        "resident_parameters": encoder_count
        + sum(parameter.numel() for parameter in matched_model.parameters()),
        "encoder_parameters_frozen": encoder_count,
        "encoder_parameters_updated": 0,
        "feature_extraction_seconds": extraction_seconds,
        "zeroed_screen_feature_seconds": blank_seconds,
        "cpu_only": True,
        "cloud_cost_usd": 0,
    }
    (output / "experiment-results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "PASS_EXECUTION", "output": str(output), **results["summary"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
