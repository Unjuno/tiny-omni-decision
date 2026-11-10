"""Compare global and coarse spatial VideoMAE features on a frozen train-only split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_clevr4_content_overlap import _load_jsonl, sha256_file
from scripts.extract_videomae_clevr4_frozen_features import (
    ARCHIVE_BYTES,
    CHECKPOINT_SHA256,
    SOURCE_SHA256,
    image_tensor,
    load_encoder,
)

SEED = 17
PER_CLASS = 1_000
TRAIN_PER_CLASS = 800
GRID_SIZE = 4
EXPECTED_TRAIN_MEDIA = "e48687af785a74864d2d4eb6e632a50d381d821a3ade8c2227eda7624647f410"
EXPECTED_TRAIN_LABELS = "45de9d36d557619789dd6160e808f7ff0c531f06a298f0b2af80c9720103b695"


def stable_split(
    media_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]
) -> tuple[list[dict], list[dict]]:
    """Select 1,000 examples/class and assign a deterministic 80/20 asset split."""
    if [r["id"] for r in media_rows] != [r["id"] for r in label_rows]:
        raise ValueError("media and label ID order differs")
    by_count: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_assets: set[str] = set()
    seen_hashes: set[str] = set()
    for media, label in zip(media_rows, label_rows, strict=True):
        sid, asset, digest = str(media["id"]), str(media["asset_id"]), str(media["media_sha256"])
        if sid in seen_ids or asset in seen_assets or digest in seen_hashes:
            raise ValueError("duplicate train ID, asset, or media identity")
        seen_ids.add(sid)
        seen_assets.add(asset)
        seen_hashes.add(digest)
        count = int(label["count"])
        if count not in range(1, 11):
            raise ValueError(f"invalid count target for {sid}")
        by_count[count].append((media, label))
    selected: list[tuple[str, dict, dict]] = []
    if set(by_count) != set(range(1, 11)):
        raise ValueError("all ten count labels are required")
    for count in range(1, 11):
        ranked = sorted(
            by_count[count],
            key=lambda pair: hashlib.sha256(f"{SEED}:select:{pair[0]['id']}".encode()).hexdigest(),
        )
        if len(ranked) < PER_CLASS:
            raise ValueError(f"count {count} has fewer than {PER_CLASS} examples")
        for rank, (media, label) in enumerate(ranked[:PER_CLASS]):
            split = "train" if rank < TRAIN_PER_CLASS else "holdout"
            selected.append((split, media, label))
    selected.sort(key=lambda item: (item[0], item[1]["id"]))
    train = [
        dict(**m, count=int(y["count"]), split="train")
        for split, m, y in selected
        if split == "train"
    ]
    holdout = [
        dict(**m, count=int(y["count"]), split="holdout")
        for split, m, y in selected
        if split == "holdout"
    ]
    if {r["asset_id"] for r in train} & {r["asset_id"] for r in holdout}:
        raise ValueError("asset group crosses train/holdout")
    return train, holdout


def prepare(inputs: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    for filename, expected in (
        ("train-media-manifest.jsonl", EXPECTED_TRAIN_MEDIA),
        ("train-labels.jsonl", EXPECTED_TRAIN_LABELS),
    ):
        if sha256_file(inputs / filename) != expected:
            raise ValueError(f"frozen input hash mismatch: {filename}")
    media = _load_jsonl(inputs / "train-media-manifest.jsonl")
    labels = _load_jsonl(inputs / "train-labels.jsonl")
    train, holdout = stable_split(media, labels)
    output.mkdir(parents=True)
    rows = train + holdout
    manifest = output / "media-manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    k: r[k]
                    for k in ("id", "asset_id", "archive_member", "media_sha256", "split", "count")
                },
                sort_keys=True,
            )
            + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    receipt = {
        "status": "FROZEN_TRAIN_ONLY_SPATIAL_PROBE",
        "seed": SEED,
        "source_train_media_sha256": EXPECTED_TRAIN_MEDIA,
        "source_train_labels_sha256": EXPECTED_TRAIN_LABELS,
        "media_manifest_sha256": sha256_file(manifest),
        "selected_total": len(rows),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "count_support_train": {str(c): sum(r["count"] == c for r in train) for c in range(1, 11)},
        "count_support_holdout": {
            str(c): sum(r["count"] == c for r in holdout) for c in range(1, 11)
        },
        "unique_train_assets": len({r["asset_id"] for r in train}),
        "unique_holdout_assets": len({r["asset_id"] for r in holdout}),
        "train_holdout_asset_overlap": 0,
        "selector_read": False,
        "reporting_pool_read": False,
        "official_validation_read": False,
        "sealed_audit_read": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _encode_spatial(
    model: torch.nn.Module, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = model.patch_embed(inputs)
    if model.pos_embed is not None:
        x = x + model.pos_embed.expand(x.shape[0], -1, -1).type_as(x).to(x.device).clone().detach()
    x = model.pos_drop(x)
    for block in model.blocks:
        x = block(x)
    x = model.norm(x)
    # VideoMAE uses 16 repeated frames -> 8 tubelets and 14x14 spatial patches.
    if x.shape[1:] != (8 * 14 * 14, 384):
        raise ValueError(f"unexpected VideoMAE token shape: {tuple(x.shape)}")
    normalized_tokens = model.fc_norm(x).reshape(x.shape[0], 8, 14, 14, 384)
    tokens = normalized_tokens.mean(dim=1)
    grid = F.adaptive_avg_pool2d(tokens.permute(0, 3, 1, 2), (GRID_SIZE, GRID_SIZE))
    global_features = model.fc_norm(x.mean(1))
    token_mean = normalized_tokens.mean(dim=(1, 2, 3))
    return global_features.float(), token_mean.float(), grid.flatten(1).float()


def extract(
    prepared: Path,
    archive: Path,
    output: Path,
    batch_size: int = 8,
    max_images: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Clevr-4 archive byte length changed")
    receipt = json.loads((prepared / "receipt.json").read_text(encoding="utf-8"))
    manifest = prepared / "media-manifest.jsonl"
    if sha256_file(manifest) != receipt["media_manifest_sha256"]:
        raise ValueError("prepared manifest hash mismatch")
    rows = _load_jsonl(manifest)
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")
    if len(rows) != 10_000:
        raise ValueError("expected exactly 10,000 fixed train/holdout images")
    if max_images is not None:
        rows = rows[:max_images]
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA is required for pinned extraction")
    output.mkdir(parents=True)
    model = load_encoder().eval().requires_grad_(False).cuda()
    global_values, token_mean_values, grid_values = [], [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with zipfile.ZipFile(archive) as source, torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            images = []
            for row in batch:
                raw = source.read(row["archive_member"])
                if hashlib.sha256(raw).hexdigest() != row["media_sha256"]:
                    raise ValueError(f"media hash mismatch: {row['id']}")
                images.append(image_tensor(raw))
            inputs = torch.stack(images).cuda()
            pooled, token_mean, grid = _encode_spatial(model, inputs)
            if (
                not torch.isfinite(pooled).all()
                or not torch.isfinite(token_mean).all()
                or not torch.isfinite(grid).all()
            ):
                raise ValueError("non-finite VideoMAE features")
            global_values.append(pooled.cpu().numpy())
            token_mean_values.append(token_mean.cpu().numpy())
            grid_values.append(grid.cpu().numpy())
            if (start + len(batch)) % 512 == 0:
                print(f"encoded {start + len(batch)}/{len(rows)}", flush=True)
    global_array = np.concatenate(global_values).astype(np.float32, copy=False)
    token_mean_array = np.concatenate(token_mean_values).astype(np.float32, copy=False)
    grid_array = np.concatenate(grid_values).astype(np.float32, copy=False)
    np.savez_compressed(
        output / "features.npz",
        ids=np.asarray([r["id"] for r in rows]),
        global_features=global_array,
        token_mean_features=token_mean_array,
        spatial_grid=grid_array,
        targets=np.asarray([r["count"] for r in rows], dtype=np.int64),
        splits=np.asarray([r["split"] for r in rows]),
    )
    result = {
        "status": "COMPLETE",
        "manifest_sha256": sha256_file(manifest),
        "feature_sha256": sha256_file(output / "features.npz"),
        "rows": len(rows),
        "smoke": max_images is not None,
        "global_shape": list(global_array.shape),
        "token_mean_shape": list(token_mean_array.shape),
        "spatial_shape": list(grid_array.shape),
        "source_sha256": SOURCE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "gpu": torch.cuda.get_device_name(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "seconds": time.perf_counter() - started,
        "selector_read": False,
        "reporting_pool_read": False,
        "official_validation_read": False,
        "sealed_audit_read": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del model
    return result


def _fit_linear(
    x: np.ndarray, y: np.ndarray, x_eval: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    train = torch.from_numpy(x.astype(np.float32, copy=False)).cuda()
    test = torch.from_numpy(x_eval.astype(np.float32, copy=False)).cuda()
    mean, std = train.mean(0), train.std(0).clamp_min(1e-6)
    train, test = (train - mean) / std, (test - mean) / std
    target = torch.as_tensor(y, dtype=torch.long, device="cuda")
    model = torch.nn.Linear(train.shape[1], 10).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    history = []
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train), target)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("non-finite classifier gradient")
        optimizer.step()
        if epoch in {1, 10, 25, 50, 100}:
            with torch.no_grad():
                history.append(
                    {
                        "epoch": epoch,
                        "train_ce": float(F.cross_entropy(model(train), target).item()),
                    }
                )
    model.eval()
    with torch.no_grad():
        logits = model(test).cpu().numpy()
        train_logits = model(train).cpu().numpy()
    fit = {
        "epochs": 100,
        "parameters": sum(p.numel() for p in model.parameters()),
        "state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "standardization_mean": mean.cpu(),
        "standardization_std": std.cpu(),
        "train_history": history,
    }
    return logits, train_logits, fit


def _metrics(target: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    shifted = logits - logits.max(1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(1, keepdims=True)
    y = target.astype(np.int64)
    p = probs[np.arange(len(y)), y]
    one_hot = np.eye(10)[y]
    confidence, prediction = probs.max(1), probs.argmax(1)
    ece = sum(
        np.mean(mask) * abs(float(np.mean(prediction[mask] == y[mask]) - np.mean(confidence[mask])))
        for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:], strict=True)
        for mask in [
            ((confidence >= low) & (confidence < high if high < 1 else confidence <= high))
        ]
        if mask.any()
    )
    return {
        "accuracy": float(np.mean(prediction == y)),
        "nll": float(-np.log(np.clip(p, 1e-12, 1)).mean()),
        "brier": float(np.square(probs - one_hot).sum(1).mean()),
        "ece_15": float(ece),
    }


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(1, keepdims=True)


def _paired_bootstrap(
    per_example: np.ndarray, *, replicates: int = 10_000, seed: int = SEED
) -> dict[str, Any]:
    if per_example.ndim != 1 or per_example.size == 0 or replicates <= 0:
        raise ValueError("paired bootstrap requires nonempty per-example deltas")
    generator = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for start in range(0, replicates, 250):
        size = min(250, replicates - start)
        indices = generator.integers(0, per_example.size, size=(size, per_example.size))
        samples.append(per_example[indices].mean(axis=1))
    quantiles = np.quantile(np.concatenate(samples), [0.025, 0.975])
    return {
        "estimate": float(per_example.mean()),
        "percentile_95_ci": [float(value) for value in quantiles],
        "bootstrap_replicates": replicates,
        "seed": seed,
        "unit": "one unique image/asset",
    }


def summarize_predictions(predictions_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    rows = _load_jsonl(predictions_path)
    ids = [str(row["id"]) for row in rows]
    targets = np.asarray([int(row["target"]) - 1 for row in rows], dtype=np.int64)
    if len(ids) != len(set(ids)) or not len(rows) or np.any((targets < 0) | (targets >= 10)):
        raise ValueError("prediction identities or targets are invalid")
    logits = {
        "global_384": np.asarray([row["global_logits"] for row in rows], dtype=np.float64),
        "token_mean_384": np.asarray([row["token_mean_logits"] for row in rows], dtype=np.float64),
        "spatial_grid_4x4": np.asarray([row["spatial_logits"] for row in rows], dtype=np.float64),
    }
    if any(
        value.shape != (len(rows), 10) or not np.isfinite(value).all() for value in logits.values()
    ):
        raise ValueError("prediction logits are malformed or non-finite")
    probabilities = {name: _probabilities(value) for name, value in logits.items()}
    per_example: dict[str, dict[str, np.ndarray]] = {}
    class_metrics = {}
    for name, probs in probabilities.items():
        predicted = probs.argmax(1)
        class_metrics[name] = {
            str(count + 1): float(np.mean(predicted[targets == count] == count))
            for count in range(10)
        }
        per_example[name] = {
            "correct": (predicted == targets).astype(np.float64),
            "nll": -np.log(np.clip(probs[np.arange(len(rows)), targets], 1e-12, 1)),
            "brier": np.square(probs - np.eye(10)[targets]).sum(1),
        }
    differences = {}
    comparisons = (
        ("spatial_grid_4x4", "global_384"),
        ("spatial_grid_4x4", "token_mean_384"),
        ("token_mean_384", "global_384"),
    )
    for candidate, baseline in comparisons:
        for metric in ("correct", "nll", "brier"):
            differences[f"{metric}_{candidate}_minus_{baseline}"] = _paired_bootstrap(
                per_example[candidate][metric] - per_example[baseline][metric]
            )
    result = {
        "status": "COMPLETE_FIXED_HOLDOUT_STATISTICS",
        "examples": len(rows),
        "unique_assets": len(ids),
        "prediction_sha256": sha256_file(predictions_path),
        "metrics": {name: _metrics(targets, value) for name, value in logits.items()},
        "per_count_accuracy": class_metrics,
        "paired_differences": differences,
        "holdout_reused_for_model_selection": False,
        "sealed_audit_read": False,
        "selector_read": False,
        "reporting_pool_read": False,
    }
    output.mkdir(parents=True)
    (output / "statistics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def analyze(prepared: Path, features_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((prepared / "receipt.json").read_text(encoding="utf-8"))
    feature_receipt = json.loads((features_dir / "result.json").read_text(encoding="utf-8"))
    feature_path = features_dir / "features.npz"
    if sha256_file(feature_path) != feature_receipt["feature_sha256"]:
        raise ValueError("feature file hash mismatch")
    with np.load(feature_path, allow_pickle=False) as cache:
        if len(cache["ids"]) != 10_000:
            raise ValueError("unexpected feature count")
        splits = cache["splits"].astype(str)
        targets = cache["targets"].astype(np.int64) - 1
        ids = cache["ids"].astype(str)
        global_features = cache["global_features"]
        token_mean_features = cache["token_mean_features"]
        spatial = cache["spatial_grid"]
    with np.load(feature_path, allow_pickle=False) as cache:
        if ids.tolist() != [str(r["id"]) for r in _load_jsonl(prepared / "media-manifest.jsonl")]:
            raise ValueError("feature IDs differ from frozen manifest")
        if global_features.shape != (10_000, 384) or spatial.shape != (
            10_000,
            GRID_SIZE * GRID_SIZE * 384,
        ):
            raise ValueError("unexpected feature matrix shape")
        if token_mean_features.shape != (10_000, 384):
            raise ValueError("unexpected token-mean feature shape")
    train_mask, holdout_mask = splits == "train", splits == "holdout"
    if train_mask.sum() != 8_000 or holdout_mask.sum() != 2_000:
        raise ValueError("fixed train/holdout split size changed")
    if len(set(ids[train_mask]) & set(ids[holdout_mask])):
        raise ValueError("sample identity overlap")
    results = {}
    for name, features in (
        ("global_384", global_features),
        ("token_mean_384", token_mean_features),
        ("spatial_grid_4x4", spatial),
    ):
        logits, train_logits, fit = _fit_linear(
            features[train_mask], targets[train_mask], features[holdout_mask], seed=SEED
        )
        results[name] = {
            "metrics": _metrics(targets[holdout_mask], logits),
            "train_metrics": _metrics(targets[train_mask], train_logits),
            "fit": fit,
            "logits": logits,
        }
    output.mkdir(parents=True)
    predictions_path = output / "holdout-predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": str(ids[i]),
                    "target": int(targets[i] + 1),
                    "global_logits": results["global_384"]["logits"][j].tolist(),
                    "token_mean_logits": results["token_mean_384"]["logits"][j].tolist(),
                    "spatial_logits": results["spatial_grid_4x4"]["logits"][j].tolist(),
                },
                sort_keys=True,
            )
            + "\n"
            for j, i in enumerate(np.flatnonzero(holdout_mask))
        ),
        encoding="utf-8",
    )
    for name, result in results.items():
        torch.save(
            {
                "state_dict": result["fit"].pop("state_dict"),
                "standardization_mean": result["fit"].pop("standardization_mean"),
                "standardization_std": result["fit"].pop("standardization_std"),
                "input_dimension": result["fit"]["parameters"] // 10 - 1,
                "seed": SEED,
            },
            output / f"{name}.pt",
        )
    for value in results.values():
        del value["logits"]
    report = {
        "status": "COMPLETE_TRAIN_ONLY_SPATIAL_FEATURE_DIAGNOSTIC",
        "seed": SEED,
        "train_examples": int(train_mask.sum()),
        "holdout_examples": int(holdout_mask.sum()),
        "unique_train_assets": receipt["unique_train_assets"],
        "unique_holdout_assets": receipt["unique_holdout_assets"],
        "conditions": results,
        "holdout_ids_sha256": hashlib.sha256(
            ("\n".join(ids[holdout_mask].tolist()) + "\n").encode()
        ).hexdigest(),
        "prediction_sha256": sha256_file(predictions_path),
        "encoder_parameters": 22_153_286,
        "feature_dimensions": {
            "global_384": 384,
            "token_mean_384": 384,
            "spatial_grid_4x4": GRID_SIZE * GRID_SIZE * 384,
        },
        "sealed_audit_read": False,
        "selector_read": False,
        "reporting_pool_read": False,
        "interpretation": (
            "single fixed train-only comparison; development evidence, "
            "not blind or product-wide quality"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "extract", "analyze", "summarize"):
        child = sub.add_parser(command)
        child.add_argument("--prepared", type=Path) if command != "prepare" else child.add_argument(
            "--inputs", type=Path, required=True
        )
        if command == "extract":
            child.add_argument("--archive", type=Path, required=True)
        if command == "analyze":
            child.add_argument("--features", type=Path, required=True)
        if command == "summarize":
            child.add_argument("--predictions", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if command == "extract":
            child.add_argument("--batch-size", type=int, default=8)
            child.add_argument("--max-images", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.inputs, args.output)
    elif args.command == "extract":
        result = extract(
            args.prepared,
            args.archive,
            args.output,
            args.batch_size,
            args.max_images,
        )
    elif args.command == "analyze":
        result = analyze(args.prepared, args.features, args.output)
    else:
        result = summarize_predictions(args.predictions, args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
