"""Fit a fixed text-free image Decision probe on cached CLEVR-4 train features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file
from tiny_omni_decision.textfree_image import TextFreeImageReadout, deterministic_asset_split

SEED = 17
EPOCHS = 128
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.1
ENCODER_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
ENCODER_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
SOURCE_REVISION = "cddc78fb2a8359dc958987b2c750bfdd4bfd2c73"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    count = len(targets)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(targets)
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    ece = 0.0
    for bin_index in range(15):
        mask = (confidence >= bin_index / 15) & (
            confidence < (bin_index + 1) / 15 if bin_index < 14 else confidence <= 1
        )
        if mask.any():
            ece += mask.float().mean().item() * abs(
                correct[mask].float().mean().item() - confidence[mask].mean().item()
            )
    return {
        "count": count,
        "accuracy": correct.float().mean().item(),
        "nll": F.cross_entropy(logits.float(), targets).item(),
        "brier": (probabilities - one_hot).square().sum(dim=-1).mean().item(),
        "ece_15_bins": ece,
        "mean_confidence": confidence.mean().item(),
    }


def _predict(model: TextFreeImageReadout, features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: model(features, name) for name in CLEVR4_TAXONOMIES}


def _evaluate(
    model: TextFreeImageReadout,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        logits = _predict(model, features)
    by_taxonomy = {name: _metrics(logits[name], targets[name]) for name in CLEVR4_TAXONOMIES}
    macro = {
        metric: sum(float(values[metric]) for values in by_taxonomy.values()) / len(by_taxonomy)
        for metric in ("accuracy", "nll", "brier", "ece_15_bins", "mean_confidence")
    }
    return {"macro": macro, "by_taxonomy": by_taxonomy, "logits": logits}


def _cpu_state(model: TextFreeImageReadout) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _write_predictions(
    path: Path,
    model: TextFreeImageReadout,
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    asset_hashes: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream, torch.inference_mode():
        for index, (row, asset_hash) in enumerate(zip(rows, asset_hashes, strict=True)):
            decisions = {}
            for taxonomy, classes in CLEVR4_TAXONOMIES.items():
                options = [
                    {"option_id": f"{taxonomy}:{class_id}", "class_id": class_id}
                    for class_id in classes
                ]
                decisions[taxonomy] = model.decide(features[index], taxonomy, options)
            stream.write(
                json.dumps(
                    {
                        "id": str(row["id"]),
                        "asset_sha256": asset_hash,
                        "target_class_ids": {
                            taxonomy: str(row[taxonomy]) for taxonomy in CLEVR4_TAXONOMIES
                        },
                        "decisions": decisions,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def run(
    feature_cache: Path,
    sample_dir: Path,
    output_dir: Path,
    requested_device: str,
    epochs: int = EPOCHS,
) -> dict[str, Any]:
    run_started_counter = time.perf_counter()
    feature_cache, sample_dir, output_dir = (
        feature_cache.resolve(),
        sample_dir.resolve(),
        output_dir.resolve(),
    )
    if not feature_cache.is_file() or not sample_dir.is_dir():
        raise FileNotFoundError("feature cache and sample directory must exist")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite image probe output: {output_dir}")

    train_path = sample_dir / "train-labels.jsonl"
    media_path = sample_dir / "train-image-manifest.jsonl"
    label_rows = _rows(train_path)
    media_rows = _rows(media_path)
    if not label_rows or len(label_rows) != len(media_rows):
        raise ValueError("train labels and image manifest must have the same nonzero row count")
    if any(row.get("split") != "train" for row in label_rows + media_rows):
        raise ValueError("only official train records may enter this probe")
    ids = [str(row["id"]) for row in label_rows]
    media_ids = [str(row["id"]) for row in media_rows]
    asset_hashes = [str(row["sha256"]) for row in media_rows]
    if len(set(ids)) != len(ids) or ids != media_ids or len(set(asset_hashes)) != len(asset_hashes):
        raise ValueError("train IDs or media asset identities are duplicated or misaligned")
    for row in label_rows:
        if any(
            str(row[taxonomy]) not in classes for taxonomy, classes in CLEVR4_TAXONOMIES.items()
        ):
            raise ValueError("unknown CLEVR-4 class ID in the training labels")
    train_labels_hash = sha256_file(train_path)
    media_manifest_hash = sha256_file(media_path)
    feature_cache_hash = sha256_file(feature_cache)

    with np.load(feature_cache, allow_pickle=False) as cache:
        cached_ids = cache["train_ids"].astype(str).tolist()
        cached_hashes = cache["train_media_sha256"].astype(str).tolist()
        feature_array = cache["train_features"]
    if cached_ids != ids or cached_hashes != asset_hashes:
        raise ValueError(
            "cached train features do not align with pinned train IDs and media hashes"
        )
    if feature_array.shape != (len(ids), 768) or feature_array.dtype != np.float32:
        raise ValueError("expected an [N,768] float32 VideoPrism train feature matrix")
    if not np.isfinite(feature_array).all():
        raise ValueError("non-finite cached image feature")

    train_indexes, validation_indexes = deterministic_asset_split(
        asset_hashes, SEED, VALIDATION_FRACTION
    )
    train_assets = {asset_hashes[index] for index in train_indexes}
    validation_assets = {asset_hashes[index] for index in validation_indexes}
    if train_assets & validation_assets:
        raise ValueError("internal train and development split shares media assets")
    if len(validation_indexes) < 2:
        raise ValueError("internal development split is too small")

    device_name = requested_device
    if requested_device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes < 256 * 1024 * 1024:
            raise RuntimeError(f"insufficient free CUDA memory before run: {free_bytes} bytes")
        preflight_cuda_memory = {"free_bytes": free_bytes, "total_bytes": total_bytes}
        torch.cuda.reset_peak_memory_stats(device)
    else:
        preflight_cuda_memory = None

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    model = TextFreeImageReadout(768, CLEVR4_TAXONOMIES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    feature_tensor = torch.from_numpy(feature_array.copy()).to(device)
    target_tensor = {
        taxonomy: torch.tensor(
            [CLEVR4_TAXONOMIES[taxonomy].index(str(row[taxonomy])) for row in label_rows],
            dtype=torch.long,
            device=device,
        )
        for taxonomy in CLEVR4_TAXONOMIES
    }
    train_index_tensor = torch.tensor(train_indexes, dtype=torch.long, device=device)
    validation_index_tensor = torch.tensor(validation_indexes, dtype=torch.long, device=device)
    x_train = feature_tensor[train_index_tensor]
    x_validation = feature_tensor[validation_index_tensor]
    y_train = {name: values[train_index_tensor] for name, values in target_tensor.items()}
    y_validation = {name: values[validation_index_tensor] for name, values in target_tensor.items()}
    for taxonomy, classes in CLEVR4_TAXONOMIES.items():
        for class_index, class_id in enumerate(classes):
            if not bool((y_train[taxonomy] == class_index).any()) or not bool(
                (y_validation[taxonomy] == class_index).any()
            ):
                raise ValueError(f"class {taxonomy}:{class_id} missing from a split")

    source_root = Path(__file__).resolve().parents[1]
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "UNKNOWN"
    readout_path = source_root / "src" / "tiny_omni_decision" / "textfree_image.py"
    source_hashes = {
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "readout_implementation_sha256": sha256_file(readout_path),
    }
    effective_config = {
        "seed": SEED,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "schedule": "constant",
        "validation_fraction": VALIDATION_FRACTION,
        "split": "SHA256(seed:media_sha256) ordering, first floor(0.1*N) held out",
        "checkpoint_selection": (
            f"minimum macro validation NLL over {epochs} epochs; no performance early stopping"
        ),
        "initialization": "new random linear heads from seed 17; no warm start",
        "feature_width": 768,
        "taxonomies": CLEVR4_TAXONOMIES,
        "feature_source": "frozen 8-frame VideoPrism-B feature for each still CLEVR-4 image",
        "input_contract": (
            "image feature only; no question, text, audio, video question, "
            "or text encoder"
        ),
        "source_commit": source_commit,
        "output_id": output_dir.name,
        "requested_device": requested_device,
        "source_hashes": source_hashes,
        "input_hashes": {
            "feature_cache_sha256": feature_cache_hash,
            "train_labels_sha256": train_labels_hash,
            "train_image_manifest_sha256": media_manifest_hash,
        },
    }
    config_bytes = json.dumps(effective_config, sort_keys=True, separators=(",", ":")).encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    (output_dir / "effective-config.json").write_text(
        json.dumps(effective_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    started_at = datetime.now(UTC).isoformat()
    preflight = {
        "seed": SEED,
        "train_image_count": len(train_indexes),
        "internal_validation_image_count": len(validation_indexes),
        "train_asset_count": len(train_assets),
        "internal_validation_asset_count": len(validation_assets),
        "asset_overlap": len(train_assets & validation_assets),
        "train_id_order_sha256": hashlib.sha256(
            "\n".join(ids[index] for index in train_indexes).encode()
        ).hexdigest(),
        "internal_validation_id_order_sha256": hashlib.sha256(
            "\n".join(ids[index] for index in validation_indexes).encode()
        ).hexdigest(),
        "feature_cache_sha256": feature_cache_hash,
        "train_labels_sha256": train_labels_hash,
        "train_image_manifest_sha256": media_manifest_hash,
        "class_counts": {
            taxonomy: {
                "train": {
                    class_id: sum(
                        str(label_rows[index][taxonomy]) == class_id for index in train_indexes
                    )
                    for class_id in classes
                },
                "internal_validation": {
                    class_id: sum(
                        str(label_rows[index][taxonomy]) == class_id for index in validation_indexes
                    )
                    for class_id in classes
                },
            }
            for taxonomy, classes in CLEVR4_TAXONOMIES.items()
        },
        "official_validation_arrays_loaded": False,
        "sealed_audit_loaded": False,
        "raw_media_reopened": False,
        "source_archive_full_sha512_verified": False,
        "cuda_preflight_memory": preflight_cuda_memory,
    }
    (output_dir / "sampler-preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, Any] | None = None
    rng = random.Random(SEED)
    training_seconds = 0.0
    evaluation_seconds = 0.0
    updates = 0
    model.train()
    best_path = output_dir / "best-readout.pt"
    history_path = output_dir / "training-history.jsonl"
    with history_path.open("w", encoding="utf-8", newline="\n") as history_stream:
        for epoch in range(1, epochs + 1):
            model.train()
            order = list(range(len(train_indexes)))
            rng.shuffle(order)
            losses = []
            train_started = time.perf_counter()
            for start in range(0, len(order), BATCH_SIZE):
                batch_indexes = torch.tensor(
                    order[start : start + BATCH_SIZE], dtype=torch.long, device=device
                )
                batch_features = x_train[batch_indexes]
                per_taxonomy = [
                    F.cross_entropy(
                        model(batch_features, taxonomy),
                        y_train[taxonomy][batch_indexes],
                    )
                    for taxonomy in CLEVR4_TAXONOMIES
                ]
                loss = torch.stack(per_taxonomy).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                updates += 1
                losses.append(float(loss.detach().cpu()))
            training_seconds += time.perf_counter() - train_started
            eval_started = time.perf_counter()
            train_result = _evaluate(model, x_train, y_train)
            validation_result = _evaluate(model, x_validation, y_validation)
            evaluation_seconds += time.perf_counter() - eval_started
            score = float(validation_result["macro"]["nll"])
            improved = score < best_score
            epoch_row = {
                "epoch": epoch,
                "optimizer_updates": updates,
                "train_id_order_sha256": hashlib.sha256(
                    "\n".join(ids[train_indexes[i]] for i in order).encode()
                ).hexdigest(),
                "train_cross_entropy": sum(losses) / len(losses),
                "train": {k: v for k, v in train_result.items() if k != "logits"},
                "validation": {k: v for k, v in validation_result.items() if k != "logits"},
                "best_so_far_epoch": epoch if improved else best_epoch,
                "best_so_far_macro_validation_nll": score if improved else best_score,
            }
            history.append(epoch_row)
            if improved:
                best_score = score
                best_epoch = epoch
                best_state = _cpu_state(model)
                best_validation = validation_result
                torch.save(best_state, best_path)
                (output_dir / "best-checkpoint.json").write_text(
                    json.dumps(
                        {
                            "epoch": best_epoch,
                            "optimizer_updates": updates,
                            "macro_validation_nll": best_score,
                            "validation_metrics": {
                                k: v for k, v in best_validation.items() if k != "logits"
                            },
                            "adapter_sha256": sha256_file(best_path),
                        },
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            history_stream.write(json.dumps(epoch_row, allow_nan=False) + "\n")
            history_stream.flush()
            if epoch % 8 == 0 or epoch == 1:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "updates": updates,
                            "train_macro_accuracy": train_result["macro"]["accuracy"],
                            "validation_macro_accuracy": validation_result["macro"]["accuracy"],
                            "validation_macro_nll": score,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    if best_state is None or best_validation is None:
        raise RuntimeError("fixed training budget did not produce a selected checkpoint")
    final_state = _cpu_state(model)
    final_path = output_dir / "final-readout.pt"
    torch.save(final_state, final_path)

    selected = TextFreeImageReadout(768, CLEVR4_TAXONOMIES).to(device).eval()
    selected.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    eval_started = time.perf_counter()
    selected_validation = _evaluate(selected, x_validation, y_validation)
    best_validation_logits = best_validation["logits"]
    reload_logits = selected_validation["logits"]
    if not all(
        torch.equal(best_validation_logits[name], reload_logits[name]) for name in CLEVR4_TAXONOMIES
    ):
        raise ValueError("selected checkpoint reload changed validation logits")
    selected_train = _evaluate(selected, x_train, y_train)
    final_model = TextFreeImageReadout(768, CLEVR4_TAXONOMIES).to(device).eval()
    final_model.load_state_dict(torch.load(final_path, map_location=device, weights_only=True))
    final_validation = _evaluate(final_model, x_validation, y_validation)
    evaluation_seconds += time.perf_counter() - eval_started
    internal_validation_rows = [label_rows[index] for index in validation_indexes]
    internal_validation_hashes = [asset_hashes[index] for index in validation_indexes]
    _write_predictions(
        output_dir / "selected-dev-predictions.jsonl",
        selected,
        internal_validation_rows,
        x_validation,
        internal_validation_hashes,
    )
    _write_predictions(
        output_dir / "final-dev-predictions.jsonl",
        final_model,
        internal_validation_rows,
        x_validation,
        internal_validation_hashes,
    )

    params = sum(parameter.numel() for parameter in selected.parameters())
    train_id_order_hash = hashlib.sha256(
        "\n".join(ids[i] for i in train_indexes).encode()
    ).hexdigest()
    validation_id_order_hash = hashlib.sha256(
        "\n".join(ids[i] for i in validation_indexes).encode()
    ).hexdigest()
    counts_by_taxonomy = {
        taxonomy: {
            "train": {
                class_id: sum(str(label_rows[i][taxonomy]) == class_id for i in train_indexes)
                for class_id in classes
            },
            "validation": {
                class_id: sum(str(label_rows[i][taxonomy]) == class_id for i in validation_indexes)
                for class_id in classes
            },
        }
        for taxonomy, classes in CLEVR4_TAXONOMIES.items()
    }
    history_path = output_dir / "training-history.jsonl"
    report = {
        "schema_version": 1,
        "status": "complete_textfree_image_development_probe",
        "task_claim": (
            "closed-set four-taxonomy CLEVR-4 image classification from frozen features; "
            "not natural-image reasoning or product quality"
        ),
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "seed": SEED,
        "effective_config": effective_config,
        "effective_config_sha256": config_hash,
        "source_revision": f"sgvaze/clevr4@{SOURCE_REVISION}",
        "encoder": {
            "name": "google/videoprism-base-f16r288",
            "revision": ENCODER_REVISION,
            "implementation_commit": ENCODER_COMMIT,
            "parameters": 114_365_184,
        },
        "input_contract": effective_config["input_contract"],
        "source_archive_full_sha512_verified": False,
        "train_images": len(train_indexes),
        "internal_validation_images": len(validation_indexes),
        "train_image_hashes_unique": len(train_assets),
        "validation_image_hashes_unique": len(validation_assets),
        "train_validation_asset_overlap": len(train_assets & validation_assets),
        "train_taxonomy_questions_per_epoch": len(train_indexes) * len(CLEVR4_TAXONOMIES),
        "validation_taxonomy_questions": len(validation_indexes) * len(CLEVR4_TAXONOMIES),
        "unique_train_images_consumed": len(train_indexes),
        "total_train_image_exposures": len(train_indexes) * epochs,
        "repeated_train_image_exposures": len(train_indexes) * (epochs - 1),
        "total_train_taxonomy_task_exposures": (
            len(train_indexes) * len(CLEVR4_TAXONOMIES) * epochs
        ),
        "optimizer_updates": updates,
        "selected_epoch": best_epoch,
        "selected_optimizer_updates": best_epoch * math.ceil(len(train_indexes) / BATCH_SIZE),
        "selected_train_image_exposures": len(train_indexes) * best_epoch,
        "selected_unique_train_images_consumed": len(train_indexes),
        "selected_checkpoint_rule": (
            f"minimum macro validation NLL over the fixed {epochs}-epoch run"
        ),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "total_wall_seconds": time.perf_counter() - run_started_counter,
        "completed_epochs": epochs,
        "best_validation_metrics": {k: v for k, v in best_validation.items() if k != "logits"},
        "selected_train_metrics": {k: v for k, v in selected_train.items() if k != "logits"},
        "final_validation_metrics": {k: v for k, v in final_validation.items() if k != "logits"},
        "validation_class_counts": counts_by_taxonomy,
        "selected_dev_id_order_sha256": validation_id_order_hash,
        "train_id_order_sha256": train_id_order_hash,
        "media_manifest_sha256": sha256_file(media_path),
        "train_labels_sha256": sha256_file(train_path),
        "feature_cache_sha256": sha256_file(feature_cache),
        "selected_readout_sha256": sha256_file(best_path),
        "final_readout_sha256": sha256_file(final_path),
        "selected_predictions_sha256": sha256_file(output_dir / "selected-dev-predictions.jsonl"),
        "final_predictions_sha256": sha256_file(output_dir / "final-dev-predictions.jsonl"),
        "history_sha256": sha256_file(history_path),
        "trainable_parameters": params,
        "frozen_encoder_plus_readout_parameters": 114_365_184 + params,
        "save_reload_exact_logits": True,
        "training_seconds_excluding_evaluation": training_seconds,
        "epoch_and_final_evaluation_seconds": evaluation_seconds,
        "requested_device": requested_device,
        "actual_device": str(device),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sealed_audit_loaded": False,
        "official_validation_loaded": False,
        "split_limit": (
            "The internal development split is carved from official CLEVR-4 train images. "
            "The earlier full-coverage image+text run used all official train examples; this "
            "readout is not a blind comparison and no prior model comparison is made here."
        ),
        "media_limit": (
            "Frozen image features were checked against the SHA-256 image manifest. The raw "
            "source ZIP full SHA-512 was not verified in the prior extraction receipt."
        ),
    }
    report_path = output_dir / "run-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "run-report.sha256").write_text(sha256_file(report_path) + "\n", encoding="ascii")
    return report


def _main() -> None:
    args = _args()
    print(
        json.dumps(
            run(
                args.feature_cache,
                args.sample_dir,
                args.output_dir,
                args.device,
                args.epochs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
