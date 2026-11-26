"""Compare CE and ordinal loss on the fixed CLEVR-4 train-only image split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file
from tiny_omni_decision.ordinal import cumulative_ordinal_nll
from tiny_omni_decision.textfree_image import TextFreeImageReadout, deterministic_asset_split

SEED = 17
BOOTSTRAP_SEED = 20261029
BOOTSTRAP_REPLICATES = 10_000
DEFAULT_EPOCHS = 512
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.1


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = predictions.eq(targets)
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    ece = 0.0
    for index in range(15):
        mask = (confidence >= index / 15) & (
            confidence < (index + 1) / 15 if index < 14 else confidence <= 1
        )
        if mask.any():
            ece += mask.float().mean().item() * abs(
                correct[mask].float().mean().item() - confidence[mask].mean().item()
            )
    expected = (probabilities * torch.arange(1, logits.shape[-1] + 1)).sum(dim=-1)
    predicted_count = predictions + 1
    target_count = targets + 1
    return {
        "count": len(targets),
        "accuracy": correct.float().mean().item(),
        "nll": F.cross_entropy(logits.float(), targets).item(),
        "brier": (probabilities - one_hot).square().sum(dim=-1).mean().item(),
        "ece_15_bins": ece,
        "mean_confidence": confidence.mean().item(),
        "within_one_count_accuracy": ((predicted_count - target_count).abs() <= 1)
        .float()
        .mean()
        .item(),
        "mae_argmax": (predicted_count - target_count).abs().float().mean().item(),
        "mae_expected_count": (expected - target_count).abs().mean().item(),
        "class_counts": {
            str(index + 1): int(targets.eq(index).sum()) for index in range(logits.shape[-1])
        },
    }


def _readout(seed: int) -> torch.nn.Linear:
    torch.manual_seed(seed)
    model = TextFreeImageReadout(768, CLEVR4_TAXONOMIES)
    return model.classifiers["count"]


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _fit_arm(
    *,
    loss_name: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    validation_ids: list[str],
    validation_asset_hashes: list[str],
    epochs: int,
    arm_dir: Path,
) -> dict[str, Any]:
    rng = random.Random(SEED)
    head = _readout(SEED)
    initial_hash = _state_hash(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_nll = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_logits: torch.Tensor | None = None
    history: list[dict[str, Any]] = []
    updates = 0
    started = time.perf_counter()
    arm_dir.mkdir(parents=True, exist_ok=False)
    history_path = arm_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8", newline="\n") as stream:
        for epoch in range(1, epochs + 1):
            head.train()
            order = list(range(len(x_train)))
            rng.shuffle(order)
            losses: list[float] = []
            for offset in range(0, len(order), BATCH_SIZE):
                indexes = torch.tensor(order[offset : offset + BATCH_SIZE])
                logits = head(x_train[indexes])
                targets = y_train[indexes]
                if loss_name == "cross_entropy":
                    loss = F.cross_entropy(logits, targets)
                elif loss_name == "cumulative_ordinal_nll":
                    loss = cumulative_ordinal_nll(logits, targets)
                else:
                    raise ValueError(f"unknown loss: {loss_name}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                updates += 1
                losses.append(float(loss.detach()))

            head.eval()
            with torch.inference_mode():
                train_logits = head(x_train)
                validation_logits = head(x_validation)
            train_metrics = _metrics(train_logits, y_train)
            validation_metrics = _metrics(validation_logits, y_validation)
            score = float(validation_metrics["nll"])
            improved = score < best_nll
            row = {
                "epoch": epoch,
                "optimizer_updates": updates,
                "mean_training_objective": sum(losses) / len(losses),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "best_epoch_so_far": epoch if improved else best_epoch,
                "best_validation_nll_so_far": score if improved else best_nll,
            }
            history.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if epoch == 1 or epoch % 32 == 0:
                stream.flush()
            if improved:
                best_nll = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in head.state_dict().items()
                }
                best_logits = validation_logits.detach().cpu().clone()
                torch.save(best_state, arm_dir / "best-count-head.pt")
        stream.flush()

    if best_state is None or best_logits is None:
        raise RuntimeError("fixed budget did not produce a selected checkpoint")
    final_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    torch.save(final_state, arm_dir / "final-count-head.pt")
    selected = _readout(SEED)
    selected.load_state_dict(
        torch.load(arm_dir / "best-count-head.pt", map_location="cpu", weights_only=True)
    )
    selected.eval()
    with torch.inference_mode():
        reloaded_logits = selected(x_validation)
        selected_train_logits = selected(x_train)
    if not torch.equal(best_logits, reloaded_logits):
        raise ValueError("selected count-head reload changed validation logits")
    prediction_path = arm_dir / "dev-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, (logits, sample_id, asset_hash) in enumerate(
            zip(reloaded_logits, validation_ids, validation_asset_hashes, strict=True)
        ):
            probs = torch.softmax(logits.float(), dim=-1).tolist()
            prediction = int(torch.argmax(logits)) + 1
            stream.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "asset_sha256": asset_hash,
                        "target": int(y_validation[index]) + 1,
                        "prediction": prediction,
                        "probabilities": probs,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    return {
        "loss": loss_name,
        "initial_head_sha256": initial_hash,
        "selected_epoch": best_epoch,
        "optimizer_updates": updates,
        "selected_checkpoint_sha256": sha256_file(arm_dir / "best-count-head.pt"),
        "final_checkpoint_sha256": sha256_file(arm_dir / "final-count-head.pt"),
        "history_sha256": sha256_file(history_path),
        "prediction_sha256": sha256_file(prediction_path),
        "trainable_parameters": sum(value.numel() for value in head.parameters()),
        "selected_train_metrics": _metrics(selected_train_logits, y_train),
        "selected_validation_metrics": _metrics(reloaded_logits, y_validation),
        "selected_dev_logits": reloaded_logits,
        "training_wall_seconds": time.perf_counter() - started,
        "history": history,
    }


def _paired_bootstrap(
    ce_logits: torch.Tensor, ordinal_logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, Any]:
    ce_probs = torch.softmax(ce_logits.float(), dim=-1).numpy()
    ordinal_probs = torch.softmax(ordinal_logits.float(), dim=-1).numpy()
    target = targets.numpy()
    ce_pred = ce_probs.argmax(axis=-1)
    ordinal_pred = ordinal_probs.argmax(axis=-1)
    ce_nll = -np.log(np.clip(ce_probs[np.arange(len(target)), target], 1e-12, 1))
    ordinal_nll = -np.log(np.clip(ordinal_probs[np.arange(len(target)), target], 1e-12, 1))
    delta_accuracy = (ordinal_pred == target).astype(float) - (ce_pred == target).astype(float)
    delta_nll = ordinal_nll - ce_nll
    delta_mae = np.abs(ordinal_pred - target) - np.abs(ce_pred - target)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(target), size=(BOOTSTRAP_REPLICATES, len(target)))
    result: dict[str, Any] = {
        "unit": "image row; one count decision per unique image",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
    }
    for name, delta in (
        ("accuracy_ordinal_minus_ce", delta_accuracy),
        ("nll_ordinal_minus_ce", delta_nll),
        ("mae_ordinal_minus_ce", delta_mae),
    ):
        bootstrap = delta[samples].mean(axis=1)
        result[name] = {
            "point": float(delta.mean()),
            "percentile_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        }
    return result


def run(
    *, feature_cache: Path, corpus_dir: Path, baseline_fit_dir: Path, output_dir: Path, epochs: int
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    fit_report_path = baseline_fit_dir / "run-report.json"
    fit_report_hash_path = baseline_fit_dir / "run-report.sha256"
    baseline_path = baseline_fit_dir / "best-readout.pt"
    expected_fit_report_hash = fit_report_hash_path.read_text(encoding="utf-8").strip().split()[0]
    if sha256_file(fit_report_path) != expected_fit_report_hash:
        raise ValueError("baseline fit report hash mismatch")
    fit_report = json.loads(fit_report_path.read_text(encoding="utf-8"))
    if (
        fit_report.get("official_validation_loaded") is not False
        or fit_report.get("sealed_audit_loaded") is not False
    ):
        raise ValueError("baseline fit did not exclude official validation and sealed audit")
    if sha256_file(baseline_path) != fit_report.get("selected_readout_sha256"):
        raise ValueError("baseline selected checkpoint hash mismatch")
    feature_cache = feature_cache.resolve()
    corpus_dir = corpus_dir.resolve()
    output_dir = output_dir.resolve()
    if not feature_cache.is_file() or not corpus_dir.is_dir():
        raise FileNotFoundError("feature cache and train corpus directory must exist")
    if sha256_file(feature_cache) != fit_report["feature_cache_sha256"]:
        raise ValueError("feature cache differs from the pinned baseline fit")
    if shutil_free_bytes(output_dir.parent) < 64 * 1024 * 1024:
        raise RuntimeError("less than 64 MiB free for the explicitly bounded CPU probe output")

    labels_path = corpus_dir / "train-labels.jsonl"
    manifest_path = corpus_dir / "train-image-manifest.jsonl"
    labels = _jsonl(labels_path)
    manifest = _jsonl(manifest_path)
    ids = [str(row["id"]) for row in labels]
    manifest_ids = [str(row["id"]) for row in manifest]
    asset_hashes = [str(row["sha256"]) for row in manifest]
    if (
        not ids
        or ids != manifest_ids
        or any(row.get("split") != "train" for row in labels + manifest)
    ):
        raise ValueError("only aligned official train rows may enter the probe")
    if len(set(ids)) != len(ids) or len(set(asset_hashes)) != len(asset_hashes):
        raise ValueError("train IDs and asset hashes must be unique")
    if sha256_file(labels_path) != fit_report["train_labels_sha256"]:
        raise ValueError("training labels differ from the pinned baseline fit")
    if sha256_file(manifest_path) != fit_report["media_manifest_sha256"]:
        raise ValueError("training asset manifest differs from the pinned baseline fit")
    with np.load(feature_cache, allow_pickle=False) as cache:
        cached_ids = cache["train_ids"].astype(str).tolist()
        cached_hashes = cache["train_media_sha256"].astype(str).tolist()
        features = cache["train_features"]
    if cached_ids != ids or cached_hashes != asset_hashes:
        raise ValueError("cached train features do not exactly align with IDs and asset hashes")
    if (
        features.shape != (len(ids), 768)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("expected finite [N,768] float32 frozen train features")

    train_indexes, validation_indexes = deterministic_asset_split(
        asset_hashes, SEED, VALIDATION_FRACTION
    )
    train_id_hash = hashlib.sha256(
        "\n".join(ids[index] for index in train_indexes).encode()
    ).hexdigest()
    validation_id_hash = hashlib.sha256(
        "\n".join(ids[index] for index in validation_indexes).encode()
    ).hexdigest()
    if (
        train_id_hash != fit_report["train_id_order_sha256"]
        or validation_id_hash != fit_report["selected_dev_id_order_sha256"]
    ):
        raise ValueError("train-derived split order differs from the baseline fit")
    if set(asset_hashes[index] for index in train_indexes) & set(
        asset_hashes[index] for index in validation_indexes
    ):
        raise ValueError("image asset identity overlap across internal train/dev split")
    validation_ids = [ids[index] for index in validation_indexes]
    validation_asset_hashes = [asset_hashes[index] for index in validation_indexes]
    targets_all = torch.tensor(
        [CLEVR4_TAXONOMIES["count"].index(str(row["count"])) for row in labels],
        dtype=torch.long,
    )
    x = torch.from_numpy(features.copy())
    x_train, x_validation = x[train_indexes], x[validation_indexes]
    y_train, y_validation = targets_all[train_indexes], targets_all[validation_indexes]
    baseline = TextFreeImageReadout(768, CLEVR4_TAXONOMIES)
    baseline.load_state_dict(
        torch.load(baseline_path, map_location="cpu", weights_only=True), strict=True
    )
    baseline.eval()
    with torch.inference_mode():
        baseline_logits = baseline(x_validation, "count").cpu()

    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "UNKNOWN"
    source_root = Path(__file__).resolve().parents[1]
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    effective_config = {
        "seed": SEED,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "schedule": "constant",
        "input": "cached frozen VideoPrism-B 8-frame still-image features; image only",
        "train_split": "same SHA256(seed:asset_sha256), 90% training partition as pinned baseline",
        "internal_dev": (
            "same SHA256(seed:asset_sha256), 10% of official train images as pinned baseline"
        ),
        "changed_factor": "count-head training loss only: categorical CE vs cumulative ordinal NLL",
        "initialization_and_batch_order": (
            "same seed-17 count-head initialization and Python shuffle stream in both arms"
        ),
        "selector": "minimum categorical internal-dev NLL over fixed epochs for both arms",
        "trainable_parameters_per_arm": 7690,
        "device": "cpu",
        "cuda_used": False,
        "official_validation_arrays_loaded": False,
        "sealed_audit_loaded": False,
    }
    effective_config_hash = hashlib.sha256(
        json.dumps(effective_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output_dir / "effective-config.json").write_text(
        json.dumps(effective_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    preflight = {
        "train_images": len(ids),
        "unique_train_images_per_arm": len(train_indexes),
        "internal_validation_images": len(validation_indexes),
        "asset_overlap": 0,
        "train_id_order_sha256": train_id_hash,
        "validation_id_order_sha256": validation_id_hash,
        "train_labels_sha256": sha256_file(labels_path),
        "train_manifest_sha256": sha256_file(manifest_path),
        "feature_cache_sha256": sha256_file(feature_cache),
        "official_validation_arrays_loaded": False,
        "sealed_audit_loaded": False,
        "cuda_used": False,
        "available_output_bytes_before_run": shutil_free_bytes(output_dir.parent),
    }
    (output_dir / "sampler-preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Each arm resets to the exact same seed, head initialization and shuffled ID order.
    ce = _fit_arm(
        loss_name="cross_entropy",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        validation_ids=validation_ids,
        validation_asset_hashes=validation_asset_hashes,
        epochs=epochs,
        arm_dir=output_dir / "cross_entropy",
    )
    ordinal = _fit_arm(
        loss_name="cumulative_ordinal_nll",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        validation_ids=validation_ids,
        validation_asset_hashes=validation_asset_hashes,
        epochs=epochs,
        arm_dir=output_dir / "cumulative_ordinal_nll",
    )
    if ce["initial_head_sha256"] != ordinal["initial_head_sha256"]:
        raise ValueError("controlled loss arms did not start from identical head parameters")

    report = {
        "schema_version": 1,
        "status": "complete_train_only_count_loss_comparison",
        "task": "CLEVR-4 image-only object-count classification, 10 ordered classes",
        "source_commit": source_commit,
        "command": [sys.executable, *sys.argv],
        "pythonpath_environment": os.environ.get("PYTHONPATH"),
        "effective_config": effective_config,
        "effective_config_sha256": effective_config_hash,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "ordinal_loss_source_sha256": sha256_file(
            source_root / "src/tiny_omni_decision/ordinal.py"
        ),
        "readout_source_sha256": sha256_file(
            source_root / "src/tiny_omni_decision/textfree_image.py"
        ),
        "baseline_source_fit_report_sha256": sha256_file(fit_report_path),
        "baseline_source_checkpoint_sha256": sha256_file(baseline_path),
        "feature_cache_sha256": sha256_file(feature_cache),
        "train_labels_sha256": sha256_file(labels_path),
        "train_manifest_sha256": sha256_file(manifest_path),
        "seed": SEED,
        "counts": {
            "total_official_train_images": len(ids),
            "unique_train_images_per_arm": len(train_indexes),
            "internal_validation_images": len(validation_indexes),
            "asset_overlap": 0,
            "validation_id_order_sha256": validation_id_hash,
            "train_id_order_sha256": train_id_hash,
            "epochs_per_arm": epochs,
            "optimizer_updates_per_arm": math.ceil(len(train_indexes) / BATCH_SIZE) * epochs,
            "training_example_exposures_per_arm": len(train_indexes) * epochs,
        },
        "baseline_joint_four_head_count_metrics_on_same_internal_dev": _metrics(
            baseline_logits, y_validation
        ),
        "baseline_model_scope": {
            "trainable_parameters": fit_report["trainable_parameters"],
            "full_image_path_parameters": fit_report["frozen_encoder_plus_readout_parameters"],
            "frozen_encoder_revision": fit_report["encoder"]["revision"],
        },
        "arms": {
            "cross_entropy": {
                key: value
                for key, value in ce.items()
                if key != "history" and key != "selected_dev_logits"
            },
            "cumulative_ordinal_nll": {
                key: value
                for key, value in ordinal.items()
                if key != "history" and key != "selected_dev_logits"
            },
        },
        "paired_internal_dev_bootstrap": _paired_bootstrap(
            ce["selected_dev_logits"], ordinal["selected_dev_logits"], y_validation
        ),
        "packages": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            (
                "This is only a synthetic image-count task; it makes no natural-image or "
                "multimodal claim."
            ),
            (
                "The internal development split is from official train data; it is not a "
                "blind benchmark."
            ),
            "The official validation arrays and sealed audit were deliberately not loaded.",
            (
                "The paired bootstrap describes uncertainty on the same development sample "
                "used for checkpoint selection."
            ),
            (
                "This controlled comparison isolates loss within count-only fitting; it is not "
                "a direct causal comparison to the earlier four-head joint optimizer trajectory."
            ),
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_hash = sha256_file(report_path)
    (output_dir / "report.sha256").write_text(f"{report_hash}  report.json\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return report


def shutil_free_bytes(path: Path) -> int:
    import shutil

    return shutil.disk_usage(path).free


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--baseline-fit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    run(
        feature_cache=args.feature_cache,
        corpus_dir=args.corpus_dir,
        baseline_fit_dir=args.baseline_fit_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
