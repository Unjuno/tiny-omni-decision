"""Fit parameter-matched AV readouts on one fixed synthetic feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tiny_omni_decision.synthetic_av_readouts import ConditionalAVReadout, TemporalLSTMReadout
from tiny_omni_decision.synthetic_av_task import validate_manifest
from tiny_omni_decision.synthetic_av_training import (
    BestByNLL,
    classification_metrics,
    epoch_permutations,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/pretrained_reuse/synthetic_av_readouts_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_training_device(device: str, smoke: bool, environment: dict[str, str]) -> None:
    if device == "cuda" and not environment.get("TINY_OMNI_GPU_GATE_RESERVED_MIB"):
        raise RuntimeError("CUDA readout training requires the HotUGPU reservation")
    if smoke and device != "cpu":
        raise ValueError("readout smoke must run on CPU")
    if device == "cpu" and not smoke:
        raise ValueError("full-budget training must use the local CUDA path; CPU is smoke-only")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported training device: {device}")


def load_training_data(
    render_dir: Path, feature_dir: Path, allow_smoke: bool = False
) -> dict[str, Any]:
    manifest_path = Path(render_dir) / "render-manifest.json"
    report_path = Path(feature_dir) / "feature-report.json"
    cache_path = Path(feature_dir) / "examples.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest_path)
    if report.get("status") != "PASS_SYNTHETIC_AV_ALIGNED_FEATURE_CACHE":
        raise ValueError("feature cache receipt is not complete")
    if report.get("input_manifest_sha256") != manifest_hash:
        raise ValueError("feature cache was built from another rendered manifest")
    if sha256_file(cache_path) != report.get("output_sha256"):
        raise ValueError("aligned feature cache hash mismatch")
    with np.load(cache_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    selected_groups = set(report.get("selected_group_ids", []))
    expected_ids = [
        row["sample_id"]
        for row in manifest["examples"]
        if not selected_groups or row["group_id"] in selected_groups
    ]
    if arrays["example_ids"].astype(str).tolist() != expected_ids:
        raise ValueError("feature cache IDs/order differ from the frozen manifest")
    if report.get("smoke_subset") and not allow_smoke:
        raise ValueError("full-budget model search refuses a smoke-subset cache")
    if not np.isfinite(arrays["video_features"]).all() or not np.isfinite(
        arrays["audio_features"]
    ).all():
        raise ValueError("feature cache contains non-finite values")
    train_ids = [row["sample_id"] for row in manifest["examples"] if row["split"] == "train"]
    validation_ids = [
        row["sample_id"] for row in manifest["examples"] if row["split"] == "validation"
    ]
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation example identities overlap")
    train_groups = {
        row["group_id"] for row in manifest["examples"] if row["split"] == "train"
    }
    validation_groups = {
        row["group_id"] for row in manifest["examples"] if row["split"] == "validation"
    }
    if train_groups & validation_groups:
        raise ValueError("train and validation groups overlap")
    cache_train_ids = set(arrays["example_ids"][arrays["splits"] == "train"].astype(str).tolist())
    cache_validation_ids = set(
        arrays["example_ids"][arrays["splits"] == "validation"].astype(str).tolist()
    )
    if not cache_train_ids.issubset(train_ids) or not cache_validation_ids.issubset(
        validation_ids
    ):
        raise ValueError("feature cache contains IDs outside the frozen train/validation split")
    if not report.get("smoke_subset") and (
        cache_train_ids != set(train_ids) or cache_validation_ids != set(validation_ids)
    ):
        raise ValueError("full feature cache omits frozen train/validation examples")
    cached_manifest_rows = [
        row
        for row in manifest["examples"]
        if row["sample_id"] in cache_train_ids | cache_validation_ids
    ]
    cached_group_ids = {row["group_id"] for row in cached_manifest_rows}
    cached_groups = [
        group for group in manifest["groups"] if group["group_id"] in cached_group_ids
    ]
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
        "feature_report": report,
        "feature_cache_sha256": sha256_file(cache_path),
        "arrays": arrays,
        "train_indices": np.flatnonzero(arrays["splits"] == "train"),
        "validation_indices": np.flatnonzero(arrays["splits"] == "validation"),
        "train_ids": arrays["example_ids"][arrays["splits"] == "train"].astype(str).tolist(),
        "validation_ids": arrays["example_ids"][arrays["splits"] == "validation"]
        .astype(str)
        .tolist(),
        "train_group_ids": sorted(
            set(arrays["group_ids"][arrays["splits"] == "train"].astype(str).tolist())
        ),
        "validation_group_ids": sorted(
            set(arrays["group_ids"][arrays["splits"] == "validation"].astype(str).tolist())
        ),
        "cached_manifest_groups": cached_groups,
    }


def _construct_model(architecture: str, model_config: dict[str, Any]):
    if architecture == "temporal_lstm":
        return TemporalLSTMReadout(
            384,
            768,
            2,
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
        )
    if architecture == "conditional_av":
        return ConditionalAVReadout(
            384,
            768,
            2,
            hidden_size=int(model_config["hidden_size"]),
            num_heads=int(model_config["attention_heads"]),
            cross_depth=int(model_config["bidirectional_cross_attention_layers"]),
        )
    raise ValueError(f"unsupported architecture: {architecture}")


def _model_inputs(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    audio = arrays["audio_features"][arrays["audio_symbol_indices"]]
    video = arrays["video_features"][arrays["video_sequence_indices"]]
    query = F.one_hot(torch.from_numpy(arrays["query_polarity"].astype(np.int64)), 2).float()
    return {
        "audio": torch.as_tensor(audio, dtype=torch.float32, device=device),
        "video": torch.as_tensor(video, dtype=torch.float32, device=device),
        "query": query.to(device),
        "targets": torch.as_tensor(arrays["targets"], dtype=torch.long, device=device),
    }


def _atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _evaluate(model, inputs: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        logits = model(
            inputs["audio"][indices], inputs["video"][indices], inputs["query"][indices]
        )
        targets = inputs["targets"][indices]
        metrics = classification_metrics(logits, targets, ece_bins=15)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        predictions = logits.argmax(dim=-1).detach().cpu().numpy()
        target_values = targets.detach().cpu().numpy()
    return {
        "metrics": metrics,
        "logits": logits.detach().cpu(),
        "probabilities": probabilities,
        "predictions": predictions,
        "targets": target_values,
    }


def _fit_one(
    architecture: str,
    model_config: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    sample_permutations: list[list[int]],
    run_dir: Path,
    run_config: dict[str, Any],
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    seed = int(run_config["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = _construct_model(architecture, model_config).to(device)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    expected_count = int(model_config["expected_trainable_parameters"])
    if parameter_count != expected_count:
        raise ValueError(
            f"{architecture} has {parameter_count} trainable parameters; expected {expected_count}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_config["learning_rate"]),
        weight_decay=float(run_config["weight_decay"]),
    )
    batch_size = int(run_config["batch_size"])
    epochs = len(sample_permutations)
    validation_interval = int(run_config["validation_interval_epochs"])
    if smoke:
        validation_interval = 1
    selector = BestByNLL()
    model_dir = run_dir / architecture
    model_dir.mkdir(parents=True, exist_ok=False)
    history_path = model_dir / "history.jsonl"
    started = time.perf_counter()
    update = 0
    records = []
    training_indices = np.asarray(train_indices, dtype=np.int64)
    validation_tensor = torch.as_tensor(validation_indices, dtype=torch.long, device=device)
    for epoch_index, permutation in enumerate(sample_permutations):
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct_sum = torch.zeros((), device=device)
        seen = 0
        for start in range(0, len(permutation), batch_size):
            local_rows = np.asarray(permutation[start : start + batch_size], dtype=np.int64)
            indices_np = training_indices[local_rows]
            indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                inputs["audio"][indices], inputs["video"][indices], inputs["query"][indices]
            )
            targets = inputs["targets"][indices]
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite train loss at {architecture} step {update + 1}"
                )
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(
                    f"non-finite gradient at {architecture} step {update + 1}"
                )
            optimizer.step()
            update += 1
            batch_count = len(local_rows)
            loss_sum += loss.detach() * batch_count
            correct_sum += (logits.detach().argmax(dim=-1) == targets).sum()
            seen += batch_count

        epoch_record: dict[str, Any] = {
            "architecture": architecture,
            "epoch": epoch_index + 1,
            "optimizer_step": update,
            "rolling_train_scope": (
                "online epoch metrics: parameters change between minibatches; "
                "not a fixed-checkpoint generalization-gap estimate"
            ),
            "train_ce": float((loss_sum / seen).item()),
            "train_accuracy": float((correct_sum / seen).item()),
            "examples_consumed_cumulative": int((epoch_index + 1) * len(training_indices)),
            "unique_train_examples_consumed_cumulative": len(training_indices),
            "unique_train_groups": len(set(run_config["train_group_ids"])),
            "wall_seconds": time.perf_counter() - started,
        }
        if (epoch_index + 1) % validation_interval == 0 or epoch_index + 1 == epochs:
            evaluated = _evaluate(model, inputs, validation_tensor)
            val_metrics = evaluated["metrics"]
            epoch_record["validation"] = val_metrics
            epoch_record["validation_example_count"] = len(validation_indices)
            epoch_record["validation_group_count"] = len(set(run_config["validation_group_ids"]))
            if selector.update(val_metrics["nll"], update):
                best_reference_logits = evaluated["logits"].clone()
                best_path = model_dir / "best.pt"
                _atomic_torch_save(
                    best_path,
                    {
                        "architecture": architecture,
                        "step": update,
                        "epoch": epoch_index + 1,
                        "validation_nll": val_metrics["nll"],
                        "state_dict": {
                            name: tensor.detach().cpu()
                            for name, tensor in model.state_dict().items()
                        },
                    },
                )
                epoch_record["selected_best"] = True
            else:
                epoch_record["selected_best"] = False
            print(
                json.dumps(
                    {
                        "architecture": architecture,
                        "epoch": epoch_index + 1,
                        "optimizer_step": update,
                        "validation_nll": val_metrics["nll"],
                        "best_step": selector.step,
                    }
                ),
                flush=True,
            )
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(epoch_record, sort_keys=True) + "\n")
        records.append(epoch_record)

    terminal_path = model_dir / "terminal.pt"
    _atomic_torch_save(
        terminal_path,
        {
            "architecture": architecture,
            "step": update,
            "epoch": epochs,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
        },
    )
    best_payload = torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    verification_model = _construct_model(architecture, model_config).to(device)
    verification_model.load_state_dict(best_payload["state_dict"], strict=True)
    verification_model.eval()
    with torch.inference_mode():
        best_logits = verification_model(
            inputs["audio"][validation_tensor],
            inputs["video"][validation_tensor],
            inputs["query"][validation_tensor],
        )
    best_reload_max_abs_logit_error = float(
        (best_logits.detach().cpu() - best_reference_logits).abs().max().item()
    )
    if not torch.isfinite(best_logits).all():
        raise FloatingPointError("best-checkpoint reload produced non-finite logits")
    if best_reload_max_abs_logit_error > 1e-6:
        raise RuntimeError("best checkpoint reload does not reproduce selected validation logits")
    return {
        "architecture": architecture,
        "trainable_parameters": parameter_count,
        "optimizer_updates": update,
        "examples_consumed_total": update * batch_size,
        "best_step": selector.step,
        "examples_consumed_at_best_step": selector.step * batch_size,
        "unique_train_examples_at_best_step": len(training_indices),
        "best_validation_nll": selector.score,
        "best_checkpoint_sha256": sha256_file(model_dir / "best.pt"),
        "terminal_checkpoint_sha256": sha256_file(terminal_path),
        "best_reload_max_abs_logit_error": best_reload_max_abs_logit_error,
        "history_sha256": sha256_file(history_path),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "wall_seconds": time.perf_counter() - started,
        "history": records,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    require_training_device(args.device, args.smoke, os.environ)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; training was not started")
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = load_training_data(
        args.render_dir.resolve(strict=True),
        args.feature_dir.resolve(strict=True),
        allow_smoke=args.smoke,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    now = datetime.now(UTC).isoformat()
    run = {
        "status": "RUNNING",
        "started_utc": now,
        "experiment_id": config["experiment_id"],
        "source_commit": os.environ.get("CODEX_SOURCE_COMMIT", "UNKNOWN"),
        "effective_config": config,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": data["manifest_sha256"],
        "feature_cache_sha256": data["feature_cache_sha256"],
        "feature_report": data["feature_report"],
        "train_example_ids": data["train_ids"],
        "validation_example_ids": data["validation_ids"],
        "train_group_ids": data["train_group_ids"],
        "validation_group_ids": data["validation_group_ids"],
        "train_examples": len(data["train_ids"]),
        "validation_examples": len(data["validation_ids"]),
        "unique_sequences_train": len(
            {
                sequence
                for group in data["cached_manifest_groups"]
                if group["split"] == "train"
                for sequence in group["sequence_ids"].values()
            }
        ),
        "unique_sequences_validation": len(
            {
                sequence
                for group in data["cached_manifest_groups"]
                if group["split"] == "validation"
                for sequence in group["sequence_ids"].values()
            }
        ),
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "smoke": bool(args.smoke),
    }
    effective = dict(config["training"])
    if args.smoke:
        effective.update({"epochs": 2, "batch_size": 4, "validation_interval_epochs": 1})
    run["effective_training_overrides"] = effective
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    (output / "effective-config.json").write_text(json.dumps(config, indent=2) + "\n")

    device = torch.device(args.device)
    inputs = _model_inputs(data["arrays"], device)
    epochs = int(effective["epochs"])
    batch_size = int(effective["batch_size"])
    train_count = len(data["train_indices"])
    if train_count % batch_size:
        raise ValueError("training example count must divide evenly into configured batches")
    permutations = epoch_permutations(train_count, epochs, int(config["seed"]))
    permutation_payload = json.dumps(permutations, separators=(",", ":")).encode("utf-8")
    run["sampling_preflight"] = {
        "policy": "seeded torch.randperm per epoch; common precomputed sequence for both readouts",
        "epoch_permutations_sha256": hashlib.sha256(permutation_payload).hexdigest(),
        "examples_per_epoch": train_count,
        "updates_per_epoch": train_count // batch_size,
        "epochs": epochs,
        "optimizer_updates_per_readout": epochs * (train_count // batch_size),
        "examples_consumed_per_readout": epochs * train_count,
        "unique_train_examples": train_count,
        "max_sample_repeats_within_epoch": 1,
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    models_results = []
    for architecture, architecture_config in config["models"].items():
        models_results.append(
            _fit_one(
                architecture,
                architecture_config,
                inputs,
                data["train_indices"],
                data["validation_indices"],
                permutations,
                output,
                {
                    **effective,
                    "seed": int(config["seed"]),
                "train_group_ids": data["train_group_ids"],
                "validation_group_ids": data["validation_group_ids"],
                },
                device,
                args.smoke,
            )
        )
    run.update(
        {
            "status": "PASS_SYNTHETIC_AV_READOUT_TRAINING",
            "finished_utc": datetime.now(UTC).isoformat(),
            "results": models_results,
            "total_optimizer_updates": sum(item["optimizer_updates"] for item in models_results),
        }
    )
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        parsed = argparse.ArgumentParser(add_help=False)
        parsed.add_argument("--output", type=Path, required=True)
        try:
            output = parsed.parse_known_args()[0].output.resolve()
            run_path = output / "run.json"
            if run_path.is_file():
                run = json.loads(run_path.read_text(encoding="utf-8"))
                run.update(
                    {
                        "status": "FAILED",
                        "finished_utc": datetime.now(UTC).isoformat(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        except (SystemExit, OSError, json.JSONDecodeError):
            pass
        raise
