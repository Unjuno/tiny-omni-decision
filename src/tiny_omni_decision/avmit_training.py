"""Frozen-feature AVMIT SAVC training and evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .avmit_readout import AVMITFeatureDecisionScorer
from .robomnist_av_match_evaluation import classification_metrics
from .textfree_audio import fit_scalar_temperature

SEED = 17
GROUP_BATCH_SIZE = 64
MAX_EPOCHS = 20
PATIENCE = 4
BOOTSTRAP_REPLICATES = 2000
CONFIG = {
    "seed": SEED,
    "video_num_frames": "precomputed AVMIT sequence; source-defined variable length",
    "temporal_readouts": ["mean", "gru"],
    "hidden_width": 64,
    "candidate_scorer_width": 128,
    "group_batch_size": GROUP_BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "loss": "cross_entropy",
    "checkpoint_selector": "lowest calibration NLL",
    "temperature_fit": "one scalar on calibration logits of selected checkpoint",
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": SEED,
}


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def cache_fingerprint(cache_root: Path, pair_dir: Path) -> dict[str, Any]:
    """Revalidate only the three frozen non-sealed feature and pair files."""
    report_path = cache_root / "cache-report.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("sealed_payloads_opened") != 0 or report.get("raw_media_opened") != 0:
        raise ValueError("feature-cache report does not prove sealed/raw-media isolation")
    file_hashes: dict[str, dict[str, str]] = {}
    pair_hashes: dict[str, str] = {}
    asset_sets: dict[str, set[str]] = {}
    sequence_lengths: dict[str, dict[str, dict[str, float | int]]] = {}
    for split in ("train", "calibration", "validation"):
        pair_path = pair_dir / f"{split}.jsonl"
        pair_payload = pair_path.read_bytes()
        pair_hash = hashlib.sha256(pair_payload).hexdigest()
        if pair_hash != report["feature_cache"][split]["pair_manifest_sha256"]:
            raise ValueError(f"pair-manifest hash changed for {split}")
        pair_hashes[split] = pair_hash
        split_info = report["feature_cache"][split]
        root = cache_root / split
        file_hashes[split] = {}
        for name, expected in split_info["files"].items():
            path = root / name
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if path.stat().st_size != expected["bytes"] or actual != expected["sha256"]:
                raise ValueError(f"cached feature file changed: {split}/{name}")
            file_hashes[split][name] = actual
        index_rows = [
            json.loads(line)
            for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        asset_sets[split] = {row["asset_id_sha256"] for row in index_rows}
        if len(asset_sets[split]) != len(index_rows):
            raise ValueError(f"duplicate source identity in feature cache: {split}")
        sequence_lengths[split] = {}
        for modality in ("audio", "video"):
            lengths = np.asarray([row[f"{modality}_steps"] for row in index_rows], dtype=np.int64)
            sequence_lengths[split][modality] = {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": float(lengths.mean()),
            }
    if (
        asset_sets["train"] & asset_sets["calibration"]
        or asset_sets["train"] & asset_sets["validation"]
        or asset_sets["calibration"] & asset_sets["validation"]
    ):
        raise ValueError("cached media identity overlap across splits")
    config_hash = hashlib.sha256(
        json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "cache_root": str(cache_root.resolve()),
        "pair_manifest_dir": str(pair_dir.resolve()),
        "cache_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "cache_file_sha256": file_hashes,
        "pair_manifest_sha256": pair_hashes,
        "sequence_lengths": sequence_lengths,
        "config": CONFIG,
        "config_sha256": config_hash,
    }


@dataclass(frozen=True)
class PairGroup:
    group_id: str
    rows: tuple[dict[str, Any], ...]


class FeatureStore:
    """Read-only memory-mapped access to a verified split feature cache."""

    def __init__(self, split_dir: Path):
        self.split_dir = split_dir
        self.index = {
            row["asset_id_sha256"]: row
            for row in (
                json.loads(line)
                for line in (split_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        if len(self.index) == 0:
            raise ValueError(f"empty feature index: {split_dir}")
        self.audio = np.memmap(split_dir / "audio.f32", dtype="<f4", mode="r")
        self.video = np.memmap(split_dir / "video.f32", dtype="<f4", mode="r")

    def sequence(self, asset_id: str, modality: str) -> np.ndarray:
        if modality not in {"audio", "video"}:
            raise ValueError("modality must be audio or video")
        try:
            row = self.index[asset_id]
        except KeyError as error:
            raise ValueError(f"feature asset is absent from split cache: {asset_id}") from error
        offset = int(row[f"{modality}_offset_bytes"]) // 4
        steps = int(row[f"{modality}_steps"])
        width = int(row[f"{modality}_width"])
        values = self.audio if modality == "audio" else self.video
        return np.asarray(values[offset : offset + steps * width].reshape(steps, width))


def load_pair_groups(pair_manifest: Path) -> list[PairGroup]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in pair_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(row["pair_group_sha256"], []).append(row)
    groups = [PairGroup(group_id, tuple(rows)) for group_id, rows in sorted(grouped.items())]
    if not groups or any(len(group.rows) != 8 for group in groups):
        raise ValueError("each AVMIT pair group must contain exactly eight rows")
    seen_assets: set[str] = set()
    for group in groups:
        labels = [row["target"] for row in group.rows]
        if labels.count("same_action") != 4 or labels.count("different_action") != 4:
            raise ValueError("each AVMIT pair group must be balanced 4/4")
        if any(row["audio_asset_id_sha256"] == row["video_asset_id_sha256"] for row in group.rows):
            raise ValueError("same-clip audio/video pairs are forbidden")
        assets = {
            row[key]
            for row in group.rows
            for key in ("audio_asset_id_sha256", "video_asset_id_sha256")
        }
        if len(assets) != 4:
            raise ValueError("each AVMIT group must contain exactly four distinct source clips")
        if seen_assets & assets:
            raise ValueError("a source clip appears in more than one pair group in a split")
        seen_assets.update(assets)
    verify_unimodal_balance(groups)
    return groups


def verify_unimodal_balance(groups: list[PairGroup]) -> dict[str, int]:
    for group in groups:
        for key in ("audio_asset_id_sha256", "video_asset_id_sha256"):
            counts: dict[str, dict[str, int]] = {}
            for row in group.rows:
                target_counts = counts.setdefault(
                    row[key], {"same_action": 0, "different_action": 0}
                )
                target_counts[row["target"]] += 1
            if any(
                counts_for_asset != {"same_action": 1, "different_action": 1}
                for counts_for_asset in counts.values()
            ):
                raise ValueError("group does not prove the 50% source-unimodal ceiling")
    return {"groups": len(groups), "rows": len(groups) * 8, "unimodal_accuracy": 0.5}


def _collate(
    groups: list[PairGroup], store: FeatureStore, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    rows = [row for group in groups for row in group.rows]
    audio = [store.sequence(row["audio_asset_id_sha256"], "audio") for row in rows]
    video = [store.sequence(row["video_asset_id_sha256"], "video") for row in rows]
    audio_lengths = torch.tensor([len(value) for value in audio], dtype=torch.long)
    video_lengths = torch.tensor([len(value) for value in video], dtype=torch.long)
    audio_width = audio[0].shape[1]
    video_width = video[0].shape[1]
    audio_batch = np.zeros((len(rows), int(audio_lengths.max()), audio_width), dtype=np.float32)
    video_batch = np.zeros((len(rows), int(video_lengths.max()), video_width), dtype=np.float32)
    for index, values in enumerate(audio):
        audio_batch[index, : len(values)] = values
    for index, values in enumerate(video):
        video_batch[index, : len(values)] = values
    targets = torch.tensor(
        [0 if row["target"] == "same_action" else 1 for row in rows], dtype=torch.long
    )
    audio_tensor = torch.from_numpy(audio_batch).to(device)
    video_tensor = torch.from_numpy(video_batch).to(device)
    targets = targets.to(device)
    audio_lengths = audio_lengths.to(device)
    video_lengths = video_lengths.to(device)
    options, valid = AVMITFeatureDecisionScorer.option_features(len(rows), device=device)
    return audio_tensor, audio_lengths, video_tensor, video_lengths, options, valid, targets


def _predict(
    model: AVMITFeatureDecisionScorer,
    groups: list[PairGroup],
    store: FeatureStore,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(groups), GROUP_BATCH_SIZE):
            batch_groups = groups[start : start + GROUP_BATCH_SIZE]
            tensors = _collate(batch_groups, store, device)
            logits = model(*tensors[:-1]).float().cpu().numpy()
            targets = tensors[-1].cpu().numpy()
            all_logits.append(logits)
            all_targets.append(targets)
            for group, offset in zip(batch_groups, range(0, len(batch_groups) * 8, 8), strict=True):
                for index, row in enumerate(group.rows):
                    row_id = hashlib.sha256(
                        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                    prediction_rows.append(
                        {
                            "sample_id": row_id,
                            "pair_group_sha256": group.group_id,
                            "target": int(targets[offset + index]),
                            "logits": logits[offset + index].tolist(),
                        }
                    )
    return np.concatenate(all_logits), np.concatenate(all_targets), prediction_rows


def _metrics(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0):
    z = torch.as_tensor(logits, dtype=torch.float64) / temperature
    probabilities = torch.softmax(z, dim=-1).numpy()
    metrics = classification_metrics(probabilities, targets)
    predicted = probabilities.argmax(-1)
    metrics["confusion_matrix"] = [
        [int(np.sum((targets == target) & (predicted == prediction))) for prediction in range(2)]
        for target in range(2)
    ]
    return metrics, probabilities


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_predictions(path: Path, rows: list[dict[str, Any]], temperature: float = 1.0) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            logits = torch.tensor(row["logits"], dtype=torch.float64) / temperature
            probability = torch.softmax(logits, dim=-1).tolist()
            handle.write(
                json.dumps(
                    {
                        **row,
                        "prediction": int(np.argmax(probability)),
                        "probabilities": probability,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else None,
    }


def paired_group_bootstrap(
    mean_logits: np.ndarray,
    gru_logits: np.ndarray,
    targets: np.ndarray,
    groups: list[PairGroup],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    if len(mean_logits) != len(groups) * 8 or len(gru_logits) != len(mean_logits):
        raise ValueError("bootstrap requires eight aligned predictions per validation group")
    mean_prob = torch.softmax(torch.as_tensor(mean_logits, dtype=torch.float64), -1).numpy()
    gru_prob = torch.softmax(torch.as_tensor(gru_logits, dtype=torch.float64), -1).numpy()
    mean_nll = -np.log(np.clip(mean_prob[np.arange(len(targets)), targets], 1e-12, 1.0))
    gru_nll = -np.log(np.clip(gru_prob[np.arange(len(targets)), targets], 1e-12, 1.0))
    mean_acc = (mean_prob.argmax(-1) == targets).astype(np.float64)
    gru_acc = (gru_prob.argmax(-1) == targets).astype(np.float64)
    means = {
        "mean_accuracy": mean_acc.reshape(-1, 8).mean(1),
        "gru_accuracy": gru_acc.reshape(-1, 8).mean(1),
        "accuracy_delta_gru_minus_mean": (gru_acc - mean_acc).reshape(-1, 8).mean(1),
        "mean_nll": mean_nll.reshape(-1, 8).mean(1),
        "gru_nll": gru_nll.reshape(-1, 8).mean(1),
        "nll_delta_gru_minus_mean": (gru_nll - mean_nll).reshape(-1, 8).mean(1),
    }
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(groups), size=(replicates, len(groups)))
    output = {key: float(value.mean()) for key, value in means.items()}
    output["replicates"] = replicates
    output["seed"] = seed
    output["unit"] = "pair_group (all eight rows sampled together)"
    output["percentile_95_ci"] = {
        key: [
            float(np.quantile(value[indexes].mean(1), 0.025)),
            float(np.quantile(value[indexes].mean(1), 0.975)),
        ]
        for key, value in means.items()
    }
    return output


def run_smoke(cache_root: Path, pair_dir: Path, output_dir: Path, *, device_name: str = "cuda"):
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output_dir}")
    device = resolve_device(device_name)
    fingerprint = cache_fingerprint(cache_root, pair_dir)
    cache_report = json.loads((cache_root / "cache-report.json").read_text(encoding="utf-8"))
    if any(
        split_info["audio_width"] != 1024 or split_info["video_width"] != 1280
        for split_info in cache_report["feature_cache"].values()
    ):
        raise ValueError("cached AVMIT feature widths differ from the frozen scorer config")
    groups = load_pair_groups(pair_dir / "train.jsonl")[:8]
    balance = verify_unimodal_balance(groups)
    store = FeatureStore(cache_root / "train")
    results = {}
    for policy in ("mean", "gru"):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.reset_peak_memory_stats(device)
        model = AVMITFeatureDecisionScorer(
            audio_width=1024,
            video_width=1280,
            temporal_policy=policy,
            hidden_width=64,
            scorer_width=128,
        ).to(device)
        batch = _collate(groups, store, device)
        before_audio, before_video = batch[0].clone(), batch[2].clone()
        output = model(*batch[:-1])
        reverse = model(batch[0], batch[1], batch[2], batch[3], batch[4].flip(1), batch[5].flip(1))
        torch.testing.assert_close(reverse, output.flip(1), rtol=0, atol=1e-6)
        loss = torch.nn.functional.cross_entropy(output, batch[-1])
        loss.backward()
        if not torch.isfinite(loss) or not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ValueError(f"{policy} smoke produced an invalid gradient")
        torch.testing.assert_close(batch[0], before_audio)
        torch.testing.assert_close(batch[2], before_video)
        checkpoint = output_dir / f"smoke-{policy}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        restored = AVMITFeatureDecisionScorer(
            audio_width=1024,
            video_width=1280,
            temporal_policy=policy,
            hidden_width=64,
            scorer_width=128,
        ).to(device)
        restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        restored.eval()
        model.eval()
        with torch.inference_mode():
            torch.testing.assert_close(restored(*batch[:-1]), model(*batch[:-1]), rtol=0, atol=0)
        results[policy] = {
            "groups": len(groups),
            "rows": len(groups) * 8,
            "loss": float(loss.detach().cpu()),
            "trainable_parameters": model.parameter_count(),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        del model, restored, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "seed": SEED,
        "device": str(device),
        "environment": _environment(device),
        **fingerprint,
        "balance_control": balance,
        "smoke": results,
        "status": "pass",
    }
    _save_json(output_dir / "smoke-report.json", report)
    return report


def train_and_evaluate(
    cache_root: Path,
    pair_dir: Path,
    output_dir: Path,
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output_dir}")
    device = resolve_device(device_name)
    fingerprint = cache_fingerprint(cache_root, pair_dir)
    groups = {
        split: load_pair_groups(pair_dir / f"{split}.jsonl")
        for split in ("train", "calibration", "validation")
    }
    balance = verify_unimodal_balance(groups["validation"])
    stores = {split: FeatureStore(cache_root / split) for split in groups}
    output_dir.mkdir(parents=True, exist_ok=False)
    model_results = {}
    history: dict[str, list[dict[str, Any]]] = {}
    started = time.monotonic()

    # Training and calibration selection finish for both policies before the
    # development-validation split is opened for model prediction.
    for policy in ("mean", "gru"):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.reset_peak_memory_stats(device)
        model = AVMITFeatureDecisionScorer(
            audio_width=1024,
            video_width=1280,
            temporal_policy=policy,
            hidden_width=64,
            scorer_width=128,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        best_nll = float("inf")
        best_epoch = 0
        stale_epochs = 0
        consumed_groups = 0
        policy_dir = output_dir / policy
        policy_dir.mkdir()
        history[policy] = []
        final_state = None
        training_seconds = 0.0
        calibration_eval_seconds = 0.0

        for epoch in range(1, MAX_EPOCHS + 1):
            training_started = time.monotonic()
            model.train()
            permutation = torch.randperm(len(groups["train"]), generator=generator).tolist()
            total_loss = 0.0
            total_rows = 0
            for start in range(0, len(permutation), GROUP_BATCH_SIZE):
                selected = [
                    groups["train"][index]
                    for index in permutation[start : start + GROUP_BATCH_SIZE]
                ]
                batch = _collate(selected, stores["train"], device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(*batch[:-1])
                loss = torch.nn.functional.cross_entropy(logits, batch[-1])
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite {policy} training loss at epoch {epoch}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    raise ValueError(f"nonfinite {policy} gradient at epoch {epoch}")
                optimizer.step()
                count = int(batch[-1].numel())
                total_loss += float(loss.detach()) * count
                total_rows += count
                consumed_groups += len(selected)
            training_seconds += time.monotonic() - training_started

            evaluation_started = time.monotonic()
            cal_logits, cal_targets, cal_rows = _predict(
                model, groups["calibration"], stores["calibration"], device
            )
            calibration_eval_seconds += time.monotonic() - evaluation_started
            cal_metrics, _ = _metrics(cal_logits, cal_targets)
            epoch_record = {
                "epoch": epoch,
                "train_ce": total_loss / total_rows,
                "train_rows_this_epoch": total_rows,
                "cumulative_train_rows_consumed": consumed_groups * 8,
                "calibration_metrics": cal_metrics,
                "calibration_nll_selector": cal_metrics["nll"],
                "improved": cal_metrics["nll"] < best_nll,
            }
            history[policy].append(epoch_record)
            final_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            if cal_metrics["nll"] < best_nll:
                best_nll = cal_metrics["nll"]
                best_epoch = epoch
                stale_epochs = 0
                torch.save(final_state, policy_dir / "best.pt")
            else:
                stale_epochs += 1
            _save_json(policy_dir / "training-history.json", history[policy])
            if stale_epochs >= PATIENCE:
                break

        if final_state is None:
            raise AssertionError("training ended without a final model state")
        torch.save(final_state, policy_dir / "final.pt")
        best_state = torch.load(policy_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        model.eval()
        train_logits, train_targets, train_rows = _predict(
            model, groups["train"], stores["train"], device
        )
        cal_logits, cal_targets, cal_rows = _predict(
            model, groups["calibration"], stores["calibration"], device
        )
        temperature = fit_scalar_temperature(
            torch.as_tensor(cal_logits), torch.as_tensor(cal_targets, dtype=torch.long)
        )
        calibration_uncalibrated = _metrics(cal_logits, cal_targets)[0]
        model_results[policy] = {
            "model": model,
            "best_epoch": best_epoch,
            "epochs_completed": len(history[policy]),
            "train_rows_consumed": consumed_groups * 8,
            "unique_train_rows": len(groups["train"]) * 8,
            "repeated_train_rows": max(0, consumed_groups * 8 - len(groups["train"]) * 8),
            "unique_train_groups": len(groups["train"]),
            "unique_train_assets": len(stores["train"].index),
            "trainable_parameters": model.parameter_count(),
            "best_checkpoint_bytes": (policy_dir / "best.pt").stat().st_size,
            "final_checkpoint_bytes": (policy_dir / "final.pt").stat().st_size,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "temperature": temperature,
            "training_seconds": training_seconds,
            "calibration_eval_seconds": calibration_eval_seconds,
            "train": (_metrics(train_logits, train_targets)[0], train_rows),
            "calibration_uncalibrated": calibration_uncalibrated,
            "calibration": (_metrics(cal_logits, cal_targets, temperature)[0], cal_rows),
            "calibration_logits": cal_logits,
            "calibration_targets": cal_targets,
            "calibration_rows": cal_rows,
            "model_dir": policy_dir,
        }
        del optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # One paired validation pass per frozen candidate, after selection and
    # temperature fitting have completed for both candidates.
    for _policy, result in model_results.items():
        validation_started = time.monotonic()
        val_logits, val_targets, val_rows = _predict(
            result["model"], groups["validation"], stores["validation"], device
        )
        result["validation_eval_seconds"] = time.monotonic() - validation_started
        result["validation_uncalibrated"] = _metrics(val_logits, val_targets)[0]
        result["validation_calibrated"] = _metrics(val_logits, val_targets, result["temperature"])[
            0
        ]
        result["validation_logits"] = val_logits
        result["validation_targets"] = val_targets
        result["validation_rows"] = val_rows
        _save_predictions(
            result["model_dir"] / "calibration-predictions.jsonl",
            result["calibration_rows"],
            result["temperature"],
        )
        _save_predictions(
            result["model_dir"] / "validation-predictions.jsonl", val_rows, result["temperature"]
        )

    comparison = paired_group_bootstrap(
        model_results["mean"]["validation_logits"],
        model_results["gru"]["validation_logits"],
        model_results["mean"]["validation_targets"],
        groups["validation"],
    )
    summary = {
        "seed": SEED,
        "device": str(device),
        "environment": _environment(device),
        **fingerprint,
        "protocol": {
            "group_batch_size": GROUP_BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "adamw_lr": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "criterion": "cross_entropy",
            "selection": "lowest calibration NLL; stop after 4 non-improving epochs",
            "validation_passes_after_freeze": 1,
        },
        "source_counts": {
            split: {
                "groups": len(split_groups),
                "rows": len(split_groups) * 8,
                "unique_assets": len(stores[split].index),
            }
            for split, split_groups in groups.items()
        },
        "unimodal_control": balance,
        "chance_probability_control": {
            "accuracy": 0.5,
            "nll": float(np.log(2.0)),
            "brier": 0.5,
            "ece_15": 0.0,
            "mean_confidence": 0.5,
            "confusion_matrix": [
                [len(groups["validation"]) * 4, 0],
                [len(groups["validation"]) * 4, 0],
            ],
        },
        "models": {},
        "paired_validation_bootstrap": comparison,
        "wall_seconds": time.monotonic() - started,
    }
    for policy, result in model_results.items():
        summary["models"][policy] = {
            key: value
            for key, value in result.items()
            if key
            in {
                "best_epoch",
                "epochs_completed",
                "train_rows_consumed",
                "unique_train_rows",
                "repeated_train_rows",
                "unique_train_groups",
                "unique_train_assets",
                "trainable_parameters",
                "best_checkpoint_bytes",
                "final_checkpoint_bytes",
                "peak_vram_bytes",
                "temperature",
                "training_seconds",
                "calibration_eval_seconds",
                "validation_eval_seconds",
                "validation_uncalibrated",
                "validation_calibrated",
            }
        }
        summary["models"][policy]["train_metrics"] = result["train"][0]
        summary["models"][policy]["calibration_uncalibrated_metrics"] = result[
            "calibration_uncalibrated"
        ]
        summary["models"][policy]["calibration_metrics"] = result["calibration"][0]
        summary["models"][policy]["train_to_calibration_nll_gap"] = (
            result["calibration_uncalibrated"]["nll"] - result["train"][0]["nll"]
        )
    _save_json(output_dir / "results.json", summary)
    return summary
