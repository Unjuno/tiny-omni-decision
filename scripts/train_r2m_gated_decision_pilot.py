"""Fit and evaluate the bounded R2-M frozen-feature matching pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tiny_omni_decision.r2m_activity import (
    standardize_sensor_feature_cache_from_train,
)
from tiny_omni_decision.time_aware_gated_fusion import (
    TemporalFeatureStream,
    TimeAwareGatedFusion,
)

SEED = 17
QUERY_CODES = torch.tensor([[1.0, 0.0], [0.0, 1.0]])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout
    source_files = (
        Path(__file__),
        repository / "src/tiny_omni_decision/time_aware_gated_fusion.py",
        repository / "src/tiny_omni_decision/r2m_activity.py",
        repository / "src/tiny_omni_decision/r2m_pairs.py",
        repository / "scripts/extract_r2m_frozen_features.py",
        repository / "scripts/build_r2m_audio_vibration_pair_index.py",
    )
    return {
        "base_commit": commit,
        "working_tree_clean": not bool(status.strip()),
        "working_tree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_file_sha256": {
            str(path.relative_to(repository)): sha256_file(path)
            for path in source_files
        },
    }


def load_features(cache_dir: Path) -> dict[str, dict[str, object]]:
    import h5py  # noqa: F401 - confirms the recorded feature environment

    features: dict[str, dict[str, object]] = {}
    for sidecar_path in sorted(cache_dir.glob("features/*.json")):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        npz_path = sidecar_path.with_suffix(".npz")
        if sha256_file(npz_path) != sidecar.get("feature_sha256"):
            raise ValueError(f"feature cache hash mismatch: {npz_path.name}")
        with np.load(npz_path, allow_pickle=False) as data:
            ids = data["sample_ids"].astype(str)
            audio = data["audio_values"]
            offsets = data["audio_offsets"]
            sensors = data["sensor_values"]
            durations = data["durations_seconds"]
            splits = data["splits"].astype(str)
            activities = data["activities"].astype(str)
            participants = data["participant_ids"]
            trials = data["trial_ids"]
            for index, identifier in enumerate(ids):
                if identifier in features:
                    raise ValueError(f"duplicate frozen feature ID: {identifier}")
                features[identifier] = {
                    "audio": audio[offsets[index]:offsets[index + 1]].copy(),
                    "sensor": sensors[index].copy(),
                    "duration": float(durations[index]),
                    "split": str(splits[index]),
                    "activity": str(activities[index]),
                    "participant_id": int(participants[index]),
                    "trial_id": int(trials[index]),
                }
    return features


class PairExamples(Dataset):
    def __init__(self, pairs: list[dict[str, object]],
                 features: dict[str, dict[str, object]]):
        self.rows: list[dict[str, object]] = []
        self.features = features
        self.positive_to_negative: dict[str, str] = {}
        for pair in pairs:
            anchor = str(pair["anchor_sample_id"])
            positive = str(pair["positive_sensor_sample_id"])
            negative = str(pair["negative_sensor_sample_id"])
            self.positive_to_negative[anchor] = negative
            for sensor_id, label in ((positive, 1), (negative, 0)):
                if anchor not in features or sensor_id not in features:
                    raise ValueError("pair index references a missing cached feature")
                if (features[anchor]["split"] != pair["split"]
                        or features[sensor_id]["split"] != pair["split"]):
                    raise ValueError("pair feature crosses a split boundary")
                self.rows.append({
                    "pair_id": str(pair["pair_id"]),
                    "anchor_id": anchor,
                    "sensor_id": sensor_id,
                    "label": label,
                    "split": str(pair["split"]),
                    "activity": str(pair["activity"]),
                    "negative_relation": str(pair["negative_participant_relation"]),
                })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        anchor = self.features[str(row["anchor_id"])]
        sensor = self.features[str(row["sensor_id"])]
        return {**row, "audio": anchor["audio"], "duration": anchor["duration"],
                "sensor": sensor["sensor"]}


def collate(examples: list[dict[str, object]], *, mode: str = "normal") -> dict[str, object]:
    batch = len(examples)
    audio_lengths = [np.asarray(row["audio"]).shape[0] for row in examples]
    max_audio = max(audio_lengths)
    audio_values = torch.zeros(batch, max_audio, 384, dtype=torch.float32)
    audio_valid = torch.zeros(batch, max_audio, dtype=torch.bool)
    audio_times = torch.zeros(batch, max_audio, dtype=torch.float32)
    sensor_values = torch.zeros(batch, 128, 3, dtype=torch.float32)
    sensor_valid = torch.ones(batch, 128, dtype=torch.bool)
    sensor_times = torch.zeros(batch, 128, dtype=torch.float32)
    for index, row in enumerate(examples):
        audio = torch.from_numpy(np.asarray(row["audio"], dtype=np.float32))
        sensor = torch.from_numpy(np.asarray(row["sensor"], dtype=np.float32))
        duration = float(row["duration"])
        audio_values[index, :len(audio)] = audio
        audio_valid[index, :len(audio)] = True
        audio_times[index, :len(audio)] = torch.linspace(0, duration * 1000, len(audio))
        sensor_values[index] = sensor
        sensor_times[index] = torch.linspace(0, duration * 1000, 128)
    audio_present = torch.ones(batch, dtype=torch.bool)
    sensor_present = torch.ones(batch, dtype=torch.bool)
    if mode == "without_audio":
        audio_present.zero_()
        audio_valid.zero_()
    elif mode == "without_sensor":
        sensor_present.zero_()
        sensor_valid.zero_()
    elif mode == "sensor_plus_250ms":
        sensor_times += 250.0
    streams = {
        "audio": TemporalFeatureStream(audio_values, audio_times, audio_valid,
                                        audio_present),
        "sensor": TemporalFeatureStream(sensor_values, sensor_times, sensor_valid,
                                         sensor_present),
    }
    queries = QUERY_CODES.unsqueeze(0).expand(batch, -1, -1).clone()
    labels = torch.tensor([int(row["label"]) for row in examples], dtype=torch.long)
    if mode == "force_mismatch":
        labels.fill_(0)
    return {"streams": streams, "queries": queries,
            "option_valid": torch.ones(batch, 2, dtype=torch.bool),
            "labels": labels, "rows": examples}


def score_batch(model, scorer, batch, device, mode="normal"):
    packed = collate(batch, mode=mode)
    streams = {
        key: TemporalFeatureStream(value.values.to(device), value.timestamps_ms.to(device),
                                   value.valid.to(device), value.present.to(device))
        for key, value in packed["streams"].items()
    }
    queries = packed["queries"].to(device)
    option_valid = packed["option_valid"].to(device)
    labels = packed["labels"].to(device)
    states = model(streams, queries, option_valid)
    logits = scorer(states).squeeze(-1)
    return logits, labels


def metric_summary(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    chosen = probabilities[np.arange(len(labels)), labels]
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (prediction == labels).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for bin_index, lower in enumerate(edges[:-1]):
        upper = edges[bin_index + 1]
        upper_mask = confidence < upper if bin_index < 14 else confidence <= upper
        mask = (confidence >= lower) & upper_mask
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    one_hot = np.eye(2, dtype=np.float64)[labels]
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.maximum(chosen, 1e-12)).mean()),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_15bin": float(ece),
        "mean_confidence": float(confidence.mean()),
        "examples": int(len(labels)),
    }


def paired_choice_summary(
    rows: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    grouped: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["pair_id"])][int(row["target"])] = row
    logits_rows: list[list[float]] = []
    prediction_rows: list[dict[str, object]] = []
    for pair_id, candidates in sorted(grouped.items()):
        if set(candidates) != {0, 1}:
            continue
        scores = []
        for target in (0, 1):
            probability = float(candidates[target]["probabilities"][1])
            probability = min(max(probability, 1e-12), 1.0 - 1e-12)
            scores.append(math.log(probability / (1.0 - probability)))
        logits_rows.append(scores)
        pair_probability = torch.tensor(scores, dtype=torch.float64).softmax(0).numpy()
        prediction_rows.append({
            "pair_id": pair_id,
            "negative_relation": candidates[1].get("negative_relation"),
            "target_option": 1,
            "prediction": int(pair_probability.argmax()),
            "probabilities": pair_probability.tolist(),
        })
    if not logits_rows:
        return {"examples": 0, "accuracy": None, "nll": None,
                "brier": None, "ece_15bin": None}, prediction_rows
    logits_array = np.asarray(logits_rows, dtype=np.float64)
    labels_array = np.ones(len(logits_rows), dtype=np.int64)
    metrics = metric_summary(logits_array, labels_array)
    relations = sorted({str(row["negative_relation"]) for row in prediction_rows
                        if row.get("negative_relation") is not None})
    metrics["by_negative_relation"] = {
        relation: metric_summary(
            logits_array[[index for index, row in enumerate(prediction_rows)
                          if row.get("negative_relation") == relation]],
            labels_array[[index for index, row in enumerate(prediction_rows)
                          if row.get("negative_relation") == relation]],
        )
        for relation in relations
    }
    return metrics, prediction_rows


def temporal_correlation_baseline(
    examples: PairExamples,
    *,
    output_path: Path | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Use fixed zero-lag correlation of audio/sensor energy traces as a baseline."""
    logits_rows: list[list[float]] = []
    labels: list[int] = []
    output_rows: list[dict[str, object]] = []
    paired: dict[str, dict[int, float]] = defaultdict(dict)

    def standardized(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        deviation = values.std()
        if deviation < 1e-12:
            return np.zeros_like(values)
        return (values - values.mean()) / deviation

    for index in range(len(examples)):
        row = examples[index]
        audio = np.asarray(row["audio"], dtype=np.float64)
        sensor = np.asarray(row["sensor"], dtype=np.float64)
        audio_energy = np.linalg.norm(audio, axis=1)
        sensor_energy = np.linalg.norm(sensor, axis=1)
        audio_grid = np.interp(
            np.linspace(0.0, 1.0, len(sensor_energy)),
            np.linspace(0.0, 1.0, len(audio_energy)), audio_energy,
        )
        audio_z = standardized(audio_grid)
        sensor_z = standardized(sensor_energy)
        score = float(np.mean(audio_z * sensor_z))
        label = int(row["label"])
        logits_rows.append([0.0, score])
        labels.append(label)
        probability = torch.tensor([0.0, score], dtype=torch.float64).softmax(0).numpy()
        prediction = int(probability.argmax())
        output_rows.append({
            "pair_id": row["pair_id"],
            "anchor_sample_id": row["anchor_id"],
            "sensor_sample_id": row["sensor_id"],
            "target": label,
            "negative_relation": row["negative_relation"],
            "prediction": prediction,
            "score": score,
            "probabilities": probability.tolist(),
        })
        paired[str(row["pair_id"])][label] = score

    logits_array = np.asarray(logits_rows, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    metrics = metric_summary(logits_array, labels_array)
    paired_correct = [score_map[1] > score_map[0]
                      for score_map in paired.values()
                      if set(score_map) == {0, 1}]
    metrics["paired_candidate_accuracy"] = (
        float(np.mean(paired_correct)) if paired_correct else None
    )
    metrics["paired_questions"] = len(paired_correct)
    paired_metrics, paired_rows = paired_choice_summary(output_rows)
    metrics["paired_choice_metrics"] = paired_metrics
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        paired_path = output_path.with_name(
            "validation-temporal-correlation-paired-choices.jsonl"
        )
        with paired_path.open("w", encoding="utf-8") as stream:
            for row in paired_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    return metrics, output_rows


def evaluate(model, scorer, dataset: PairExamples, device, *, batch_size: int,
             mode: str = "normal") -> tuple[dict[str, object], list[dict[str, object]]]:
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    rows_all: list[dict[str, object]] = []
    model.eval()
    scorer.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[index] for index in range(start, min(start + batch_size,
                                                                  len(dataset)))]
            logits, labels = score_batch(model, scorer, rows, device, mode)
            logits_all.append(logits.float().cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            probabilities = logits.softmax(dim=-1).float().cpu().numpy()
            for row, probability in zip(rows, probabilities, strict=True):
                predicted = int(probability.argmax())
                rows_all.append({
                    "pair_id": row["pair_id"], "anchor_sample_id": row["anchor_id"],
                    "sensor_sample_id": row["sensor_id"], "split": row["split"],
                    "activity": row["activity"],
                    "negative_relation": row["negative_relation"],
                    "target": int(row["label"]), "prediction": predicted,
                    "probabilities": probability.tolist(),
                })
    logits_array = np.concatenate(logits_all)
    labels_array = np.concatenate(labels_all)
    metrics = metric_summary(logits_array, labels_array)
    by_relation: dict[str, dict[str, float]] = {}
    for relation in ("cross_participant", "within_participant"):
        indexes = [i for i, row in enumerate(rows_all)
                   if row["negative_relation"] == relation]
        if indexes:
            by_relation[relation] = metric_summary(logits_array[indexes],
                                                   labels_array[indexes])
    metrics["by_negative_relation"] = by_relation
    pair_scores: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows_all:
        pair_scores[str(row["pair_id"])][int(row["target"])] = float(
            row["probabilities"][1]
        )
    pair_correct = [scores[1] > scores[0] for scores in pair_scores.values()
                    if set(scores) == {0, 1}]
    metrics["paired_candidate_accuracy"] = (
        float(np.mean(pair_correct)) if pair_correct else None
    )
    metrics["paired_questions"] = len(pair_correct)
    paired_metrics, _ = paired_choice_summary(rows_all)
    metrics["paired_choice_metrics"] = paired_metrics
    return metrics, rows_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--sensor-normalization", choices=("none", "train-zscore"), default="none",
        help="optional per-axis standardization from unique train-cache trials only",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("epochs, batch size, and learning rate must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("refusing to overwrite experiment output")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)
    features = load_features(args.cache_dir)
    pair_artifact = json.loads(args.pair_index.read_text(encoding="utf-8"))
    if pair_artifact.get("sealed_audit_accessed") is not False:
        raise SystemExit("pair index does not establish sealed-audit isolation")
    if set(pair_artifact.get("split_scope", ())) != {"train", "calibration", "validation"}:
        raise SystemExit("pair index contains a non-iterative split scope")
    if any(row.get("split") not in {"train", "calibration", "validation"}
           for row in features.values()):
        raise SystemExit("feature cache contains a sealed, quarantined or unknown split")
    sensor_normalization: dict[str, object] | None = None
    if args.sensor_normalization == "train-zscore":
        features, sensor_normalization = standardize_sensor_feature_cache_from_train(
            features
        )
    pair_rows = pair_artifact["pairs"]
    partitions = {
        split: PairExamples([pair for pair in pair_rows if pair["split"] == split],
                            features)
        for split in ("train", "calibration", "validation")
    }
    train_data = partitions["train"]
    calibration_data = partitions["calibration"]
    validation_data = partitions["validation"]
    if not train_data or not calibration_data or not validation_data:
        raise SystemExit("train/calibration/validation pair sets must all be nonempty")

    model = TimeAwareGatedFusion(
        {"audio": 384, "sensor": 3}, query_dim=2, width=96, heads=4,
        use_time=True, use_gates=True,
    ).to(device)
    scorer = nn.Linear(96, 1).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_count += sum(parameter.numel() for parameter in scorer.parameters())
    cache_index_path = args.cache_dir / "feature-cache-index.json"
    cache_index_sha = sha256_file(cache_index_path)
    pair_payload = {key: value for key, value in pair_artifact.items()
                    if key != "manifest_sha256"}
    pair_hash = hashlib.sha256(json.dumps(
        pair_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")).hexdigest()
    if pair_hash != pair_artifact.get("manifest_sha256"):
        raise SystemExit("pair index hash verification failed")
    effective_config = {
        "schema": "r2m_audio_vibration_gated_pilot_config_v1",
        "seed": SEED,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW(weight_decay=0.01)",
        "scheduler": "CosineAnnealingLR over full fixed epoch budget",
        "selection": "minimum calibration binary option NLL; train all requested epochs",
        "loss": "two-option cross entropy",
        "modality_dimensions": {"audio": 384, "sensor": 3},
        "query_codes": {"not_same_trial": [1.0, 0.0],
                         "same_trial": [0.0, 1.0]},
        "fusion_width": 96,
        "fusion_heads": 4,
        "use_time": True,
        "use_gates": True,
        "sensor_normalization": (
            sensor_normalization if sensor_normalization is not None
            else {"method": "none", "train_only": True}
        ),
        "trainable_parameters": parameter_count,
        "pair_index_sha256": pair_artifact["manifest_sha256"],
        "feature_cache_index_sha256": cache_index_sha,
        "feature_samples": len(features),
        "source_provenance": source_provenance(),
        "sealed_audit_accessed": False,
    }
    effective_config_path = args.output_dir / "effective-config.json"
    effective_config_path.write_text(
        json.dumps(effective_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *scorer.parameters()], lr=args.learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=generator, collate_fn=lambda rows: rows,
                              num_workers=0)
    history_path = args.output_dir / "training-history.jsonl"
    best_path = args.output_dir / "best-calibration-checkpoint.pt"
    final_path = args.output_dir / "final-checkpoint.pt"
    best_nll = math.inf
    best_epoch = 0
    global_step = 0
    run_start = time.perf_counter()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        scorer.train()
        loss_values: list[float] = []
        gradient_norms: list[float] = []
        for rows in train_loader:
            packed = collate(rows)
            streams = {
                key: TemporalFeatureStream(value.values.to(device),
                                           value.timestamps_ms.to(device),
                                           value.valid.to(device), value.present.to(device))
                for key, value in packed["streams"].items()
            }
            logits = scorer(model(streams, packed["queries"].to(device),
                                  packed["option_valid"].to(device))).squeeze(-1)
            labels = packed["labels"].to(device)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), *scorer.parameters()], 1.0
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("nonfinite gradient norm")
            gradient_norms.append(float(grad_norm.detach().cpu()))
            optimizer.step()
            global_step += 1
            loss_values.append(float(loss.detach().cpu()))

        calibration_metrics, _ = evaluate(
            model, scorer, calibration_data, device, batch_size=args.batch_size
        )
        scheduler.step()
        history = {
            "epoch": epoch,
            "optimizer_updates_total": global_step,
            "train_ce_rolling_mean": float(np.mean(loss_values)),
            "train_ce_metric_scope": (
                "per-minibatch rolling mean; not a fixed-example generalization gap"
            ),
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "calibration": calibration_metrics,
            "learning_rate_next_epoch": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": round(time.perf_counter() - run_start, 3),
        }
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(history, sort_keys=True) + "\n")
        if calibration_metrics["nll"] < best_nll:
            best_nll = float(calibration_metrics["nll"])
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "scorer": scorer.state_dict(),
                "calibration_metrics": calibration_metrics,
            }, best_path)
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(json.dumps({"epoch": epoch, "updates": global_step,
                              "train_ce": history["train_ce_rolling_mean"],
                              "calibration_nll": best_nll,
                              "best_epoch": best_epoch,
                              "elapsed_seconds": history["elapsed_seconds"]},
                             sort_keys=True), flush=True)

    torch.save({"epoch": args.epochs, "model": model.state_dict(),
                "scorer": scorer.state_dict()}, final_path)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    scorer.load_state_dict(checkpoint["scorer"])
    selected_validation, predictions = evaluate(
        model, scorer, validation_data, device, batch_size=args.batch_size
    )
    baseline_path = args.output_dir / "validation-temporal-correlation-baseline.jsonl"
    temporal_baseline, _ = temporal_correlation_baseline(
        validation_data, output_path=baseline_path
    )
    baseline_pair_path = args.output_dir / "validation-temporal-correlation-paired-choices.jsonl"
    # Predeclared no-stream controls and an explicit quarter-second time shift.
    controls = {}
    control_prediction_hashes = {}
    for mode in ("without_audio", "without_sensor", "sensor_plus_250ms"):
        controls[mode], control_predictions = evaluate(
            model, scorer, validation_data, device, batch_size=args.batch_size,
            mode=mode,
        )
        path = args.output_dir / f"validation-control-{mode}.jsonl"
        with path.open("w", encoding="utf-8") as output:
            for row in control_predictions:
                output.write(json.dumps(row, sort_keys=True) + "\n")
        control_prediction_hashes[mode] = sha256_file(path)
    mismatch_rows = []
    for row in validation_data.rows:
        if int(row["label"]) == 1:
            negative_id = validation_data.positive_to_negative[str(row["anchor_id"])]
            mismatch_rows.append({**row, "sensor_id": negative_id, "label": 0})
    mismatch = PairExamples([], features)
    mismatch.rows = mismatch_rows
    (controls["positive_audio_with_same_activity_wrong_trial_sensor"],
     mismatch_predictions) = evaluate(
        model, scorer, mismatch, device, batch_size=args.batch_size,
    )
    mismatch_path = args.output_dir / "validation-control-wrong-trial-sensor.jsonl"
    with mismatch_path.open("w", encoding="utf-8") as output:
        for row in mismatch_predictions:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    control_prediction_hashes["positive_audio_with_same_activity_wrong_trial_sensor"] = (
        sha256_file(mismatch_path)
    )
    prediction_path = args.output_dir / "validation-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as output:
        for row in predictions:
            output.write(json.dumps(row, sort_keys=True) + "\n")

    metrics = {
        "schema": "r2m_audio_vibration_gated_pilot_v1",
        "status": "COMPLETED",
        "seed": SEED,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "scheduler": "CosineAnnealingLR over the full fixed epoch budget",
        "sensor_normalization": effective_config["sensor_normalization"],
        "effective_config_sha256": sha256_file(effective_config_path),
        "parameter_count_trainable": parameter_count,
        "selected_checkpoint_epoch_by_calibration_nll": best_epoch,
        "calibration_best_nll": best_nll,
        "validation_selected_checkpoint": selected_validation,
        "validation_fixed_temporal_correlation_baseline": temporal_baseline,
        "validation_controls": controls,
        "pair_index_sha256": pair_artifact["manifest_sha256"],
        "pair_count_by_split": pair_artifact["pair_count_by_split"],
        "feature_sample_count": len(features),
        "feature_cache_index_sha256": sha256_file(
            args.cache_dir / "feature-cache-index.json"
        ),
        "best_checkpoint_sha256": sha256_file(best_path),
        "final_checkpoint_sha256": sha256_file(final_path),
        "training_history_sha256": sha256_file(history_path),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "validation_baseline_predictions_sha256": sha256_file(baseline_path),
        "validation_baseline_paired_choices_sha256": sha256_file(baseline_pair_path),
        "validation_control_predictions_sha256": control_prediction_hashes,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "wall_seconds": round(time.perf_counter() - run_start, 3),
        "sealed_audit_accessed": False,
        "product_admission_claimed": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"validation": selected_validation,
                      "controls": controls,
                      "best_epoch": best_epoch,
                      "trainable_parameters": parameter_count,
                      "peak_vram_bytes": metrics["peak_vram_bytes"],
                      "wall_seconds": metrics["wall_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
