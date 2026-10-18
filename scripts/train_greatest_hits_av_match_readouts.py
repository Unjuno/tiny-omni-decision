"""Fit fixed-seed audio-only, video-only and joint AV readouts on frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/pretrained_reuse/greatest_hits_av_match_readouts_v0.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_metrics(targets: list[int], probabilities: list[list[float]]) -> dict[str, float]:
    if not targets or len(targets) != len(probabilities):
        raise ValueError("nonempty paired targets/probabilities required")
    accuracy = 0.0
    nll = 0.0
    brier = 0.0
    confidence = 0.0
    ece = 0.0
    bin_count = 15
    bins = [[] for _ in range(bin_count)]
    for target, values in zip(targets, probabilities, strict=True):
        if len(values) != 2 or any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("finite binary probability vectors required")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        prediction = max(range(2), key=values.__getitem__)
        accuracy += float(prediction == target)
        nll -= math.log(max(values[target], 1e-12))
        brier += sum((values[index] - float(index == target)) ** 2 for index in range(2))
        conf = values[prediction]
        confidence += conf
        bins[min(bin_count - 1, int(conf * bin_count))].append((conf, float(prediction == target)))
    count = len(targets)
    for current in bins:
        if current:
            mean_conf = sum(item[0] for item in current) / len(current)
            mean_acc = sum(item[1] for item in current) / len(current)
            ece += len(current) / count * abs(mean_conf - mean_acc)
    return {
        "count": count,
        "accuracy": accuracy / count,
        "nll": nll / count,
        "brier": brier / count,
        "ece_15": ece,
        "mean_confidence": confidence / count,
    }


def metric_for_rows(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, float]:
    return probability_metrics(
        [int(rows[index]["target"]) for index in indices],
        [rows[index]["probabilities"] for index in indices],
    )


def paired_cluster_bootstrap(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    import numpy as np

    if [row["id"] for row in rows_a] != [row["id"] for row in rows_b]:
        raise ValueError("paired bootstrap requires identical ordered decision IDs")
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows_a):
        by_cluster[str(row["pair_cluster_id"])].append(index)
    cluster_ids = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {"accuracy": [], "nll": [], "brier": []}
    for _ in range(replicates):
        sample_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sample_indices = [
            index for cluster in sample_clusters for index in by_cluster[str(cluster)]
        ]
        metrics_a = metric_for_rows(rows_a, sample_indices)
        metrics_b = metric_for_rows(rows_b, sample_indices)
        for metric in deltas:
            deltas[metric].append(metrics_a[metric] - metrics_b[metric])
    intervals = {}
    for metric, values in deltas.items():
        ordered = sorted(values)
        intervals[metric] = {
            "mean_delta": sum(values) / replicates,
            "ci95_percentile": [
                ordered[int(0.025 * replicates)],
                ordered[min(replicates - 1, int(0.975 * replicates))],
            ],
        }
    return {
        "cluster_count": len(cluster_ids),
        "replicates": replicates,
        "seed": seed,
        "delta_a_minus_b": intervals,
    }


def _load_feature_map(features_dir: Path, expected_rows: int) -> dict[str, Any]:
    import numpy as np

    receipt = json.loads((features_dir / "result.json").read_text(encoding="utf-8"))
    manifest_path = features_dir / "features.jsonl"
    if (
        receipt.get("status") != "PASS_FROZEN_FEATURE_EXTRACTION_REPLAY"
        or receipt.get("completed") != expected_rows
        or receipt.get("features_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError(f"complete hash-matched features required: {features_dir}")
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    feature_map = {}
    for row in manifest:
        path = features_dir / row["feature_file"]
        if sha256_file(path) != row["feature_sha256"]:
            raise ValueError("feature hash mismatch")
        value = np.load(path, allow_pickle=False)
        if value.shape != tuple(row["feature_shape"]) or not np.isfinite(value).all():
            raise ValueError("invalid/nonfinite feature")
        feature_map[row["source_group"]] = value.astype(np.float32, copy=False)
    if len(feature_map) != expected_rows:
        raise ValueError("duplicate or missing source-group feature rows")
    return feature_map


def _read_corpus(preflight_dir: Path) -> dict[str, list[dict[str, Any]]]:
    expected_rows = {"train": 252, "calibration": 76, "validation": 112}
    result_path = preflight_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    corpora = {}
    for split, count in expected_rows.items():
        path = preflight_dir / f"{split}.jsonl"
        if result.get("output_sha256", {}).get(path.name) != sha256_file(path):
            raise ValueError(f"frozen {split} Decision JSONL hash mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != count:
            raise ValueError(f"frozen {split} Decision row count mismatch")
        if any(
            row["options"]
            != ["No, they are different recordings.", "Yes, they are the same recording."]
            for row in rows
        ):
            raise ValueError("binary option order differs from the frozen Decision corpus")
        if any(int(row["target"]) not in (0, 1) for row in rows):
            raise ValueError("binary targets required")
        corpora[split] = rows
    return corpora


def train_experiment(config_path: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.optim import LBFGS, AdamW

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "greatest-hits-av-match-readouts-v0":
        raise ValueError("unexpected experiment configuration")
    if config["optimizer"]["device"] != "cpu" or config["optimizer"]["epochs"] != 100:
        raise ValueError("only the preregistered CPU 100-epoch budget is allowed")
    if config["evaluation"]["sealed_audit_opened"]:
        raise ValueError("sealed audit must remain closed")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    paths = {key: Path(value) for key, value in config["dataset"].items() if isinstance(value, str)}
    preflight_dir = paths["preflight"]
    inputs_dir = paths["inputs"]
    audio_dir = paths["audio_features"]
    video_dir = paths["video_features"]
    corpora = _read_corpus(preflight_dir)
    input_audit_dir = Path(config["dataset"]["input_audit"])
    input_audit_path = input_audit_dir / "audit.json"
    input_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
    if (input_audit_dir / "audit.sha256").read_text(encoding="ascii").strip() != sha256_file(
        input_audit_path
    ):
        raise ValueError("input audit digest sidecar mismatch")
    if input_audit.get("status") != "PASS_INPUT_ARRAY_RELOAD_AND_CONTENT_DISJOINTNESS":
        raise ValueError("independent input-array integrity audit is required")
    if (
        input_audit.get("normalized_cross_split_overlap") != 0
        or input_audit.get("raw_cross_split_overlap") != 0
    ):
        raise ValueError("input content overlap gate failed")

    audio_features = _load_feature_map(audio_dir, 220)
    video_features = _load_feature_map(video_dir, 220)
    for modality, feature_dir in (("audio", audio_dir), ("video", video_dir)):
        feature_receipt = json.loads((feature_dir / "result.json").read_text(encoding="utf-8"))
        encoder = config["encoders"][modality]
        if (
            feature_receipt.get("revision") != encoder["revision"]
            or feature_receipt.get("checkpoint_sha256") != encoder["checkpoint_sha256"]
            or feature_receipt.get("parameters") != encoder["parameters"]
            or feature_receipt.get("feature_shape") != [encoder["feature_size"]]
        ):
            raise ValueError(f"pinned {modality} encoder receipt differs from experiment config")
    for replay_key in ("audio_replay", "video_replay"):
        replay_dir = paths[replay_key]
        audit_path = replay_dir / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (replay_dir / "audit.sha256").read_text(encoding="ascii").strip() != sha256_file(
            audit_path
        ):
            raise ValueError(f"feature replay audit digest mismatch: {replay_key}")
        if (
            audit.get("status") != "PASS_FRESH_PROCESS_FEATURE_REPLAY"
            or audit.get("completed") != 220
        ):
            raise ValueError(f"complete fresh-process replay is required: {replay_key}")
        if audit.get("max_abs_delta") is None or audit["max_abs_delta"] > 1e-6:
            raise ValueError(f"feature replay numerical threshold failed: {replay_key}")

    for split, rows in corpora.items():
        used_groups = {
            row[field] for row in rows for field in ("audio_source_group", "video_source_group")
        }
        if not used_groups <= set(audio_features) or not used_groups <= set(video_features):
            raise ValueError(f"feature set does not cover frozen {split} source groups")

    class LinearHead(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.linear = nn.Linear(input_size, 2)

        def forward(self, audio, video):
            value = audio if audio is not None else video
            return self.linear(value)

    class PairedCosine(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_projection = nn.Linear(384, 64, bias=False)
            self.video_projection = nn.Linear(768, 64, bias=False)

        def forward(self, audio, video):
            left = F.normalize(self.audio_projection(audio), dim=-1)
            right = F.normalize(self.video_projection(video), dim=-1)
            score = (left * right).sum(dim=-1) / 0.1
            return torch.stack((-score, score), dim=-1)

    class_conditions = ("audio_only", "video_only", "joint_av")

    def tensor_split(split: str):
        rows = corpora[split]
        audio = torch.tensor(
            np.stack([audio_features[row["audio_source_group"]] for row in rows]),
            dtype=torch.float32,
        )
        video = torch.tensor(
            np.stack([video_features[row["video_source_group"]] for row in rows]),
            dtype=torch.float32,
        )
        targets = torch.tensor([int(row["target"]) for row in rows], dtype=torch.long)
        clusters: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            clusters[str(row["pair_cluster_id"])].append(index)
        if any(len(indices) != 4 for indices in clusters.values()):
            raise ValueError("every frozen pair cluster must contain four Decision rows")
        if any(
            sorted(int(targets[index]) for index in indices) != [0, 0, 1, 1]
            for indices in clusters.values()
        ):
            raise ValueError("every pair cluster must retain two match and two mismatch targets")
        return rows, audio, video, targets, dict(clusters)

    tensorized = {split: tensor_split(split) for split in corpora}
    train_rows, train_audio, train_video, train_targets, train_clusters = tensorized["train"]
    history: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    final_metrics: dict[str, Any] = {}
    checkpoint_metadata: dict[str, Any] = {}
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "RUNNING",
        "experiment_id": config["experiment_id"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256_file(config_path),
        "config_path": str(config_path),
        "inputs_result_sha256": sha256_file(inputs_dir / "result.json"),
        "preflight_result_sha256": sha256_file(preflight_dir / "result.json"),
        "audio_feature_result_sha256": sha256_file(audio_dir / "result.json"),
        "video_feature_result_sha256": sha256_file(video_dir / "result.json"),
        "input_audit_sha256": sha256_file(input_audit_dir / "audit.json"),
        "seed": seed,
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "encoder_parameters": sum(
            int(value["parameters"]) for value in config["encoders"].values()
        ),
        "readout_parameters": {"audio_only": 770, "video_only": 1538, "joint_av": 73728},
        "total_parameters_by_condition": {
            "audio_only": 1961239 + 770,
            "video_only": 114365184 + 1538,
            "joint_av": 1961239 + 114365184 + 73728,
        },
        "split_accounting": {},
        "training": True,
        "gpu_used": False,
        "cloud_cost": 0,
        "sealed_audit_opened": False,
        "conditions": {},
    }
    (output_dir / "effective-config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def construct(condition: str):
        torch.manual_seed(seed)
        if condition == "audio_only":
            return LinearHead(384)
        if condition == "video_only":
            return LinearHead(768)
        return PairedCosine()

    def logits_for(model, condition, audio, video):
        if condition == "audio_only":
            return model(audio, None)
        if condition == "video_only":
            return model(None, video)
        return model(audio, video)

    def evaluate(model, condition, split: str):
        rows, audio, video, targets, _ = tensorized[split]
        with torch.inference_mode():
            logits = logits_for(model, condition, audio, video)
            loss = F.cross_entropy(logits, targets).item()
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
        metrics = probability_metrics(targets.tolist(), probabilities)
        metrics["cross_entropy"] = loss
        return logits.detach(), metrics, probabilities

    try:
        epochs = int(config["optimizer"]["epochs"])
        for condition in class_conditions:
            model = construct(condition)
            optimizer = AdamW(
                model.parameters(),
                lr=float(config["optimizer"]["learning_rate"]),
                weight_decay=float(config["optimizer"]["weight_decay"]),
            )
            condition_history = []
            best_nll = math.inf
            best_epoch = None
            best_state = None
            last_state = None
            updates = 0
            train_presentations = 0
            condition_start = time.perf_counter()
            for epoch in range(epochs + 1):
                if epoch > 0:
                    model.train()
                    cluster_names = sorted(train_clusters)
                    permutation = torch.randperm(len(cluster_names)).tolist()
                    for position in permutation:
                        cluster = cluster_names[position]
                        indices = train_clusters[cluster]
                        index_tensor = torch.tensor(indices, dtype=torch.long)
                        logits = logits_for(
                            model,
                            condition,
                            train_audio.index_select(0, index_tensor),
                            train_video.index_select(0, index_tensor),
                        )
                        loss = F.cross_entropy(logits, train_targets.index_select(0, index_tensor))
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        updates += 1
                        train_presentations += len(indices)
                    model.eval()
                _, train_metrics, _ = evaluate(model, condition, "train")
                _, validation_metrics, _ = evaluate(model, condition, "validation")
                row = {
                    "condition": condition,
                    "epoch": epoch,
                    "optimizer_updates": updates,
                    "train_examples_presented": train_presentations,
                    "train_unique_examples": len(train_rows),
                    "train_unique_pair_clusters": len(train_clusters),
                    "learning_rate": float(config["optimizer"]["learning_rate"]),
                    "train_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                    "elapsed_seconds": time.perf_counter() - condition_start,
                }
                condition_history.append(row)
                history.append(row)
                with (output_dir / "training-history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                if validation_metrics["nll"] < best_nll:
                    best_nll = validation_metrics["nll"]
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                if epoch == epochs:
                    last_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                (output_dir / "result.json").write_text(
                    json.dumps(
                        {**result, "completed_condition": condition, "completed_epoch": epoch},
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            final_metrics[condition] = condition_history[-1]
            assert best_state is not None and last_state is not None and best_epoch is not None
            best_path = output_dir / f"{condition}-best.pt"
            final_path = output_dir / f"{condition}-final.pt"
            torch.save({"epoch": best_epoch, "state_dict": best_state}, best_path)
            torch.save({"epoch": epochs, "state_dict": last_state}, final_path)

            model.load_state_dict(
                torch.load(best_path, map_location="cpu", weights_only=True)["state_dict"],
                strict=True,
            )
            model.eval()
            best_val_logits, best_val_metrics, _ = evaluate(model, condition, "validation")
            cal_rows, cal_audio, cal_video, cal_targets, _ = tensorized["calibration"]
            with torch.inference_mode():
                cal_logits = logits_for(model, condition, cal_audio, cal_video).detach()
            log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            temperature_optimizer = LBFGS(
                [log_temperature], max_iter=100, line_search_fn="strong_wolfe"
            )

            def temperature_closure(
                optimizer=temperature_optimizer,
                log_temp=log_temperature,
                logits=cal_logits,
                targets=cal_targets,
            ):
                optimizer.zero_grad()
                temperature = log_temp.exp().clamp(0.05, 20.0)
                calibration_loss = F.cross_entropy(logits / temperature, targets)
                calibration_loss.backward()
                return calibration_loss

            temperature_optimizer.step(temperature_closure)
            temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("calibration temperature is invalid")
            calibrated_val_probabilities = (
                torch.softmax(best_val_logits / temperature, dim=1).cpu().tolist()
            )
            calibrated_val_rows = [
                {
                    **row,
                    "prediction": max(
                        range(2), key=calibrated_val_probabilities[index].__getitem__
                    ),
                    "probabilities": calibrated_val_probabilities[index],
                }
                for index, row in enumerate(corpora["validation"])
            ]
            cal_probabilities = torch.softmax(cal_logits / temperature, dim=1).cpu().tolist()
            calibrated_val_metrics = probability_metrics(
                [int(row["target"]) for row in corpora["validation"]],
                calibrated_val_probabilities,
            )
            calibration_metrics = probability_metrics(cal_targets.tolist(), cal_probabilities)
            signature_metrics = {}
            for signature in sorted(
                {tuple(row["source_signature"]) for row in corpora["validation"]}
            ):
                indices = [
                    index
                    for index, row in enumerate(calibrated_val_rows)
                    if tuple(row["source_signature"]) == signature
                ]
                signature_metrics["|".join(signature)] = metric_for_rows(
                    calibrated_val_rows, indices
                )
            selected[condition] = {
                "best_epoch": best_epoch,
                "final_epoch": epochs,
                "checkpoint_selection_metric": "minimum uncalibrated validation NLL; earliest tie",
                "best_validation_uncalibrated": best_val_metrics,
                "validation_after_calibration": calibrated_val_metrics,
                "calibration_split_metrics": calibration_metrics,
                "temperature": temperature,
                "source_signature_validation_metrics": signature_metrics,
                "train_at_final": final_metrics[condition]["train_metrics"],
                "validation_at_final": final_metrics[condition]["validation_metrics"],
                "overfit_signals": {
                    "train_nll_fell_after_selected_epoch": final_metrics[condition][
                        "train_metrics"
                    ]["nll"]
                    < condition_history[best_epoch]["train_metrics"]["nll"],
                    "validation_nll_rose_after_selected_epoch": final_metrics[condition][
                        "validation_metrics"
                    ]["nll"]
                    > best_val_metrics["nll"],
                    "train_accuracy_rose_validation_accuracy_fell": final_metrics[condition][
                        "train_metrics"
                    ]["accuracy"]
                    > condition_history[best_epoch]["train_metrics"]["accuracy"]
                    and final_metrics[condition]["validation_metrics"]["accuracy"]
                    < best_val_metrics["accuracy"],
                },
                "optimizer_updates": updates,
                "train_rows_presented": train_presentations,
                "checkpoint_bytes": {
                    "best": best_path.stat().st_size,
                    "final": final_path.stat().st_size,
                },
                "checkpoint_sha256": {
                    "best": sha256_file(best_path),
                    "final": sha256_file(final_path),
                },
            }
            checkpoint_metadata[condition] = {
                "best_path": best_path.name,
                "final_path": final_path.name,
            }
            if condition == "joint_av":
                best_joint_rows = calibrated_val_rows
            elif condition == "audio_only":
                best_audio_rows = calibrated_val_rows
            else:
                best_video_rows = calibrated_val_rows

        bootstrap_replicates = int(config["evaluation"]["bootstrap_replicates"])
        comparisons = {
            "joint_av_minus_audio_only": paired_cluster_bootstrap(
                best_joint_rows, best_audio_rows, bootstrap_replicates, seed
            ),
            "joint_av_minus_video_only": paired_cluster_bootstrap(
                best_joint_rows, best_video_rows, bootstrap_replicates, seed
            ),
        }
        pred_path = output_dir / "validation-predictions.jsonl"
        pred_path.write_text(
            "".join(
                json.dumps({"condition": condition, **row}, sort_keys=True) + "\n"
                for condition, rows in (
                    ("audio_only", best_audio_rows),
                    ("video_only", best_video_rows),
                    ("joint_av", best_joint_rows),
                )
                for row in rows
            ),
            encoding="utf-8",
        )
        result.update(
            status="PASS_COMPLETED_ALL_FIXED_READOUTS",
            finished_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            split_accounting={
                split: {
                    "rows": len(rows),
                    "pair_clusters": len({row["pair_cluster_id"] for row in rows}),
                    "unique_audio_assets": len({row["audio_source_group"] for row in rows}),
                    "unique_video_assets": len({row["video_source_group"] for row in rows}),
                }
                for split, rows in corpora.items()
            },
            history_rows=len(history),
            predictions_sha256=sha256_file(pred_path),
            conditions=selected,
            paired_cluster_bootstrap=comparisons,
            low_power=len({row["pair_cluster_id"] for row in corpora["validation"]})
            < int(config["evaluation"]["low_power_if_clusters_below"]),
            checkpoint_files=checkpoint_metadata,
            total_raw_prequantization_parameters={
                "audio_encoder_plus_readout": 1961239 + 770,
                "video_encoder_plus_readout": 114365184 + 1538,
                "joint_audio_video_plus_readout": 1961239 + 114365184 + 73728,
            },
        )
    except Exception as error:
        result.update(status="FAILED", error=repr(error))
        raise
    finally:
        result["finished_utc"] = datetime.now(UTC).isoformat()
        result["elapsed_seconds"] = time.perf_counter() - started
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = train_experiment(args.config, args.output_dir)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "elapsed_seconds",
                    "split_accounting",
                    "conditions",
                    "low_power",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
