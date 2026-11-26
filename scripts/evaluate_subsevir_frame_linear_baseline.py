"""Fit one frozen-feature last-frame linear baseline for sub-SEVIR.

This is a single, fixed diagnostic on the already exposed Path A validation
split. It uses no validation labels during fitting or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

SEED = 17
BOOTSTRAP_REPLICATES = 10_000
L2_WEIGHT = 1e-4
LBFGS_MAX_ITER = 50
EXPECTED_FEATURE_DIM = 2 * 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def compute_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    if probabilities.shape != (len(targets), 2):
        raise ValueError("probabilities must have shape [examples, 2]")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))
    one_hot = np.eye(2, dtype=np.float64)[targets]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15.0, (index + 1) / 15.0
        selected = (confidence >= lower) & (
            (confidence < upper) if index < 14 else (confidence <= upper)
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
        "correct_count": int(correct.sum()),
        "confusion": {
            "tn": int(((targets == 0) & (predictions == 0)).sum()),
            "fp": int(((targets == 0) & (predictions == 1)).sum()),
            "fn": int(((targets == 1) & (predictions == 0)).sum()),
            "tp": int(((targets == 1) & (predictions == 1)).sum()),
        },
    }


def group_bootstrap(
    targets: np.ndarray,
    probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    group_ids: list[str],
) -> dict[str, object]:
    groups = sorted(set(group_ids))
    positions = {group: index for index, group in enumerate(groups)}
    sums = np.zeros((len(groups), 3), dtype=np.float64)
    ref_sums = np.zeros_like(sums)
    counts = np.zeros(len(groups), dtype=np.float64)
    for row, group in enumerate(group_ids):
        index = positions[group]
        true = int(targets[row])
        pred = int(probabilities[row].argmax())
        ref_pred = int(reference_probabilities[row].argmax())
        sums[index] += (
            float(pred == true),
            float(-np.log(max(float(probabilities[row, true]), 1e-12))),
            float(np.square(probabilities[row] - np.eye(2)[true]).sum()),
        )
        ref_sums[index] += (
            float(ref_pred == true),
            float(-np.log(max(float(reference_probabilities[row, true]), 1e-12))),
            float(np.square(reference_probabilities[row] - np.eye(2)[true]).sum()),
        )
        counts[index] += 1

    rng = np.random.default_rng(SEED)
    sampled = rng.integers(
        0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups))
    )
    denominator = counts[sampled].sum(axis=1)
    names = ("accuracy", "nll", "brier")
    output: dict[str, object] = {"unit": "event_id", "group_count": len(groups)}
    for column, name in enumerate(names):
        values = sums[:, column][sampled].sum(axis=1) / denominator
        differences = (sums[:, column] - ref_sums[:, column])[sampled].sum(axis=1)
        differences /= denominator
        output[name] = [float(x) for x in np.quantile(values, [0.025, 0.975])]
        output[f"difference_vs_joint_lstm_{name}"] = [
            float(x) for x in np.quantile(differences, [0.025, 0.975])
        ]
    output["replicates"] = BOOTSTRAP_REPLICATES
    output["seed"] = SEED
    return output


def load_rows(
    manifest: dict[str, object], frame_indices: np.ndarray, feature_map: np.ndarray
) -> dict[str, list[dict[str, object]]]:
    index_to_feature = {int(source): i for i, source in enumerate(frame_indices)}
    if len(index_to_feature) != len(frame_indices):
        raise ValueError("feature cache contains duplicate source frame indices")
    rows: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    for window in manifest["windows"]:
        split = str(window["split"])
        sample_id = str(window["sample_id"])
        last_input_index = int(window["input_indices"][-1])
        try:
            cache_row = index_to_feature[last_input_index]
        except KeyError as exc:
            raise ValueError(f"last input frame missing from cache: {last_input_index}") from exc
        rows[split].append({
            "sample_id": sample_id,
            "event_id": str(window["event_id"]),
            "target": int(window["target"]),
            "feature": np.asarray(feature_map[cache_row], dtype=np.float32).reshape(-1),
        })
    if any(not rows[split] for split in rows):
        raise ValueError("manifest has an empty train or validation split")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--frame-indices", type=Path, required=True)
    parser.add_argument("--feature-metadata", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")

    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    features = np.load(args.features, mmap_mode="r", allow_pickle=False)
    frame_indices = np.load(args.frame_indices, mmap_mode="r", allow_pickle=False)
    if features.ndim != 3 or tuple(features.shape[1:]) != (2, 768):
        raise ValueError(f"unexpected feature shape: {features.shape}")
    if len(features) != len(frame_indices):
        raise ValueError("feature and frame-index counts differ")
    metadata = json.loads(args.feature_metadata.read_text(encoding="utf-8"))
    if metadata["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("feature cache and manifest hashes do not match")

    rows = load_rows(manifest, frame_indices, features)
    train_x = np.stack([row["feature"] for row in rows["train"]])
    train_y = np.asarray([row["target"] for row in rows["train"]], dtype=np.int64)
    validation_x = np.stack([row["feature"] for row in rows["validation"]])
    validation_y = np.asarray([row["target"] for row in rows["validation"]], dtype=np.int64)
    train_groups = [str(row["event_id"]) for row in rows["train"]]
    validation_groups = [str(row["event_id"]) for row in rows["validation"]]
    if set(train_groups) & set(validation_groups):
        raise ValueError("train/validation event overlap")

    reference = {
        item["sample_id"]: item
        for item in (json.loads(line) for line in args.reference_predictions.read_text(
            encoding="utf-8"
        ).splitlines())
        if item.get("sample_id")
    }
    validation_ids = [str(row["sample_id"]) for row in rows["validation"]]
    if set(reference) != set(validation_ids):
        raise ValueError("reference predictions do not match validation sample IDs")
    reference_rows = [reference[sample_id] for sample_id in validation_ids]
    if any(int(item["target"]) != int(row["target"]) for item, row in zip(
        reference_rows, rows["validation"], strict=True
    )):
        raise ValueError("reference prediction targets differ from frozen manifest")
    reference_probabilities = np.asarray(
        [item["probabilities"] for item in reference_rows], dtype=np.float64
    )

    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train_t = torch.from_numpy((train_x - mean) / scale)
    validation_t = torch.from_numpy((validation_x - mean) / scale)
    train_target_t = torch.from_numpy(train_y)
    model = nn.Linear(EXPECTED_FEATURE_DIM, 2, dtype=torch.float32)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    started = time.perf_counter()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_t)
        loss = F.cross_entropy(logits, train_target_t)
        loss = loss + 0.5 * L2_WEIGHT * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    fit_seconds = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        str(args.output / "linear-head.safetensors"),
    )
    np.savez_compressed(args.output / "train-standardization.npz", mean=mean, scale=scale)
    reloaded_model = nn.Linear(EXPECTED_FEATURE_DIM, 2, dtype=torch.float32)
    reloaded_model.load_state_dict(load_file(str(args.output / "linear-head.safetensors")))
    with np.load(args.output / "train-standardization.npz") as standardization:
        reload_mean = standardization["mean"]
        reload_scale = standardization["scale"]
    with torch.no_grad():
        reload_train_t = torch.from_numpy((train_x - reload_mean) / reload_scale)
        reload_validation_t = torch.from_numpy((validation_x - reload_mean) / reload_scale)
        train_probabilities = torch.softmax(reloaded_model(reload_train_t), dim=1).numpy()
        validation_probabilities = torch.softmax(
            reloaded_model(reload_validation_t), dim=1
        ).numpy()
        original_validation_probabilities = torch.softmax(model(validation_t), dim=1).numpy()
    reload_max_probability_delta = float(
        np.max(np.abs(validation_probabilities - original_validation_probabilities))
    )
    if reload_max_probability_delta > 1e-7:
        raise ValueError(
            f"saved artifact reload changed probabilities by {reload_max_probability_delta}"
        )

    train_metrics = compute_metrics(train_y, train_probabilities)
    validation_metrics = compute_metrics(validation_y, validation_probabilities)
    ci = group_bootstrap(
        validation_y,
        validation_probabilities,
        reference_probabilities,
        validation_groups,
    )
    predictions_path = args.output / "validation-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, probabilities in zip(rows["validation"], validation_probabilities, strict=True):
            stream.write(json.dumps({
                "sample_id": row["sample_id"],
                "event_id": row["event_id"],
                "target": row["target"],
                "prediction": int(probabilities.argmax()),
                "probabilities": [float(x) for x in probabilities],
            }, sort_keys=True) + "\n")
    summary = {
        "schema": "tiny-omni-decision/subsevir-frame-linear-baseline-v1",
        "status": "complete",
        "claim_scope": (
            "single diagnostic on already exposed Path A validation; "
            "not selected/promoted"
        ),
        "initialization": "fresh 2-class linear head, seed 17",
        "input": (
            "last preceding frame; concatenated frozen DINOv2-small features "
            "for satellite and radar"
        ),
        "feature_dimension": EXPECTED_FEATURE_DIM,
        "trainable_parameters": EXPECTED_FEATURE_DIM * 2 + 2,
        "frozen_encoder_parameters": int(metadata["parameters"]),
        "fit": {
            "optimizer": "full-batch torch LBFGS with strong Wolfe line search",
            "max_iterations": LBFGS_MAX_ITER,
            "l2_weight": L2_WEIGHT,
            "normalization": "feature mean/std fit on train windows only; std < 1e-6 set to 1",
            "validation_used_for_fit_or_selection": False,
            "device": "CPU",
            "wall_seconds": fit_seconds,
        },
        "source_hashes": {
            "manifest_sha256": sha256_file(args.manifest),
            "feature_cache_metadata_sha256": sha256_file(args.feature_metadata),
            "feature_file_sha256": sha256_file(args.features),
            "frame_indices_sha256": sha256_file(args.frame_indices),
            "reference_predictions_sha256": sha256_file(args.reference_predictions),
        },
        "split": {
            "train_samples": len(rows["train"]),
            "train_unique_events": len(set(train_groups)),
            "validation_samples": len(rows["validation"]),
            "validation_unique_events": len(set(validation_groups)),
            "validation_sample_ids_sha256": canonical_json_sha256(validation_ids),
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "saved_artifact_reload_max_probability_delta": reload_max_probability_delta,
        "paired_event_bootstrap_95pct_vs_selected_joint_lstm": ci,
        "selected_joint_lstm_metrics": compute_metrics(
            validation_y, reference_probabilities
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "CPU; no GPU used",
        },
        "artifacts": {
            "validation_predictions_sha256": sha256_file(predictions_path),
            "linear_head_sha256": sha256_file(args.output / "linear-head.safetensors"),
            "standardization_sha256": sha256_file(args.output / "train-standardization.npz"),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
