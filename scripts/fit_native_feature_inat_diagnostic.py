"""One fixed-budget diagnostic for the text-free native-feature Decision readout.

This is an eight-taxon image classification development diagnostic, not a
general or blind Decision benchmark. It consumes the frozen iNaturalist
feature cache created by the pinned VideoPrism pipeline and never loads media
or the encoder. Validation is logged but never used for selection or stopping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from tiny_omni_decision.native_feature_decision import (
    EnumOptionField,
    NativeFeatureDecisionScorer,
    TypedOptionSchema,
)

FEATURE_DIR = Path(
    r"C:\CodexArtifacts\inat-birds-cc0-expansion-freshval-v2-merged-features-20261029-attempt-01"
)
BASELINE_DIR = Path(
    r"C:\CodexArtifacts\inat-birds-cc0-expansion-videoprism-linear-readout-freshval-20261029-attempt-01"
)
EXPECTED = {
    "features.npy": "f19fb03b8ee6015b20a8e153a38c7bdce3e008e3761b2de4953f7bc2c0cf1913",
    "feature-rows.jsonl": "5550167e42b076c71720d9d0d4abb738ba5af0edfc8eed4b2c26cea0dd496ae3",
    "validation-predictions.jsonl": (
        "9266ffecfeef7795df38a37dd8cab9cfc89900b54e3beca69b4a28728ba32173"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")
    return actual


def load_data() -> tuple[np.ndarray, list[dict], list[str], dict[str, str]]:
    feature_path = FEATURE_DIR / "features.npy"
    rows_path = FEATURE_DIR / "feature-rows.jsonl"
    baseline_path = BASELINE_DIR / "validation-predictions.jsonl"
    hashes = {
        "features": _verified(feature_path, EXPECTED["features.npy"]),
        "feature_rows": _verified(rows_path, EXPECTED["feature-rows.jsonl"]),
        "baseline_predictions": _verified(baseline_path, EXPECTED["validation-predictions.jsonl"]),
    }
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    features = np.load(feature_path, allow_pickle=False)
    if features.dtype != np.float32 or features.shape != (len(rows), 768):
        raise ValueError("unexpected frozen feature matrix shape or dtype")
    if not np.isfinite(features).all() or len(rows) != 1575:
        raise ValueError("feature cache is incomplete or nonfinite")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs")
    classes = sorted({str(row["taxon_id"]) for row in rows})
    if len(classes) != 8:
        raise ValueError("expected the fixed eight-taxon task")
    train = [row for row in rows if row["split"] == "train"]
    valid = [row for row in rows if row["split"] == "validation"]
    if (len(train), len(valid)) != (1145, 430):
        raise ValueError("unexpected fixed cohort size")
    for field in ("observer_group_sha256", "observation_id", "photo_id"):
        if {row[field] for row in train} & {row[field] for row in valid}:
            raise ValueError(f"train/validation {field} overlap")
    if set(classes) - {str(row["taxon_id"]) for row in train}:
        raise ValueError("a candidate class is absent from train")
    if set(classes) - {str(row["taxon_id"]) for row in valid}:
        raise ValueError("a candidate class is absent from validation")
    baseline = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines()]
    if len(baseline) != len(valid):
        raise ValueError("baseline predictions do not cover validation")
    if [row["sample_id"] for row in baseline] != [row["sample_id"] for row in valid]:
        raise ValueError("baseline and candidate validation order differs")
    return features, rows, classes, hashes


def metric_summary(logits: Tensor, targets: Tensor, classes: list[str]) -> dict:
    probs = torch.softmax(logits.float(), dim=-1)
    n = targets.numel()
    onehot = F.one_hot(targets, num_classes=len(classes)).to(probs.dtype)
    confidence, prediction = probs.max(dim=-1)
    bin_edges = torch.linspace(0, 1, 16, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for index in range(15):
        mask = (confidence > bin_edges[index]) & (confidence <= bin_edges[index + 1])
        if mask.any():
            ece += (
                mask.float().mean()
                * (
                    confidence[mask].mean() - prediction[mask].eq(targets[mask]).float().mean()
                ).abs()
            )
    per_class = {}
    for index, class_id in enumerate(classes):
        mask = targets == index
        per_class[class_id] = {
            "count": int(mask.sum()),
            "accuracy": float(prediction[mask].eq(targets[mask]).float().mean()),
            "nll": float(F.cross_entropy(logits[mask].float(), targets[mask])),
        }
    return {
        "count": n,
        "accuracy": float(prediction.eq(targets).float().mean()),
        "macro_accuracy": float(np.mean([value["accuracy"] for value in per_class.values()])),
        "nll": float(F.cross_entropy(logits.float(), targets)),
        "brier": float(((probs - onehot) ** 2).sum(-1).mean()),
        "ece_15_bins": float(ece),
        "mean_confidence": float(confidence.mean()),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output must be a fresh directory: {args.output}")

    features_np, rows, classes, hashes = load_data()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    torch.use_deterministic_algorithms(True)

    train_indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    valid_indices = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    train_x = torch.from_numpy(features_np[train_indices].copy())
    valid_x = torch.from_numpy(features_np[valid_indices].copy())
    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / scale
    valid_x = (valid_x - mean) / scale
    class_index = {value: i for i, value in enumerate(classes)}
    train_y = torch.tensor([class_index[str(rows[i]["taxon_id"])] for i in train_indices])
    valid_y = torch.tensor([class_index[str(rows[i]["taxon_id"])] for i in valid_indices])

    schema = TypedOptionSchema(enums=(EnumOptionField("taxon_id", tuple(classes)),))
    typed_options = schema.encode(
        [type("TypedOption", (), {"attributes": {"taxon_id": value}})() for value in classes]
    )
    train_options = typed_options.unsqueeze(0).expand(len(train_x), -1, -1).contiguous()
    valid_options = typed_options.unsqueeze(0).expand(len(valid_x), -1, -1).contiguous()
    train_mask = torch.ones(train_options.shape[:2], dtype=torch.bool)
    valid_mask = torch.ones(valid_options.shape[:2], dtype=torch.bool)

    model = NativeFeatureDecisionScorer(768, schema.feature_width, hidden_width=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    config = {
        "task": "fixed eight-taxon iNaturalist image classification",
        "seed": 17,
        "device": "cpu",
        "epochs": 10,
        "batch_size": 64,
        "architecture": "NativeFeatureDecisionScorer(feature=768, typed_options=8, hidden=64)",
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "selection": "none; fixed 10 epochs; final checkpoint only",
        "normalization": "train mean and population std, std floor 1e-6",
        "option_schema": {"taxon_id": classes},
        "train_count": len(train_indices),
        "validation_count": len(valid_indices),
        "features_sha256": hashes,
        "prior_validation_exposure": (
            "same 430 IDs used for prior development selector; this is a post-selection "
            "diagnostic, not blind evidence"
        ),
        "general_quality_claim": False,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    config_hash = sha256(args.output / "config.json")
    history = []
    started = time.perf_counter()
    for epoch in range(1, 11):
        model.train()
        order = torch.randperm(len(train_x))
        losses = []
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            logits = model(train_x[batch], train_options[batch], train_mask[batch])
            loss = F.cross_entropy(logits, train_y[batch])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise ValueError("nonfinite training gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            train_logits = torch.cat(
                [
                    model(train_x[i : i + 128], train_options[i : i + 128], train_mask[i : i + 128])
                    for i in range(0, len(train_x), 128)
                ]
            )
            valid_logits = torch.cat(
                [
                    model(valid_x[i : i + 128], valid_options[i : i + 128], valid_mask[i : i + 128])
                    for i in range(0, len(valid_x), 128)
                ]
            )
        history.append(
            {
                "epoch": epoch,
                "updates": epoch * ((len(train_x) + 63) // 64),
                "mean_batch_train_ce": float(np.mean(losses)),
                "train": metric_summary(train_logits, train_y, classes),
                "validation": metric_summary(valid_logits, valid_y, classes),
                "best_step_selection": False,
            }
        )
    elapsed = time.perf_counter() - started

    checkpoint_path = args.output / "final.pt"
    torch.save({"state_dict": model.state_dict(), "mean": mean, "scale": scale}, checkpoint_path)
    model.eval()
    with torch.inference_mode():
        final_logits = torch.cat(
            [
                model(valid_x[i : i + 128], valid_options[i : i + 128], valid_mask[i : i + 128])
                for i in range(0, len(valid_x), 128)
            ]
        )
    final_metrics = metric_summary(final_logits, valid_y, classes)
    prediction_rows = []
    probabilities = torch.softmax(final_logits.float(), dim=-1).tolist()
    for local_index, row_index in enumerate(valid_indices):
        row = rows[row_index]
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "observation_id": row["observation_id"],
                "photo_id": row["photo_id"],
                "observer_group_sha256": row["observer_group_sha256"],
                "target_taxon_id": str(row["taxon_id"]),
                "probabilities_by_taxon_id": dict(
                    zip(classes, probabilities[local_index], strict=True)
                ),
            }
        )
    predictions_path = args.output / "validation-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    history_path = args.output / "history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    result = {
        "status": "PASS_FIXED_BUDGET_NATIVE_FEATURE_DIAGNOSTIC",
        "config_sha256": config_hash,
        "checkpoint_sha256": sha256(checkpoint_path),
        "history_sha256": sha256(history_path),
        "predictions_sha256": sha256(predictions_path),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "encoder_loaded_or_updated": False,
        "sealed_audit_loaded": False,
        "validation_used_for_selection_or_stopping": False,
        "wall_seconds": elapsed,
        "selected_epoch": None,
        "final_epoch": 10,
        "metrics": final_metrics,
        "limitation": (
            "post-selection development validation for a closed eight-species image task; "
            "not general or blind Decision quality"
        ),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
