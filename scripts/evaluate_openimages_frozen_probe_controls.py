"""Evaluate fixed image-feature sensitivity controls for the Open Images probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PERMUTATION_SEED = 20261026


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_derangement(count: int, seed: int = PERMUTATION_SEED) -> np.ndarray:
    """Return a deterministic permutation with no row receiving its own image."""
    if count < 2:
        raise ValueError("at least two validation rows are required for image shuffling")
    order = np.random.default_rng(seed).permutation(count)
    donor_indices = np.empty(count, dtype=np.int64)
    donor_indices[order] = np.roll(order, 1)
    if np.any(donor_indices == np.arange(count)) or len(np.unique(donor_indices)) != count:
        raise RuntimeError("failed to construct a fixed image derangement")
    return donor_indices


def _metrics_by_label(
    rows: list[dict[str, Any]], logits: Any, targets: Any
) -> dict[str, dict[str, Any]]:
    from scripts.train_openimages_frozen_probe import _quality

    result = {}
    for label in sorted({str(row["target_label"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row["target_label"]) == label]
        result[label] = _quality(logits[indices], targets[indices])
    return result


def _paired_summary(
    reference: np.ndarray, comparison: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    reference_pred = reference.argmax(axis=-1)
    comparison_pred = comparison.argmax(axis=-1)
    reference_correct = reference_pred == targets
    comparison_correct = comparison_pred == targets
    row_indices = np.arange(len(targets))
    reference_nll = -np.log(np.clip(reference[row_indices, targets], 1e-12, 1.0))
    comparison_nll = -np.log(np.clip(comparison[row_indices, targets], 1e-12, 1.0))
    one_hot = np.eye(reference.shape[-1], dtype=np.float32)[targets]
    reference_brier = ((reference - one_hot) ** 2).sum(axis=-1)
    comparison_brier = ((comparison - one_hot) ** 2).sum(axis=-1)
    return {
        "count": int(len(targets)),
        "prediction_agreement_count": int(np.sum(reference_pred == comparison_pred)),
        "reference_correct_comparison_wrong": int(np.sum(reference_correct & ~comparison_correct)),
        "comparison_correct_reference_wrong": int(np.sum(comparison_correct & ~reference_correct)),
        "reference_minus_comparison_accuracy": float(
            reference_correct.mean() - comparison_correct.mean()
        ),
        "reference_minus_comparison_nll": float(np.mean(reference_nll - comparison_nll)),
        "reference_minus_comparison_brier": float(np.mean(reference_brier - comparison_brier)),
        "mean_absolute_probability_delta": float(np.mean(np.abs(reference - comparison))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrained_reuse/path_a_openimages_videoprism.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    features_dir = args.features_dir.resolve(strict=True)
    run_dir = args.run_dir.resolve(strict=True)
    text_model = args.text_model.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("control artifacts must be stored outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite control output: {output}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from transformers import AutoModel, AutoTokenizer

    from scripts.train_openimages_frozen_probe import (
        IMAGE_SIZE,
        TEXT_SIZE,
        TRAINABLE_PARAMETERS,
        _load_features,
        _quality,
        _tasks,
        _text_features,
        _validate_rows,
    )
    from scripts.train_openimages_frozen_probe import (
        _sha256 as runner_sha256,
    )
    from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

    metadata_path = run_dir / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_config_path = features_dir / "config.json"
    feature_config = json.loads(feature_config_path.read_text(encoding="utf-8"))
    if metadata.get("experiment_id") != "path-a-openimages-videoprism-minilm-seed17":
        raise ValueError("unexpected source run")
    if metadata.get("sealed_audit_loaded") is not False:
        raise ValueError("source run does not certify sealed-audit isolation")
    if _sha256(config_path) != metadata["config_sha256"]:
        raise ValueError("provided config differs from the selected training run")
    if _sha256(feature_config_path) != metadata["feature_config_sha256"]:
        raise ValueError("feature config differs from the selected training run")
    if _sha256(run_dir / "executed-runner.py") != metadata["runner_sha256"]:
        raise ValueError("preserved training runner differs from the recorded run")
    if _sha256(run_dir / "best-readout.pt") != metadata["artifact_sha256"]["best-readout.pt"]:
        raise ValueError("best readout hash differs from the training receipt")
    if (
        _sha256(run_dir / "validation-predictions.jsonl")
        != metadata["artifact_sha256"]["validation-predictions.jsonl"]
    ):
        raise ValueError("saved validation prediction hash differs from the training receipt")
    for split in ("train", "validation"):
        feature_name = f"{split}-features.npz"
        rows_name = f"{split}-rows.jsonl"
        if _sha256(features_dir / feature_name) != metadata["feature_hashes"][split]:
            raise ValueError(f"{split} feature hash differs from the training receipt")
        if _sha256(features_dir / feature_name) != feature_config["output_hashes"][feature_name]:
            raise ValueError(f"{split} feature hash differs from the extraction receipt")
        if _sha256(features_dir / rows_name) != feature_config["output_hashes"][rows_name]:
            raise ValueError(f"{split} row hash differs from the extraction receipt")
    if _sha256(text_model / "model.safetensors") != metadata["text_encoder"]["weights_sha256"]:
        raise ValueError("MiniLM weights differ from the trained scorer input")

    torch.set_num_threads(4)
    torch.manual_seed(17)
    started = time.perf_counter()
    rows, images = _load_features(features_dir, "validation")
    _validate_rows(_load_features(features_dir, "train")[0], rows)
    id_digest = hashlib.sha256(
        "\n".join(str(row["sample_id"]) for row in rows).encode()
    ).hexdigest()
    if id_digest != metadata["split_ids_sha256"]["validation"]:
        raise ValueError("validation sample order differs from the selected run")

    tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=True)
    encoder = AutoModel.from_pretrained(text_model, local_files_only=True).eval()
    encoder.requires_grad_(False)
    text = _text_features(rows, tokenizer, encoder)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("MiniLM unexpectedly has trainable parameters")
    del encoder, tokenizer

    scorer = FrozenFeatureCandidateScorer(IMAGE_SIZE + TEXT_SIZE, hidden_size=128).cpu().eval()
    if sum(parameter.numel() for parameter in scorer.parameters()) != TRAINABLE_PARAMETERS:
        raise ValueError("unexpected scorer parameter count")
    state = torch.load(run_dir / "best-readout.pt", map_location="cpu", weights_only=True)
    scorer.load_state_dict(state, strict=True)

    donor = fixed_derangement(len(rows))
    conditions = {
        "aligned_image": images,
        "zero_image": np.zeros_like(images),
        "deranged_validation_image": images[donor],
    }
    targets = torch.tensor([row["options"].index(row["target"]) for row in rows], dtype=torch.long)
    prediction_path = run_dir / "validation-predictions.jsonl"
    saved_predictions = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if [row["sample_id"] for row in saved_predictions] != [row["sample_id"] for row in rows]:
        raise ValueError("saved validation predictions do not align with fixed feature rows")

    condition_results: dict[str, Any] = {}
    probabilities_by_condition: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, condition_images in conditions.items():
            x, _ = _tasks(condition_images, text, rows)
            logits = scorer(x).float()
            probabilities = torch.softmax(logits, dim=-1)
            probabilities_by_condition[name] = probabilities
            condition_results[name] = {
                "metrics": _quality(logits, targets),
                "metrics_by_target_label": _metrics_by_label(rows, logits, targets),
            }

    saved_probs = torch.tensor(
        [row["probabilities"] for row in saved_predictions], dtype=torch.float32
    )
    replay_max_probability_delta = float(
        (probabilities_by_condition["aligned_image"] - saved_probs).abs().max()
    )
    if replay_max_probability_delta > 1e-6:
        raise ValueError(
            f"fixed aligned replay differs from saved output by {replay_max_probability_delta}"
        )

    paired_comparisons = {
        name: _paired_summary(
            probabilities_by_condition["aligned_image"].numpy(),
            probabilities.numpy(),
            targets.numpy(),
        )
        for name, probabilities in probabilities_by_condition.items()
        if name != "aligned_image"
    }

    input_rows = []
    for index, row in enumerate(rows):
        item: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "image_id": row["image_id"],
            "target_label": row["target_label"],
            "target": row["target"],
            "options": row["options"],
            "target_position": int(targets[index]),
        }
        for name, probabilities in probabilities_by_condition.items():
            values = probabilities[index].tolist()
            item[name] = {
                "probabilities": values,
                "prediction": row["options"][int(np.argmax(values))],
                "correct": bool(int(np.argmax(values)) == int(targets[index])),
            }
        donor_index = int(donor[index])
        item["deranged_validation_image_donor"] = {
            "sample_id": rows[donor_index]["sample_id"],
            "image_id": rows[donor_index]["image_id"],
            "target_label": rows[donor_index]["target_label"],
        }
        input_rows.append(item)

    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "control-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in input_rows), encoding="utf-8"
    )
    result = {
        "schema_version": 1,
        "experiment_id": "path-a-openimages-videoprism-input-controls-seed17",
        "source_run_dir": str(run_dir),
        "source_run_metadata_sha256": runner_sha256(metadata_path),
        "source_best_readout_sha256": _sha256(run_dir / "best-readout.pt"),
        "validation_features_sha256": _sha256(features_dir / "validation-features.npz"),
        "validation_sample_id_order_sha256": id_digest,
        "text_weights_sha256": _sha256(text_model / "model.safetensors"),
        "config_sha256": _sha256(config_path),
        "scorer_parameters": TRAINABLE_PARAMETERS,
        "device": "CPU",
        "sealed_audit_loaded": False,
        "training_performed": False,
        "permutation_seed": PERMUTATION_SEED,
        "derangement_fixed_points": int(np.sum(donor == np.arange(len(rows)))),
        "conditions": condition_results,
        "paired_comparisons_aligned_vs_control": paired_comparisons,
        "aligned_replay_max_probability_delta_vs_saved_predictions": replay_max_probability_delta,
        "metric_interpretation": (
            "fixed validation sensitivity diagnostic; validation was used for checkpoint "
            "selection and is not blind"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "prediction_file_sha256": _sha256(predictions_path),
        "evaluator_sha256": _sha256(Path(__file__)),
        "completed_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (output / "control-metadata.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
