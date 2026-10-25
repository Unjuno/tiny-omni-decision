"""Postflight-check a completed fixed iNaturalist readout run without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for import_root in (str(ROOT / "src"), str(ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from scripts.train_inaturalist_frozen_image_readout import (  # noqa: E402
    ENCODER_CHECKPOINT_SHA256,
    ENCODER_PARAMETERS,
    OPTION_ORDER_FLOAT32_TOLERANCE,
    SEED,
    _evaluate,
    _jsonl,
    _load_split,
    _macro_metrics,
    _metrics,
    _option_order_is_stable,
    _option_order_max_delta,
    _per_class_metrics,
    _predictions,
    _sha256,
)
from tiny_omni_decision.textfree_image import TextFreeImageReadout  # noqa: E402


def _read_state(path: Path, taxonomy: str, class_ids: list[int]):
    model = TextFreeImageReadout(768, {taxonomy: tuple(str(value) for value in class_ids)})
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def _legacy_option_order_max_delta(model, taxonomy: str, features, class_ids: list[int]) -> float:
    normalized_class_ids = [str(value) for value in class_ids]
    options = [
        {"option_id": value, "class_id": value} for value in normalized_class_ids
    ]
    maximum = 0.0
    with torch.inference_mode():
        for feature in features[:32]:
            logits = model(feature, taxonomy)
            forward_indexes = torch.tensor(
                [normalized_class_ids.index(option["class_id"]) for option in options]
            )
            reverse_options = list(reversed(options))
            reverse_indexes = torch.tensor(
                [normalized_class_ids.index(option["class_id"]) for option in reverse_options]
            )
            forward_probabilities = torch.softmax(logits[forward_indexes].float(), dim=-1)
            reverse_probabilities = torch.softmax(logits[reverse_indexes].float(), dim=-1)
            reverse_by_id = {
                option["class_id"]: float(probability)
                for option, probability in zip(
                    reverse_options, reverse_probabilities, strict=True
                )
            }
            maximum = max(
                maximum,
                max(
                    abs(float(probability) - reverse_by_id[option["class_id"]])
                    for option, probability in zip(
                        options, forward_probabilities, strict=True
                    )
                ),
            )
    return maximum


def verify(
    feature_dir: Path,
    output_dir: Path,
    verification_path: Path,
    original_option_tolerance: float,
) -> dict[str, Any]:
    torch.set_num_threads(4)
    feature_dir = feature_dir.resolve(strict=True)
    output_dir = output_dir.resolve(strict=True)
    verification_path = verification_path.resolve()
    if verification_path.exists():
        raise FileExistsError(f"refusing to overwrite postflight result: {verification_path}")
    feature_config = json.loads((feature_dir / "config.json").read_text(encoding="utf-8"))
    if (
        feature_config.get("status") != "completed"
        or feature_config.get("sealed_audit_loaded") is not False
        or feature_config.get("checkpoint_sha256") != ENCODER_CHECKPOINT_SHA256
    ):
        raise ValueError("feature cache is incomplete, unpinned, or includes sealed audit")

    taxonomy = "inaturalist_bird_species"
    train_rows, x_train, y_train, class_ids = _load_split(feature_dir, "train")
    val_rows, x_val, y_val, val_class_ids = _load_split(feature_dir, "validation", class_ids)
    if class_ids != val_class_ids:
        raise ValueError("train and validation class order differs")

    history = [
        json.loads(line)
        for line in (output_dir / "training-history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    history_epochs = [row.get("epoch") for row in history]
    if not history_epochs or history_epochs != list(range(1, len(history_epochs) + 1)):
        raise ValueError("training history epochs must be complete and start at one")
    checkpoint = json.loads((output_dir / "best-checkpoint.json").read_text(encoding="utf-8"))
    expected_best = min(
        history,
        key=lambda row: float(row["validation_metrics"]["macro"]["nll"]),
    )
    if checkpoint.get("epoch") != expected_best["epoch"]:
        raise ValueError("saved best checkpoint does not match the fixed selector")

    class_names_by_id: dict[int, str] = {}
    for row in train_rows + val_rows:
        class_id = int(row["taxon_id"])
        name = str(row["label"])
        if class_id in class_names_by_id and class_names_by_id[class_id] != name:
            raise ValueError("one source taxon ID has conflicting labels")
        class_names_by_id[class_id] = name
    class_names = [class_names_by_id[value] for value in class_ids]

    best = _read_state(output_dir / "best-readout.pt", taxonomy, class_ids)
    final = _read_state(output_dir / "final-readout.pt", taxonomy, class_ids)
    with torch.inference_mode():
        best_logits = best(x_val, taxonomy)
        zero_logits = best(torch.zeros_like(x_val), taxonomy)
        generator = torch.Generator().manual_seed(SEED + 1000)
        permutation = torch.randperm(len(x_val), generator=generator)
        shuffled_logits = best(x_val[permutation], taxonomy)
        reloaded = _read_state(output_dir / "best-readout.pt", taxonomy, class_ids)
        reloaded_logits = reloaded(x_val, taxonomy)

    reload_delta = float((best_logits - reloaded_logits).abs().max())
    option_delta = _option_order_max_delta(best, taxonomy, x_val, class_ids)
    legacy_option_delta = _legacy_option_order_max_delta(best, taxonomy, x_val, class_ids)
    if reload_delta != 0.0 or not _option_order_is_stable(option_delta):
        raise AssertionError("postflight reload or option-order stability check failed")
    if legacy_option_delta <= original_option_tolerance:
        raise ValueError("the recorded historical runner failure was not reproduced")

    predicted_rows = _predictions(val_rows, best_logits, class_ids, class_names)
    saved_rows = _jsonl(output_dir / "validation-predictions.jsonl")
    if len(predicted_rows) != len(saved_rows):
        raise ValueError("saved validation prediction count differs from the fixed split")
    maximum_saved_probability_delta = 0.0
    for expected, saved in zip(predicted_rows, saved_rows, strict=True):
        identity_keys = (
            "sample_id",
            "observation_id",
            "photo_id",
            "media_sha256",
            "target_class_id",
            "target_label",
            "prediction_class_id",
            "prediction_label",
        )
        for key in identity_keys:
            if expected[key] != saved[key]:
                raise ValueError(f"saved validation prediction differs in {key}")
        for expected_option, saved_option in zip(
            expected["probabilities"], saved["probabilities"], strict=True
        ):
            if any(
                expected_option[key] != saved_option[key]
                for key in ("class_id", "class_label")
            ):
                raise ValueError("saved option identity/order differs from the fixed taxonomy")
            maximum_saved_probability_delta = max(
                maximum_saved_probability_delta,
                abs(expected_option["probability"] - saved_option["probability"]),
            )
    if maximum_saved_probability_delta > OPTION_ORDER_FLOAT32_TOLERANCE:
        raise ValueError("saved probabilities differ beyond the float32 tolerance")

    per_class = _per_class_metrics(best_logits, y_val, class_ids)
    majority = int(torch.bincount(y_train, minlength=len(class_ids)).argmax())
    uniform_logits = torch.zeros((len(y_val), len(class_ids)), dtype=torch.float32)
    majority_logits = torch.full_like(uniform_logits, -20.0)
    majority_logits[:, majority] = 20.0
    controls = {
        "aligned_image": {
            "overall": _metrics(best_logits, y_val),
            "macro": _macro_metrics(per_class),
        },
        "zero_image": {
            "overall": _metrics(zero_logits, y_val),
            "macro": _macro_metrics(_per_class_metrics(zero_logits, y_val, class_ids)),
            "top1_agreement_with_aligned": float(
                (zero_logits.argmax(-1) == best_logits.argmax(-1)).float().mean()
            ),
        },
        "validation_image_permutation": {
            "overall": _metrics(shuffled_logits, y_val),
            "macro": _macro_metrics(_per_class_metrics(shuffled_logits, y_val, class_ids)),
            "top1_agreement_with_aligned": float(
                (shuffled_logits.argmax(-1) == best_logits.argmax(-1)).float().mean()
            ),
            "permutation_seed": SEED + 1000,
        },
        "uniform_option_distribution": {
            "overall": _metrics(uniform_logits, y_val),
            "macro": _macro_metrics(_per_class_metrics(uniform_logits, y_val, class_ids)),
        },
        "training_majority_class": {
            "overall": _metrics(majority_logits, y_val),
            "macro": _macro_metrics(_per_class_metrics(majority_logits, y_val, class_ids)),
            "predicted_class_id": class_ids[majority],
        },
    }
    final_metrics = _evaluate(final, taxonomy, x_val, y_val, class_ids)
    curve_epochs = sorted(
        {
            1,
            16,
            64,
            128,
            256,
            384,
            512,
            1024,
            1276,
            1536,
            2048,
            len(history),
            int(checkpoint["epoch"]),
        }
    )
    curve = [
        {
            "epoch": row["epoch"],
            "train_accuracy": row["train_metrics"]["overall"]["accuracy"],
            "validation_accuracy": row["validation_metrics"]["overall"]["accuracy"],
            "validation_macro_accuracy": row["validation_metrics"]["macro"]["accuracy"],
            "validation_macro_nll": row["validation_metrics"]["macro"]["nll"],
        }
        for row in history
        if row["epoch"] in curve_epochs
    ]
    artifacts = [
        "best-readout.pt",
        "final-readout.pt",
        "best-checkpoint.json",
        "training-history.jsonl",
        "validation-predictions.jsonl",
    ]
    result = {
        "schema_version": 1,
        "verification_status": "passed_posthoc_after_runner_exit_1",
        "original_runner_exit_code": 1,
        "original_failure": (
            "pre-fix option-order softmax reduction differed beyond the runner threshold"
        ),
        "original_option_order_tolerance": original_option_tolerance,
        "pre_fix_option_order_delta_first_32": legacy_option_delta,
        "corrected_tolerance": OPTION_ORDER_FLOAT32_TOLERANCE,
        "verification_utc": datetime.now(UTC).isoformat(),
        "training_epochs_completed": len(history),
        "training_epochs_configured": len(history),
        "training_sample_count": len(train_rows),
        "validation_sample_count": len(val_rows),
        "unique_train_examples": len(train_rows),
        "train_exposures": len(train_rows) * len(history),
        "repeated_train_exposures": len(train_rows) * (len(history) - 1),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_nll": float(checkpoint["macro_validation_nll"]),
        "selected_validation_metrics": {
            "overall": controls["aligned_image"]["overall"],
            "macro": controls["aligned_image"]["macro"],
            "per_class": per_class,
        },
        "final_epoch_validation_metrics": {
            key: value for key, value in final_metrics.items() if key != "logits"
        },
        "input_controls": controls,
        "learning_curve": curve,
        "best_reload_max_logit_delta": reload_delta,
        "option_order_max_probability_delta_first_32": option_delta,
        "saved_prediction_max_probability_delta": maximum_saved_probability_delta,
        "trainable_readout_parameters": 768 * len(class_ids) + len(class_ids),
        "frozen_encoder_parameters": ENCODER_PARAMETERS,
        "total_unquantized_parameters": ENCODER_PARAMETERS + 768 * len(class_ids) + len(class_ids),
        "encoder_checkpoint_sha256": ENCODER_CHECKPOINT_SHA256,
        "encoder_checkpoint_bytes": 457_475_296,
        "sealed_audit_loaded": False,
        "evaluation_split": "development validation only; not a sealed audit",
        "task_scope": (
            "closed-set eight-species image classification; not open-vocabulary or "
            "multimodal decisions"
        ),
        "source_commit_of_training_code": "f1fabbd5179c18bea032ecc3288c0479c19d29c8",
        "training_wall_seconds": None,
        "evaluation_wall_seconds": None,
        "timing_note": (
            "The runner failed before it serialized timing metadata; exact durations "
            "are unavailable."
        ),
        "artifact_sha256": {name: _sha256(output_dir / name) for name in artifacts},
        "feature_config_sha256": _sha256(feature_dir / "config.json"),
        "feature_sha256": {
            split: _sha256(feature_dir / f"{split}-features.npz")
            for split in ("train", "validation")
        },
    }
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    with verification_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verification-path", type=Path)
    parser.add_argument(
        "--original-option-tolerance",
        type=float,
        choices=(1e-7, 2e-7),
        required=True,
        help="the tolerance used by this run's original trainer",
    )
    args = parser.parse_args()
    verification_path = args.verification_path or args.output_dir / "postflight-verification.json"
    print(
        json.dumps(
            verify(
                args.feature_dir,
                args.output_dir,
                verification_path,
                args.original_option_tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
