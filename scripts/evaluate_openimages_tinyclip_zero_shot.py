"""Evaluate one fixed TinyCLIP zero-shot prompt on the frozen Open Images pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_REVISION = "07a4b0bc751cb64fecd2b661c048c1dd98d69444"
EXPECTED_MODEL_SHA256 = "112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587"
PROMPT_TEMPLATE = "a photo containing {label}."
ECE_BINS = 15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def validate_rows(
    task_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    media_rows: list[dict[str, Any]],
    *,
    split: str = "validation",
) -> None:
    if not task_rows or len(task_rows) != len(feature_rows) or len(task_rows) != len(media_rows):
        raise ValueError("task, feature, and media row counts must match and be nonzero")
    seen: set[str] = set()
    seen_samples: set[str] = set()
    for task, feature, media in zip(task_rows, feature_rows, media_rows, strict=True):
        image_id = str(task.get("image_id", ""))
        if not image_id or image_id in seen:
            raise ValueError("image IDs must be nonempty and unique")
        seen.add(image_id)
        sample_id = str(task.get("sample_id", ""))
        if not sample_id or sample_id in seen_samples:
            raise ValueError("sample IDs must be nonempty and unique")
        seen_samples.add(sample_id)
        if (
            task.get("split") != split
            or feature.get("split") != split
            or media.get("split") != split
        ):
            raise ValueError("all rows must belong to the requested split")
        for row in (task, feature, media):
            if row.get("image_id") != image_id:
                raise ValueError(
                    "image order/identity differs across task, feature, and media rows"
                )
        if task.get("sample_id") != feature.get("sample_id") or task.get("sample_id") != media.get(
            "sample_id"
        ):
            raise ValueError("sample order/identity differs across inputs")
        if task.get("target") not in task.get("options", []):
            raise ValueError("target must be one of the supplied options")
        options = task.get("options")
        if not isinstance(options, list) or len(options) != 4 or len(set(options)) != 4:
            raise ValueError("each task must have exactly four unique options")
        if task.get("target") != feature.get("target") or task.get("target") != media.get("target"):
            raise ValueError("target differs across task, feature, and media rows")
        if feature.get("media_sha256") != media.get("media_sha256"):
            raise ValueError("media SHA-256 differs between feature and media inputs")
        if task.get("media_sha256") is not None and task.get("media_sha256") != feature.get(
            "media_sha256"
        ):
            raise ValueError("media SHA-256 differs from the task manifest")
        if (
            media.get("status") != "saved"
            or media.get("page_license_and_attribution_match") is not True
        ):
            raise ValueError("only saved media with a matching page declaration may be evaluated")
        if media.get("individual_rights_verified") is not False:
            raise ValueError("rights-status field must remain explicit and unverified")
        if task.get("page_license_and_attribution_match") is False:
            raise ValueError("task failed the page-declaration gate")


def validate_task_media_rows(
    task_rows: list[dict[str, Any]], media_rows: list[dict[str, Any]]
) -> None:
    """Validate an image-only task/media pair without requiring VideoPrism features."""
    if not task_rows or len(task_rows) != len(media_rows):
        raise ValueError("task and media row counts must match and be nonzero")
    seen_images: set[str] = set()
    seen_samples: set[str] = set()
    for task, media in zip(task_rows, media_rows, strict=True):
        image_id = str(task.get("image_id", ""))
        sample_id = str(task.get("sample_id", ""))
        if not image_id or image_id in seen_images:
            raise ValueError("image IDs must be nonempty and unique")
        if not sample_id or sample_id in seen_samples:
            raise ValueError("sample IDs must be nonempty and unique")
        seen_images.add(image_id)
        seen_samples.add(sample_id)
        if task.get("split") != "validation" or media.get("split") != "validation":
            raise ValueError("all rows must belong to validation")
        if media.get("image_id") != image_id or media.get("sample_id") != sample_id:
            raise ValueError("task/media row identity mismatch")
        options = task.get("options")
        if not isinstance(options, list) or len(options) != 4 or len(set(options)) != 4:
            raise ValueError("each task must have exactly four unique options")
        if task.get("target") not in options or media.get("target") != task.get("target"):
            raise ValueError("task/media target mismatch")
        if task.get("media_sha256") is not None and task.get("media_sha256") != media.get(
            "media_sha256"
        ):
            raise ValueError("task/media hash mismatch")
        if (
            media.get("status") != "saved"
            or media.get("page_license_and_attribution_match") is not True
            or media.get("individual_rights_verified") is not False
        ):
            raise ValueError("only saved media with matching page declarations may be evaluated")
        if task.get("page_license_and_attribution_match") is False:
            raise ValueError("task failed the page-declaration gate")


def quality_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if logits.ndim != 2 or logits.shape[1] != 4 or targets.shape != (len(logits),):
        raise ValueError("expected four-choice logits and one target index per row")
    if not np.isfinite(logits).all() or np.any(targets < 0) or np.any(targets >= 4):
        raise ValueError("logits/targets contain invalid values")
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    target_probabilities = np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)
    one_hot = np.eye(4, dtype=np.float64)[targets]
    confidence = probabilities.max(axis=1)
    ece = 0.0
    bin_counts: list[int] = []
    for index in range(ECE_BINS):
        low, high = index / ECE_BINS, (index + 1) / ECE_BINS
        mask = (confidence >= low) & (
            confidence < high if index < ECE_BINS - 1 else confidence <= high
        )
        count = int(mask.sum())
        bin_counts.append(count)
        if count:
            ece += count / len(targets) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {
        "count": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(target_probabilities).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15_bins": float(ece),
        "ece_bin_counts": bin_counts,
        "mean_confidence": float(confidence.mean()),
    }


def _safe_image_path(media_row: dict[str, Any]) -> Path:
    relative = Path(str(media_row["local_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("media local_path must be a safe relative path")
    artifact_root = Path(str(media_row["_artifact_root"])).resolve(strict=True)
    root = (artifact_root / "validation").resolve(strict=True)
    image_path = (root / relative).resolve(strict=True)
    if not image_path.is_relative_to(root) or not image_path.is_file():
        raise ValueError("resolved image must be a file inside the validation media directory")
    if sha256_file(image_path) != media_row.get("media_sha256"):
        raise ValueError(f"media SHA-256 mismatch for {media_row.get('image_id')}")
    return image_path


def score_images(
    model: Any,
    processor: Any,
    text_inputs: Any,
    image_paths: list[Path] | None,
    *,
    row_count: int,
    batch_size: int,
    blank: bool = False,
) -> tuple[np.ndarray, float]:
    import torch
    from PIL import Image

    logits: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, row_count, batch_size):
            batch_count = min(batch_size, row_count - start)
            images = []
            if blank:
                fill = tuple(
                    round(component * 255) for component in processor.image_processor.image_mean
                )
                images = [Image.new("RGB", (224, 224), color=fill) for _ in range(batch_count)]
            else:
                if image_paths is None:
                    raise ValueError("image paths are required unless blank-image mode is selected")
                for path in image_paths[start : start + batch_count]:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
            image_inputs = processor(images=images, return_tensors="pt")
            outputs = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
                pixel_values=image_inputs["pixel_values"],
            )
            logits.extend(outputs.logits_per_image.cpu().numpy())
            for image in images:
                image.close()
    result = np.asarray(logits, dtype=np.float32)
    if result.shape[0] != row_count:
        raise ValueError("TinyCLIP returned an unexpected image count")
    return result, time.perf_counter() - started


def gather_choice_logits(
    all_logits: np.ndarray,
    task_rows: list[dict[str, Any]],
    label_to_index: dict[str, int],
) -> np.ndarray:
    if all_logits.shape[0] != len(task_rows):
        raise ValueError("score rows do not match the frozen task order")
    return np.asarray(
        [
            [all_logits[i, label_to_index[str(option)]] for option in row["options"]]
            for i, row in enumerate(task_rows)
        ],
        dtype=np.float32,
    )


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--media-dir", type=Path, action="append", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tasks-manifest", type=Path, required=True)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-controls", action="store_true")
    parser.add_argument("--fresh-task-id-set", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 32:
        raise ValueError("batch size must be between 1 and 32")
    if args.model_revision != EXPECTED_MODEL_REVISION:
        raise ValueError("this diagnostic is frozen to the audited TinyCLIP revision")
    features_dir = args.features_dir.resolve(strict=True) if args.features_dir else None
    media_dirs = [path.resolve(strict=True) for path in args.media_dir]
    model_dir = args.model_dir.resolve(strict=True)
    tasks_path = args.tasks_manifest.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("model outputs and image identifiers must remain outside the repository")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    model_weights = model_dir / "model.safetensors"
    if sha256_file(model_weights) != EXPECTED_MODEL_SHA256:
        raise ValueError("TinyCLIP weights do not match the audited checkpoint hash")

    media_by_id: dict[str, dict[str, Any]] = {}
    media_manifest_hashes: dict[str, str] = {}
    for media_dir in media_dirs:
        manifest_path = media_dir / "validation-media-manifest.jsonl"
        media_manifest_hashes[str(manifest_path)] = sha256_file(manifest_path)
        for media_row in read_jsonl(manifest_path):
            image_id = str(media_row.get("image_id", ""))
            if not image_id:
                raise ValueError("media manifests must identify every image")
            previous = media_by_id.get(image_id)
            if (
                previous is not None
                and previous.get("media_sha256") is not None
                and media_row.get("media_sha256") is not None
                and previous.get("media_sha256") != media_row.get("media_sha256")
            ):
                raise ValueError("retried media manifests disagree on image content hash")
            media_by_id[image_id] = media_row | {"_artifact_root": str(media_dir)}
    task_manifest = json.loads(tasks_path.read_text(encoding="utf-8"))
    all_task_rows = task_manifest["examples"]
    tasks_by_id = {str(row["image_id"]): row for row in all_task_rows}
    if len(tasks_by_id) != len(all_task_rows):
        raise ValueError("frozen task manifest must contain unique image IDs")
    feature_rows: list[dict[str, Any]] | None = None
    features_path: Path | None = None
    feature_image_ids_in_rows: list[str] | None = None
    if features_dir is not None:
        feature_rows = read_jsonl(features_dir / "validation-rows.jsonl")
        feature_image_ids_in_rows = [str(row["image_id"]) for row in feature_rows]
        if not set(feature_image_ids_in_rows).issubset(tasks_by_id):
            raise ValueError("a feature row is absent from the frozen task manifest")
        task_rows = [tasks_by_id[image_id] for image_id in feature_image_ids_in_rows]
        media_rows = [media_by_id.get(image_id) for image_id in feature_image_ids_in_rows]
        if any(row is None for row in media_rows):
            raise ValueError("a validation feature is missing its source media manifest row")
        features_path = features_dir / "validation-features.npz"
        with np.load(features_path, allow_pickle=False) as stored:
            feature_sample_ids = stored["sample_ids"].astype(str).tolist()
            feature_image_ids = stored["image_ids"].astype(str).tolist()
            feature_hashes = stored["media_sha256"].astype(str).tolist()
            if stored["features"].shape != (len(feature_rows), 768):
                raise ValueError("unexpected VideoPrism feature matrix shape")
        validate_rows(task_rows, feature_rows, media_rows)
        if [row["sample_id"] for row in task_rows] != feature_sample_ids:
            raise ValueError("frozen task order differs from feature sample order")
        if [row["image_id"] for row in task_rows] != feature_image_ids:
            raise ValueError("frozen task image order differs from feature image order")
        if [row["media_sha256"] for row in media_rows] != feature_hashes:
            raise ValueError("frozen media-manifest hashes differ from feature media hashes")
    else:
        task_rows = [
            row
            for row in all_task_rows
            if str(row["image_id"]) in media_by_id
            and media_by_id[str(row["image_id"])].get("status") == "saved"
            and media_by_id[str(row["image_id"])].get("page_license_and_attribution_match") is True
        ]
        media_rows = [media_by_id[str(row["image_id"])] for row in task_rows]
        validate_task_media_rows(task_rows, media_rows)
    excluded_task_count = len(all_task_rows) - len(task_rows)

    import torch
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModelForZeroShotImageClassification.from_pretrained(model_dir, local_files_only=True)
        .cpu()
        .eval()
    )
    model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 83_101_185:
        raise ValueError(f"unexpected TinyCLIP runtime parameter count: {parameter_count}")

    labels = sorted({str(option) for row in task_rows for option in row["options"]})
    label_to_index = {label: index for index, label in enumerate(labels)}
    prompts = [PROMPT_TEMPLATE.format(label=label.lower()) for label in labels]
    text_inputs = processor(text=prompts, padding=True, truncation=True, return_tensors="pt")
    image_paths = [_safe_image_path(row) for row in media_rows]
    all_logits, elapsed_seconds = score_images(
        model,
        processor,
        text_inputs,
        image_paths,
        row_count=len(task_rows),
        batch_size=args.batch_size,
    )
    if all_logits.shape != (len(task_rows), len(labels)):
        raise ValueError("TinyCLIP returned an unexpected score matrix shape")
    choice_logits = gather_choice_logits(all_logits, task_rows, label_to_index)
    target_indices = np.asarray(
        [row["options"].index(row["target"]) for row in task_rows], dtype=np.int64
    )
    overall = quality_metrics(choice_logits, target_indices)
    by_target: dict[str, Any] = {}
    for target_label in sorted({str(row["target_label"]) for row in task_rows}):
        indices = np.asarray(
            [i for i, row in enumerate(task_rows) if str(row["target_label"]) == target_label],
            dtype=np.int64,
        )
        by_target[target_label] = quality_metrics(choice_logits[indices], target_indices[indices])
    controls: dict[str, Any] = {}
    control_prediction_rows: list[dict[str, Any]] = []
    controls_elapsed_seconds = 0.0
    if args.image_controls:
        permutation = list(range(len(image_paths)))
        rng = random.Random(20261026)
        for _ in range(1000):
            rng.shuffle(permutation)
            if all(index != donor for index, donor in enumerate(permutation)):
                break
        else:
            raise RuntimeError("failed to create deterministic image derangement")
        deranged_scores, deranged_seconds = score_images(
            model,
            processor,
            text_inputs,
            [image_paths[index] for index in permutation],
            row_count=len(task_rows),
            batch_size=args.batch_size,
        )
        zero_scores, zero_seconds = score_images(
            model,
            processor,
            text_inputs,
            None,
            row_count=len(task_rows),
            batch_size=args.batch_size,
            blank=True,
        )
        controls_elapsed_seconds = deranged_seconds + zero_seconds
        for condition, scores in (
            ("zero_image", zero_scores),
            ("deranged_image", deranged_scores),
        ):
            condition_logits = gather_choice_logits(scores, task_rows, label_to_index)
            condition_probs = probabilities_from_logits(condition_logits)
            aligned_probs = probabilities_from_logits(choice_logits)
            controls[condition] = {
                "metrics": quality_metrics(condition_logits, target_indices),
                "top1_agreement_with_aligned": float(
                    np.mean(condition_probs.argmax(axis=1) == aligned_probs.argmax(axis=1))
                ),
                "correctness_changes_vs_aligned": int(
                    np.count_nonzero(
                        (condition_probs.argmax(axis=1) == target_indices)
                        != (aligned_probs.argmax(axis=1) == target_indices)
                    )
                ),
                "mean_absolute_probability_change_vs_aligned": float(
                    np.abs(condition_probs - aligned_probs).mean()
                ),
            }
            for index, row in enumerate(task_rows):
                pred_index = int(condition_probs[index].argmax())
                control_prediction_rows.append(
                    {
                        "condition": condition,
                        "sample_id": row["sample_id"],
                        "prediction": row["options"][pred_index],
                        "probabilities": condition_probs[index].tolist(),
                    }
                )
    predictions = []
    for index, row in enumerate(task_rows):
        probs = np.exp(choice_logits[index] - choice_logits[index].max())
        probs /= probs.sum()
        pred_index = int(probs.argmax())
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "target_label": row["target_label"],
                "options": row["options"],
                "target": row["target"],
                "prediction": row["options"][pred_index],
                "probabilities": probs.tolist(),
            }
        )
    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    control_predictions_path: Path | None = None
    if control_prediction_rows:
        control_predictions_path = output_dir / "control-predictions.jsonl"
        control_predictions_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in control_prediction_rows
            ),
            encoding="utf-8",
        )
    metadata = {
        "schema_version": 1,
        "model": "TinyCLIP-ViT-39M-16-Text-19M-YFCC15M",
        "model_revision": args.model_revision,
        "model_license_declared": "MIT",
        "model_weights_sha256": EXPECTED_MODEL_SHA256,
        "runtime_parameters": parameter_count,
        "checkpoint_bytes": model_weights.stat().st_size,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_search_or_calibration": False,
        "image_controls_enabled": args.image_controls,
        "image_derangement_seed": 20261026 if args.image_controls else None,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "validation_previously_exposed": not args.fresh_task_id_set,
        "validation_task_ids_previously_scored": not args.fresh_task_id_set,
        "validation_source_support_aggregates_previously_inspected": True,
        "sealed_audit_loaded": False,
        "split": "development_validation",
        "count": len(task_rows),
        "frozen_manifest_count": len(all_task_rows),
        "excluded_unavailable_media_count": excluded_task_count,
        "feature_cache_sha256": sha256_file(features_path) if features_path else None,
        "feature_rows_sha256": (
            sha256_file(features_dir / "validation-rows.jsonl") if features_dir else None
        ),
        "task_manifest_sha256": sha256_file(tasks_path),
        "media_manifest_sha256": media_manifest_hashes,
        "result_predictions_sha256": sha256_file(predictions_path),
        "control_predictions_sha256": (
            sha256_file(control_predictions_path) if control_predictions_path else None
        ),
        "per_target_label": by_target,
        "overall": overall,
        "image_controls": controls,
        "runtime": {
            "device": "cpu",
            "cpu_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "elapsed_seconds": elapsed_seconds,
            "image_controls_elapsed_seconds": controls_elapsed_seconds,
        },
        "source_rights_status": (
            "Flickr page CC BY 2.0 declaration and attribution profile matched; "
            "individual ownership was not independently verified"
        ),
    }
    _write_json(output_dir / "result.json", metadata)
    print(
        json.dumps(
            {
                "count": overall["count"],
                "accuracy": overall["accuracy"],
                "nll": overall["nll"],
                "brier": overall["brier"],
                "ece_15_bins": overall["ece_15_bins"],
                "mean_confidence": overall["mean_confidence"],
                "elapsed_seconds": elapsed_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
