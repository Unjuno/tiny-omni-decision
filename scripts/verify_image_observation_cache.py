"""Persist frozen CLEVR-4 features and verify the existing readout after reload."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_image_probe import (
    CandidateScorer,
    _load_rows,
    _metrics,
    _option_embeddings,
    _predict,
    _tasks,
)
from tiny_omni_decision.cache import ObservationFeatureCache
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file
from tiny_omni_decision.observation_identity import image_observation_feature_key

EXPECTED_VJEPA_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
EXPECTED_CLEVR4_REVISION = "cddc78fb2a8359dc958987b2c750bfdd4bfd2c73"
EXPECTED_TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pretrained_reuse/path_a_clevr4_vjepa.yaml")
    )
    parser.add_argument("--vjepa-source", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def _tensor_from_payload(payload: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    expected_bytes = torch.empty(shape, dtype=torch.float32).numel() * 4
    if len(payload) != expected_bytes:
        raise ValueError(f"cached feature has {len(payload)} bytes; expected {expected_bytes}")
    return torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(shape).clone()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _build_predictions(
    rows: list[dict[str, Any]],
    logits: list[torch.Tensor],
    targets: list[int],
    taxonomies: list[str],
) -> list[dict[str, Any]]:
    predictions = []
    for index, (row_logits, target, taxonomy) in enumerate(
        zip(logits, targets, taxonomies, strict=True)
    ):
        options = list(CLEVR4_TAXONOMIES[taxonomy])
        probabilities = torch.softmax(row_logits.float(), dim=-1)
        prediction_index = int(probabilities.argmax())
        image_id = str(rows[index // len(CLEVR4_TAXONOMIES)]["id"])
        predictions.append(
            {
                "id": f"{image_id}:{taxonomy}",
                "image_id": image_id,
                "taxonomy": taxonomy,
                "target": options[target],
                "prediction": options[prediction_index],
                "options": options,
                "probabilities": probabilities.tolist(),
            }
        )
    return predictions


def main() -> None:
    args = _args()
    started = datetime.now(UTC)
    sample_root = args.sample_dir.resolve()
    run_dir = args.run_dir.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    image_cache_path = args.image_cache.resolve()
    vjepa_source = args.vjepa_source.resolve()
    vjepa_checkpoint = args.vjepa_checkpoint.resolve()
    text_model_path = args.text_model.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite image-cache verification output: {output}")

    report_path = run_dir / "run-report.json"
    feature_path = run_dir / "image-features.pt"
    readout_path = run_dir / "best-readout.pt"
    predictions_path = run_dir / "validation-predictions.jsonl"
    for path in (report_path, feature_path, readout_path, predictions_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"required read-only image probe artifact missing: {path}")
    original_report_bytes = report_path.read_bytes()
    run_report = json.loads(original_report_bytes)
    if run_report.get("status") != "complete_image_only_synthetic_probe":
        raise ValueError("input image probe report is not complete")
    if run_report.get("sealed_audit_loaded") is True:
        raise ValueError("input report indicates sealed audit access")
    if sha256_file(feature_path) != run_report["image_feature_cache_sha256"]:
        raise ValueError("frozen image feature cache hash mismatch")
    if (
        run_report["vjepa_checkpoint_sha256"]
        != "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
    ):
        raise ValueError("image probe was not built from the pinned V-JEPA checkpoint")
    vjepa_commit = subprocess.run(
        ["git", "-C", str(vjepa_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if vjepa_commit != EXPECTED_VJEPA_COMMIT:
        raise ValueError(f"unexpected V-JEPA source revision: {vjepa_commit}")
    vjepa_source_dirty_paths = subprocess.run(
        ["git", "-C", str(vjepa_source), "status", "--short", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not vjepa_checkpoint.is_file() or vjepa_checkpoint.stat().st_size != 1_664_223_428:
        raise ValueError("pinned V-JEPA checkpoint file is missing or has unexpected size")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["source"]["revision"] != EXPECTED_CLEVR4_REVISION:
        raise ValueError("CLEVR-4 revision is not the pinned candidate revision")
    if config["encoder"]["source_revision"] != EXPECTED_VJEPA_COMMIT:
        raise ValueError("V-JEPA config revision differs from pinned source")
    if config["candidate_text_encoder"]["revision"] != EXPECTED_TEXT_REVISION:
        raise ValueError("candidate text encoder revision differs from pinned MiniLM")

    train_rows = _load_rows(sample_root / "train-labels.jsonl")
    validation_rows = _load_rows(sample_root / "val-labels.jsonl")
    train_assets = _read_jsonl(sample_root / "train-image-manifest.jsonl")
    validation_assets = _read_jsonl(sample_root / "val-image-manifest.jsonl")
    for rows, assets, expected_split in (
        (train_rows, train_assets, "train"),
        (validation_rows, validation_assets, "val"),
    ):
        if any(row.get("split") != expected_split for row in rows):
            raise ValueError(f"{expected_split} labels contain another split")
        if [str(row["id"]) for row in rows] != [str(row["id"]) for row in assets]:
            raise ValueError(f"{expected_split} label/asset ID order differs")
        if len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError(f"duplicate {expected_split} image IDs")
    train_ids = {str(row["id"]) for row in train_rows}
    validation_ids = {str(row["id"]) for row in validation_rows}
    train_hashes = {str(row["sha256"]) for row in train_assets}
    validation_hashes = {str(row["sha256"]) for row in validation_assets}
    if train_ids & validation_ids or train_hashes & validation_hashes:
        raise ValueError("CLEVR-4 image identity overlaps frozen train/validation")

    feature_cache = torch.load(image_cache_path, map_location="cpu", weights_only=True)
    if sha256_file(image_cache_path) != run_report["image_feature_cache_sha256"]:
        raise ValueError("external frozen image feature cache hash mismatch")
    if feature_cache["source_revision"] != f"sgvaze/clevr4@{EXPECTED_CLEVR4_REVISION}":
        raise ValueError("image cache source revision mismatch")
    if feature_cache["encoder_commit"] != EXPECTED_VJEPA_COMMIT:
        raise ValueError("image cache V-JEPA revision mismatch")
    if feature_cache["encoder_sha256"] != run_report["vjepa_checkpoint_sha256"]:
        raise ValueError("image cache V-JEPA weight hash mismatch")
    for split_name, rows, assets in (
        ("train", train_rows, train_assets),
        ("validation", validation_rows, validation_assets),
    ):
        id_key = "train_ids" if split_name == "train" else "validation_ids"
        feature_key = "train_features" if split_name == "train" else "validation_features"
        hash_key = "train_asset_sha256" if split_name == "train" else "validation_asset_sha256"
        expected_ids = [str(row["id"]) for row in rows]
        expected_hashes = [str(row["sha256"]) for row in assets]
        if feature_cache[id_key] != expected_ids:
            raise ValueError(f"{split_name} cached image ID order changed")
        if feature_cache[hash_key] != expected_hashes:
            raise ValueError(f"{split_name} cached image media hashes changed")
        if feature_cache[feature_key].shape != (len(rows), 768):
            raise ValueError(f"unexpected {split_name} image feature tensor shape")

    preprocessor_module = vjepa_source / "evals/hub/preprocessor.py"
    transform_module = vjepa_source / "evals/video_classification_frozen/utils.py"
    spatial_module = vjepa_source / "src/datasets/utils/video/transforms.py"
    volume_module = vjepa_source / "src/datasets/utils/video/volume_transforms.py"
    preprocessing_components = {
        "preprocessor_sha256": sha256_file(preprocessor_module),
        "transform_factory_sha256": sha256_file(transform_module),
        "spatial_transform_sha256": sha256_file(spatial_module),
        "volume_transform_sha256": sha256_file(volume_module),
        "feature_extraction_code_sha256": sha256_file(
            Path("scripts/train_frozen_image_probe.py").resolve()
        ),
        "eval_transform": "official eval transforms, crop_size=384, RGB, mean of 576 tokens",
    }
    preprocessing_sha256 = hashlib.sha256(
        json.dumps(preprocessing_components, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_id = "sgvaze/clevr4"
    output.mkdir(parents=True, exist_ok=False)
    feature_cache_store = ObservationFeatureCache(output / "feature-cache")
    source_features = {
        "train": feature_cache["train_features"].float(),
        "validation": feature_cache["validation_features"].float(),
    }
    if any(not torch.isfinite(features).all() for features in source_features.values()):
        raise ValueError("source image feature cache contains non-finite values")
    reloaded_features: dict[str, list[torch.Tensor]] = {"train": [], "validation": []}
    cache_records: list[dict[str, Any]] = []
    all_rows = (
        ("train", train_rows, train_assets),
        ("validation", validation_rows, validation_assets),
    )
    cache_started = time.perf_counter()
    for split_name, rows, assets in all_rows:
        for index, (row, asset) in enumerate(zip(rows, assets, strict=True)):
            image_path = (
                sample_root
                / split_name.replace("validation", "val")
                / "images"
                / f"{row['id']}.png"
            )
            if not image_path.is_file() or sha256_file(image_path) != asset["sha256"]:
                raise ValueError(f"missing or changed {split_name} image: {row['id']}")
            key = image_observation_feature_key(
                source_id=source_id,
                source_revision=EXPECTED_CLEVR4_REVISION,
                media_sha256=str(asset["sha256"]),
                encoder_id="facebookresearch/vjepa2/vit_base_16_384",
                encoder_revision=EXPECTED_VJEPA_COMMIT,
                encoder_weights_sha256=str(run_report["vjepa_checkpoint_sha256"]),
                preprocessor_id="official_vjepa2_preprocessor_384",
                preprocessor_revision=EXPECTED_VJEPA_COMMIT,
                preprocessing_sha256=preprocessing_sha256,
            )
            feature = source_features[split_name][index].contiguous()
            cached = feature_cache_store.put(key, feature.numpy().tobytes())
            restored_entry = feature_cache_store.get(key)
            restored = _tensor_from_payload(restored_entry.payload, key.feature_shape)
            if not torch.equal(feature, restored):
                raise ValueError(f"persisted image feature differs for {row['id']}")
            reloaded_features[split_name].append(restored)
            cache_records.append(
                {
                    "image_id": str(row["id"]),
                    "split": split_name,
                    "cache_id": key.cache_id,
                    "media_sha256": asset["sha256"],
                    "feature_payload_sha256": cached.payload_sha256,
                }
            )
    cache_seconds = time.perf_counter() - cache_started
    train_features = torch.stack(reloaded_features["train"])
    validation_features = torch.stack(reloaded_features["validation"])
    if not torch.equal(train_features, source_features["train"]):
        raise ValueError("train image feature ordering changed on cache reload")
    if not torch.equal(validation_features, source_features["validation"]):
        raise ValueError("validation image feature ordering changed on cache reload")

    text_weights = text_model_path / "model.safetensors"
    expected_text_sha256 = run_report["text_model_weights_sha256"]
    if sha256_file(text_weights) != expected_text_sha256:
        raise ValueError("pinned MiniLM weights changed")
    tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
    text_model = AutoModel.from_pretrained(text_model_path, local_files_only=True).eval()
    for parameter in text_model.parameters():
        parameter.requires_grad_(False)
    option_embeddings, text_hidden = _option_embeddings(tokenizer, text_model, torch.device("cpu"))
    text_parameter_count = sum(parameter.numel() for parameter in text_model.parameters())
    del text_model, tokenizer
    validation_examples, validation_targets, _, validation_taxonomies = _tasks(
        validation_rows, validation_features, option_embeddings
    )
    scorer = CandidateScorer(768 + text_hidden)
    scorer.load_state_dict(
        torch.load(readout_path, map_location="cpu", weights_only=True), strict=True
    )
    inference_started = time.perf_counter()
    validation_logits = _predict(scorer, validation_examples, torch.device("cpu"))
    cpu_inference_seconds = time.perf_counter() - inference_started
    selected_metrics = _metrics(validation_logits, validation_targets)
    by_taxonomy = {
        taxonomy: _metrics(
            [
                logit
                for logit, name in zip(validation_logits, validation_taxonomies, strict=True)
                if name == taxonomy
            ],
            [
                target
                for target, name in zip(validation_targets, validation_taxonomies, strict=True)
                if name == taxonomy
            ],
        )
        for taxonomy in CLEVR4_TAXONOMIES
    }
    new_predictions = _build_predictions(
        validation_rows, validation_logits, validation_targets, validation_taxonomies
    )
    saved_predictions = _read_jsonl(predictions_path)
    if len(new_predictions) != len(saved_predictions):
        raise ValueError("validation prediction count changed")
    max_probability_delta = 0.0
    prediction_mismatches = []
    for saved, actual in zip(saved_predictions, new_predictions, strict=True):
        for field in ("id", "image_id", "taxonomy", "target", "options"):
            if saved[field] != actual[field]:
                raise ValueError(f"image prediction identity differs at {field}: {saved['id']}")
        if saved["prediction"] != actual["prediction"]:
            prediction_mismatches.append(
                {
                    "id": saved["id"],
                    "saved_prediction": saved["prediction"],
                    "cpu_prediction": actual["prediction"],
                }
            )
        delta = max(
            abs(float(left) - float(right))
            for left, right in zip(saved["probabilities"], actual["probabilities"], strict=True)
        )
        max_probability_delta = max(max_probability_delta, delta)
    max_metric_delta = max(
        abs(float(selected_metrics[key]) - float(run_report["validation_selected_metrics"][key]))
        for key in ("accuracy", "nll", "brier", "ece_15_bins")
    )
    max_metric_delta = max(
        max_metric_delta,
        max(
            abs(
                float(by_taxonomy[taxonomy][key])
                - float(run_report["validation_by_taxonomy"][taxonomy][key])
            )
            for taxonomy in CLEVR4_TAXONOMIES
            for key in ("accuracy", "nll", "brier", "ece_15_bins")
        ),
    )

    manifest_path = output / "cache-records.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in cache_records),
        encoding="utf-8",
    )
    entries = list((output / "feature-cache").glob("*.feature"))
    report = {
        "schema_version": 1,
        "status": "complete_image_observation_cache_validation",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_probe_report_sha256": sha256_file(report_path),
        "source_probe_commit": "UNKNOWN (original probe report did not record its source commit)",
        "source_probe_script_sha256": "UNKNOWN (not recorded in source probe report)",
        "source_feature_cache_sha256": sha256_file(image_cache_path),
        "source_readout_sha256": sha256_file(readout_path),
        "source_validation_predictions_sha256": sha256_file(predictions_path),
        "source_config_sha256": sha256_file(config_path),
        "clevr4_revision": EXPECTED_CLEVR4_REVISION,
        "vjepa_source_revision": vjepa_commit,
        "vjepa_source_worktree_dirty_paths": vjepa_source_dirty_paths,
        "vjepa_checkpoint_sha256": run_report["vjepa_checkpoint_sha256"],
        "vjepa_checkpoint_file_size_bytes": vjepa_checkpoint.stat().st_size,
        "vjepa_checkpoint_rehashed_in_this_diagnostic": False,
        "text_revision": EXPECTED_TEXT_REVISION,
        "text_weights_sha256": expected_text_sha256,
        "text_parameter_count": text_parameter_count,
        "source_run_text_parameter_count": "UNKNOWN (not serialized by original image probe)",
        "preprocessing_components": preprocessing_components,
        "preprocessing_sha256": preprocessing_sha256,
        "train_images": len(train_rows),
        "validation_images": len(validation_rows),
        "train_ordered_ids_sha256": _ids_sha256(train_rows),
        "validation_ordered_ids_sha256": _ids_sha256(validation_rows),
        "train_unique_assets": len(train_hashes),
        "validation_unique_assets": len(validation_hashes),
        "train_validation_image_id_overlap": len(train_ids & validation_ids),
        "train_validation_media_hash_overlap": len(train_hashes & validation_hashes),
        "cache_entries": len(entries),
        "cache_payload_bytes": len(cache_records) * 768 * 4,
        "cache_total_bytes": sum(entry.stat().st_size for entry in entries),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_manifest_path": str(manifest_path),
        "cache_directory": str(output / "feature-cache"),
        "cache_write_reload_seconds": cache_seconds,
        "validation_tasks": len(new_predictions),
        "validation_tasks_by_taxonomy": dict(Counter(validation_taxonomies)),
        "reloaded_validation_metrics": selected_metrics,
        "reloaded_validation_by_taxonomy": by_taxonomy,
        "saved_run_validation_metrics": run_report["validation_selected_metrics"],
        "saved_run_validation_by_taxonomy": run_report["validation_by_taxonomy"],
        "max_probability_delta_vs_saved_cuda_predictions": max_probability_delta,
        "max_metric_delta_vs_saved_cuda_report": max_metric_delta,
        "prediction_classes_exact_vs_saved_run": not prediction_mismatches,
        "prediction_class_mismatch_count_vs_saved_run": len(prediction_mismatches),
        "prediction_class_mismatches_vs_saved_run": prediction_mismatches,
        "cpu_validation_inference_seconds": cpu_inference_seconds,
        "device": "CPU; V-JEPA and CUDA not used",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "torchvision": importlib.metadata.version("torchvision"),
            "pillow": importlib.metadata.version("Pillow"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "sealed_audit_loaded": False,
        "test_split_loaded": False,
        "image_encoder_executions": 0,
        "training_performed": False,
        "source_archive_full_hash_verified": False,
        "note": (
            "Existing frozen train/validation features and readout were read only. All selected "
            "image assets were checked against per-image SHA-256 manifests. The source archive "
            "full SHA-512 remains unverified, and CLEVR-4 is a synthetic image taxonomy task."
        ),
    }
    report_path_out = output / "verification-report.json"
    report_path_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if report_path.read_bytes() != original_report_bytes:
        raise AssertionError("source image probe report changed during verification")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
