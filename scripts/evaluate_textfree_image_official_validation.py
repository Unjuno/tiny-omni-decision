"""One-time evaluation of the frozen text-free CLEVR-4 image readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def align_validation_rows(
    feature_ids: list[str],
    feature_hashes: list[str],
    label_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    train_ids: list[str],
    train_hashes: list[str],
) -> list[dict[str, Any]]:
    """Fail closed unless official validation features/labels/assets align exactly."""
    if not feature_ids or len(feature_ids) != len(feature_hashes):
        raise ValueError("validation feature IDs and asset hashes must be nonempty and aligned")
    if len(set(feature_ids)) != len(feature_ids) or len(set(feature_hashes)) != len(feature_hashes):
        raise ValueError("validation feature IDs and asset hashes must be unique")
    if (
        len(train_ids) != len(train_hashes)
        or len(set(train_ids)) != len(train_ids)
        or len(set(train_hashes)) != len(train_hashes)
    ):
        raise ValueError("training IDs and asset hashes must be aligned and unique")
    if set(feature_ids) & set(train_ids) or set(feature_hashes) & set(train_hashes):
        raise ValueError("official validation records overlap the original training partition")
    if len(label_rows) != len(feature_ids) or len(manifest_rows) != len(feature_ids):
        raise ValueError("feature, label, and manifest row counts differ")
    rows: list[dict[str, Any]] = []
    for index, (label, manifest) in enumerate(zip(label_rows, manifest_rows, strict=True)):
        expected_id = feature_ids[index]
        expected_hash = feature_hashes[index]
        if (
            label.get("id") != expected_id
            or manifest.get("id") != expected_id
            or label.get("split") != "val"
            or manifest.get("split") != "val"
            or manifest.get("sha256") != expected_hash
        ):
            raise ValueError(f"validation identity/order/hash mismatch at row {index}")
        rows.append(label)
    return rows


def _metrics(logits, targets, class_ids: list[str]) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    probabilities = torch.softmax(logits.float(), dim=-1)
    prediction = probabilities.argmax(dim=-1)
    correct = prediction.eq(targets)
    confidence = probabilities.max(dim=-1).values
    one_hot = F.one_hot(targets, num_classes=len(class_ids)).float()
    ece = 0.0
    class_counts = torch.bincount(targets, minlength=len(class_ids))
    recalls = []
    for class_index, count in enumerate(class_counts.tolist()):
        if count:
            class_prediction = prediction[targets.eq(class_index)] == class_index
            recalls.append(float(class_prediction.float().mean()))
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        mask = (confidence >= lower) & (
            confidence < upper if bin_index < 14 else confidence <= upper
        )
        if mask.any():
            ece += float(mask.float().mean()) * abs(
                float(correct[mask].float().mean()) - float(confidence[mask].mean())
            )
    return {
        "count": len(targets),
        "accuracy": float(correct.float().mean()),
        "balanced_accuracy_observed_classes": sum(recalls) / len(recalls),
        "nll": float(F.cross_entropy(logits.float(), targets)),
        "brier": float((probabilities - one_hot).square().sum(dim=-1).mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
        "majority_class_accuracy": int(class_counts.max()) / len(targets),
        "class_counts": {
            class_id: int(class_counts[index]) for index, class_id in enumerate(class_ids)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
    from tiny_omni_decision.textfree_image import TextFreeImageReadout

    repo_root = Path(__file__).resolve().parents[1]
    run_started_at = datetime.now(UTC).isoformat()
    fit_report_path = args.fit_dir / "run-report.json"
    fit_report = json.loads(fit_report_path.read_text(encoding="utf-8"))
    fit_report_hash_path = args.fit_dir / "run-report.sha256"
    if not fit_report_hash_path.is_file():
        raise ValueError("fit report SHA-256 sidecar is missing")
    expected_fit_report_hash = fit_report_hash_path.read_text(encoding="utf-8").strip().split()[0]
    if _sha256(fit_report_path) != expected_fit_report_hash:
        raise ValueError("fit report does not match its SHA-256 sidecar")
    checkpoint_path = args.fit_dir / "best-readout.pt"
    if fit_report.get("status") != "complete_textfree_image_development_probe":
        raise ValueError("expected the pinned text-free image Decision fit")
    if fit_report.get("official_validation_loaded") is not False:
        raise ValueError("fit report does not prove official validation was excluded")
    if fit_report.get("sealed_audit_loaded") is not False:
        raise ValueError("fit report says sealed audit data were loaded")
    if fit_report.get("selected_epoch") != 502:
        raise ValueError("unexpected selected checkpoint epoch")
    if _sha256(checkpoint_path) != fit_report["selected_readout_sha256"]:
        raise ValueError("selected readout hash differs from the fit receipt")
    if _sha256(args.feature_cache) != fit_report["feature_cache_sha256"]:
        raise ValueError("frozen feature cache hash differs from the fit receipt")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")

    cache = np.load(args.feature_cache, allow_pickle=False)
    features = cache["validation_features"]
    feature_ids = cache["validation_ids"].astype(str).tolist()
    feature_hashes = cache["validation_media_sha256"].astype(str).tolist()
    train_ids = cache["train_ids"].astype(str).tolist()
    train_hashes = cache["train_media_sha256"].astype(str).tolist()
    labels_path = args.corpus_dir / "val-labels.jsonl"
    manifest_path = args.corpus_dir / "val-image-manifest.jsonl"
    label_rows = _jsonl(labels_path)
    manifest_rows = _jsonl(manifest_path)
    rows = align_validation_rows(
        feature_ids,
        feature_hashes,
        label_rows,
        manifest_rows,
        train_ids,
        train_hashes,
    )
    if len(rows) != 2107 or features.shape != (2107, 768):
        raise ValueError("unexpected official CLEVR-4 validation split dimensions")
    if not np.isfinite(features).all():
        raise ValueError("validation feature cache contains non-finite values")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TextFreeImageReadout(768, CLEVR4_TAXONOMIES)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    feature_tensor = torch.from_numpy(features.copy())
    logits_by_taxonomy: dict[str, Any] = {}
    target_by_taxonomy: dict[str, Any] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for taxonomy, class_ids in CLEVR4_TAXONOMIES.items():
            logits_by_taxonomy[taxonomy] = model(feature_tensor, taxonomy).cpu()
            class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
            target_by_taxonomy[taxonomy] = torch.tensor(
                [class_to_index[str(row[taxonomy])] for row in rows], dtype=torch.long
            )
    elapsed = time.perf_counter() - started

    taxonomy_metrics = {
        taxonomy: _metrics(logits_by_taxonomy[taxonomy], target_by_taxonomy[taxonomy], class_ids)
        for taxonomy, class_ids in CLEVR4_TAXONOMIES.items()
    }
    macro = {
        metric: sum(values[metric] for values in taxonomy_metrics.values())
        / len(taxonomy_metrics)
        for metric in (
            "accuracy",
            "balanced_accuracy_observed_classes",
            "nll",
            "brier",
            "ece_15_bins",
            "mean_confidence",
        )
    }
    # JSON Decision contract sanity: option identity must survive a reversed supplied order.
    permutation_checks = 0
    with torch.inference_mode():
        for taxonomy, class_ids in CLEVR4_TAXONOMIES.items():
            options = [
                {"option_id": f"o{i}", "class_id": class_id}
                for i, class_id in enumerate(class_ids)
            ]
            row = feature_tensor[0]
            forward = model.decide(row, taxonomy, options)["options"]
            reversed_result = model.decide(row, taxonomy, list(reversed(options)))["options"]
            option_to_class = {option["option_id"]: option["class_id"] for option in options}
            forward_by_class = {
                option_to_class[value["option_id"]]: value["probability"] for value in forward
            }
            reverse_by_class = {
                option_to_class[value["option_id"]]: value["probability"]
                for value in reversed_result
            }
            if any(
                not math.isclose(forward_by_class[c], reverse_by_class[c], rel_tol=0, abs_tol=1e-6)
                for c in class_ids
            ):
                raise ValueError(f"candidate permutation changed class probabilities: {taxonomy}")
            permutation_checks += 1

    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row_index, row in enumerate(rows):
            for taxonomy, class_ids in CLEVR4_TAXONOMIES.items():
                logits = logits_by_taxonomy[taxonomy][row_index]
                probabilities = torch.softmax(logits.float(), dim=-1).tolist()
                predicted_index = int(torch.argmax(logits))
                record = {
                    "id": row["id"],
                    "asset_sha256": feature_hashes[row_index],
                    "taxonomy": taxonomy,
                    "target": str(row[taxonomy]),
                    "prediction": class_ids[predicted_index],
                    "probabilities": dict(zip(class_ids, probabilities, strict=True)),
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    scores_path = args.output_dir / "scores.npy"
    score_array = np.stack(
        [logits_by_taxonomy[name].numpy() for name in CLEVR4_TAXONOMIES], axis=1
    ).astype(np.float32)
    np.save(scores_path, score_array, allow_pickle=False)
    report = {
        "status": "PASS_ONE_TIME_TEXTFREE_IMAGE_OFFICIAL_VALIDATION",
        "run_started_at_utc": run_started_at,
        "command": [sys.executable, *sys.argv],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "evaluation_script_sha256": _sha256(Path(__file__).resolve()),
        "readout_source_sha256": _sha256(repo_root / "src/tiny_omni_decision/textfree_image.py"),
        "dataset_source_sha256": _sha256(repo_root / "src/tiny_omni_decision/dataset.py"),
        "json_contract_source_sha256": _sha256(
            repo_root / "src/tiny_omni_decision/classification_json.py"
        ),
        "model_source_commit": fit_report["source_commit"],
        "effective_fit_config_sha256": fit_report["effective_config_sha256"],
        "selected_epoch": fit_report["selected_epoch"],
        "selected_checkpoint_rule": fit_report["selected_checkpoint_rule"],
        "selected_checkpoint_sha256": _sha256(checkpoint_path),
        "fit_report_sha256": _sha256(fit_report_path),
        "feature_cache_sha256": _sha256(args.feature_cache),
        "validation_labels_sha256": _sha256(labels_path),
        "validation_manifest_sha256": _sha256(manifest_path),
        "validation_asset_count": len(rows),
        "training_ids": len(train_ids),
        "id_and_asset_overlap_with_training_partition": 0,
        "validation_id_order_sha256": hashlib.sha256(
            "\n".join(feature_ids).encode("utf-8")
        ).hexdigest(),
        "validation_asset_order_sha256": hashlib.sha256(
            "\n".join(feature_hashes).encode("utf-8")
        ).hexdigest(),
        "task_count": len(rows) * len(CLEVR4_TAXONOMIES),
        "encoder": fit_report["encoder"],
        "source_revision": fit_report["source_revision"],
        "trainable_parameters": fit_report["trainable_parameters"],
        "frozen_encoder_plus_readout_parameters": fit_report[
            "frozen_encoder_plus_readout_parameters"
        ],
        "metrics_by_taxonomy": taxonomy_metrics,
        "macro": macro,
        "option_order_permutation_checks": permutation_checks,
        "device": "cpu",
        "cuda_used": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "sealed_audit_loaded": False,
        "official_validation_was_previously_used_by_other_project_experiments": True,
        "independent_project_blind_evaluation": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "elapsed_seconds": elapsed,
        "predictions_sha256": _sha256(prediction_path),
        "scores_sha256": _sha256(scores_path),
        "limitations": [
            (
                "The checkpoint was frozen before this one-time evaluation, and these scores "
                "were not used for selection."
            ),
            (
                "Other project experiments used these official validation labels; this is not "
                "blind project-level evidence."
            ),
            "CLEVR-4 is synthetic image taxonomy classification, not natural-image reasoning.",
            (
                "The source archive-wide SHA-512 was not verified; per-image asset and "
                "feature-cache hashes were checked."
            ),
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
