"""Train a small CLEVR-4 readout on pinned frozen VideoPrism image features."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.extract_frozen_videoprism_clevr4_features import (
    CHECKPOINT_SHA256,
    DATASET_REVISION,
    MODEL_REVISION,
)
from scripts.train_frozen_text_probe import CandidateScorer, _metrics
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--selection-sample-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrained_reuse/path_a_image_videoprism_clevr4_full.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=64)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _features_by_id(cache_path: Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    with np.load(cache_path, allow_pickle=False) as payload:
        train_ids = payload["train_ids"].astype(str).tolist()
        validation_ids = payload["validation_ids"].astype(str).tolist()
        train_values = payload["train_features"]
        validation_values = payload["validation_features"]
        train_hashes = payload["train_media_sha256"].astype(str).tolist()
        validation_hashes = payload["validation_media_sha256"].astype(str).tolist()
    if train_values.shape != (len(train_ids), 768) or validation_values.shape != (
        len(validation_ids),
        768,
    ):
        raise ValueError("cached feature array shapes do not match IDs")
    if not np.isfinite(train_values).all() or not np.isfinite(validation_values).all():
        raise ValueError("feature cache contains NaN or Inf")
    if len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("feature cache IDs contain duplicates")
    if len(train_hashes) != len(set(train_hashes)) or len(validation_hashes) != len(
        set(validation_hashes)
    ):
        raise ValueError("feature cache media identities contain duplicates within a split")
    if set(train_ids) & set(validation_ids) or set(train_hashes) & set(validation_hashes):
        raise ValueError("feature cache contains train/validation identity overlap")
    features = {
        key: torch.from_numpy(value.copy())
        for key, value in zip(train_ids, train_values, strict=True)
    }
    features.update(
        {
            key: torch.from_numpy(value.copy())
            for key, value in zip(validation_ids, validation_values, strict=True)
        }
    )
    media = dict(zip(train_ids + validation_ids, train_hashes + validation_hashes, strict=True))
    return features, media


def _option_embeddings(
    device: torch.device, model_path: Path
) -> tuple[dict[str, Tensor], nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prompts = {
        f"{taxonomy}|{option}": f"Image attribute: {taxonomy}. Candidate class: {option}."
        for taxonomy, options in CLEVR4_TAXONOMIES.items()
        for option in options
    }
    keys = list(prompts)
    encoded = tokenizer(
        [prompts[key] for key in keys],
        max_length=64,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float().cpu()
    if not torch.isfinite(pooled).all():
        raise ValueError("MiniLM candidate embeddings contain NaN or Inf")
    return dict(zip(keys, pooled, strict=True)), model


def _tasks(
    rows: list[dict[str, Any]],
    features_by_id: dict[str, Tensor],
    option_embeddings: dict[str, Tensor],
) -> tuple[list[Tensor], list[int], list[str], list[str]]:
    tasks: list[Tensor] = []
    targets: list[int] = []
    group_ids: list[str] = []
    taxonomies: list[str] = []
    for row in rows:
        image_id = str(row["id"])
        if image_id not in features_by_id:
            raise ValueError(f"no frozen image feature for {image_id}")
        for taxonomy, options in CLEVR4_TAXONOMIES.items():
            target = str(row[taxonomy])
            if target not in options:
                raise ValueError(f"unknown CLEVR-4 {taxonomy} label {target!r}")
            option_features = torch.stack(
                [option_embeddings[f"{taxonomy}|{option}"] for option in options]
            )
            image_features = features_by_id[image_id].expand(len(options), -1)
            tasks.append(torch.cat([image_features, option_features], dim=-1))
            targets.append(options.index(target))
            group_ids.append(image_id)
            taxonomies.append(taxonomy)
    return tasks, targets, group_ids, taxonomies


def _predict(scorer: nn.Module, tasks: list[Tensor], device: torch.device) -> list[Tensor]:
    scorer.eval()
    with torch.inference_mode():
        widths = [task.shape[0] for task in tasks]
        flat_logits = scorer(torch.cat(tasks).to(device)).cpu()
    outputs = []
    offset = 0
    for width in widths:
        outputs.append(flat_logits[offset : offset + width])
        offset += width
    return outputs


def _taxonomy_metrics(
    logits: list[Tensor], targets: list[int], taxonomies: list[str]
) -> dict[str, Any]:
    return {
        taxonomy: _metrics(
            [item for item, kind in zip(logits, taxonomies, strict=True) if kind == taxonomy],
            [target for target, kind in zip(targets, taxonomies, strict=True) if kind == taxonomy],
        )
        for taxonomy in CLEVR4_TAXONOMIES
    }


def _prediction_rows(
    rows: list[dict[str, Any]], logits: list[Tensor], targets: list[int]
) -> list[dict[str, Any]]:
    predictions = []
    offset = 0
    for row in rows:
        for taxonomy, options in CLEVR4_TAXONOMIES.items():
            probabilities = torch.softmax(logits[offset].float(), dim=-1)
            target = targets[offset]
            predictions.append(
                {
                    "id": f"{row['id']}:{taxonomy}",
                    "image_id": row["id"],
                    "taxonomy": taxonomy,
                    "target": options[target],
                    "prediction": options[int(probabilities.argmax())],
                    "options": options,
                    "probabilities": probabilities.tolist(),
                }
            )
            offset += 1
    if offset != len(logits):
        raise AssertionError("prediction rows do not cover all tasks")
    return predictions


def main() -> None:
    args = _args()
    sample_dir = args.sample_dir.resolve(strict=True)
    selection_dir = args.selection_sample_dir.resolve(strict=True)
    cache_path = args.feature_cache.resolve(strict=True)
    cache_report_path = args.cache_report.resolve(strict=True)
    text_model = args.text_model.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if args.seed != 17 or args.epochs != 8 or args.batch_questions != 64:
        raise ValueError("this frozen experiment requires seed 17, 8 epochs, batch_questions 64")

    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    if cache_report.get("status") != "complete_videoprism_clevr4_full_feature_cache":
        raise ValueError("feature cache report does not describe a complete full extraction")
    encoder_metadata = cache_report.get("encoder", {})
    if encoder_metadata.get("model_revision") != MODEL_REVISION or encoder_metadata.get(
        "checkpoint_sha256"
    ) != CHECKPOINT_SHA256:
        raise ValueError("feature cache encoder metadata does not match the pinned checkpoint")
    if cache_report.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("feature cache dataset revision mismatch")
    if cache_report.get("feature_cache_sha256") != sha256_file(cache_path):
        raise ValueError("feature cache SHA-256 mismatch")
    if cache_report.get("sample_hashes", {}).get("selection_validation_manifest") != sha256_file(
        selection_dir / "val-image-manifest.jsonl"
    ):
        raise ValueError("checkpoint-selection validation manifest hash mismatch")
    for name in (
        "train_labels",
        "validation_labels",
        "train_image_manifest",
        "validation_image_manifest",
        "fetch_report",
    ):
        path_name = {
            "train_labels": "train-labels.jsonl",
            "validation_labels": "val-labels.jsonl",
            "train_image_manifest": "train-image-manifest.jsonl",
            "validation_image_manifest": "val-image-manifest.jsonl",
            "fetch_report": "fetch-report.json",
        }[name]
        if cache_report["sample_hashes"][name] != sha256_file(sample_dir / path_name):
            raise ValueError(f"source data changed since feature extraction: {name}")
    features_by_id, media_by_id = _features_by_id(cache_path)
    train_rows = _load_rows(sample_dir / "train-labels.jsonl")
    all_validation = _load_rows(sample_dir / "val-labels.jsonl")
    selection_ids = {
        str(row["id"]) for row in _load_rows(selection_dir / "val-image-manifest.jsonl")
    }
    selected_rows = [row for row in all_validation if str(row["id"]) in selection_ids]
    report_holdout_rows = [row for row in all_validation if str(row["id"]) not in selection_ids]
    if len(train_rows) != 8424 or len(selected_rows) != 512 or len(report_holdout_rows) != 1595:
        raise ValueError("full CLEVR-4 split counts differ from the frozen experiment config")
    if {row["id"] for row in train_rows} & selection_ids:
        raise ValueError("train and checkpoint-selection validation IDs overlap")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    option_embeddings, text_encoder = _option_embeddings(device, text_model)
    train_tasks, train_targets, _, _ = _tasks(train_rows, features_by_id, option_embeddings)
    selection_tasks, selection_targets, _, selection_types = _tasks(
        selected_rows, features_by_id, option_embeddings
    )
    del text_encoder, option_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()

    scorer = CandidateScorer(768 + 384, hidden_size=128).to(device)
    parameter_count = sum(parameter.numel() for parameter in scorer.parameters())
    if parameter_count != 147713:
        raise ValueError(f"unexpected readout parameter count: {parameter_count}")
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    order_rng = random.Random(args.seed)
    history = []
    best_nll = float("inf")
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_tasks)))
        order_rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), args.batch_questions):
            batch_indices = order[start : start + args.batch_questions]
            batch_tasks = [train_tasks[index] for index in batch_indices]
            targets = torch.tensor([train_targets[index] for index in batch_indices], device=device)
            widths = [task.shape[0] for task in batch_tasks]
            flat_scores = scorer(torch.cat(batch_tasks).to(device))
            logits = flat_scores.new_full((len(batch_tasks), max(widths)), -1e4)
            offset = 0
            for row_index, width in enumerate(widths):
                logits[row_index, :width] = flat_scores[offset : offset + width]
                offset += width
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        selection_logits = _predict(scorer, selection_tasks, device)
        selection_metrics = _metrics(selection_logits, selection_targets)
        history.append(
            {
                "epoch": epoch,
                "train_ce": float(np.mean(losses)),
                "checkpoint_selection_validation": selection_metrics,
                "selection_taxonomy_metrics": _taxonomy_metrics(
                    selection_logits, selection_targets, selection_types
                ),
            }
        )
        if selection_metrics["nll"] < best_nll:
            best_nll = selection_metrics["nll"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("checkpoint selection failed to produce a readout")
    scorer.load_state_dict(best_state)
    final_selection_logits = _predict(scorer, selection_tasks, device)
    selected_metrics = _metrics(final_selection_logits, selection_targets)
    selected_by_taxonomy = _taxonomy_metrics(
        final_selection_logits, selection_targets, selection_types
    )

    # Only after checkpoint selection is frozen do we materialize holdout task labels,
    # produce holdout predictions, and compute the one-time reporting metrics.
    # Task features need the same fixed option embeddings. Rebuild them only after selection.
    option_embeddings, text_encoder = _option_embeddings(device, text_model)
    holdout_tasks, holdout_targets, holdout_groups, holdout_types = _tasks(
        report_holdout_rows, features_by_id, option_embeddings
    )
    del text_encoder, option_embeddings
    holdout_logits = _predict(scorer, holdout_tasks, device)
    holdout_metrics = _metrics(holdout_logits, holdout_targets)
    holdout_by_taxonomy = _taxonomy_metrics(holdout_logits, holdout_targets, holdout_types)

    output.mkdir(parents=True, exist_ok=False)
    readout_path = output / "best-readout.pt"
    torch.save(best_state, readout_path)
    prediction_sets = (
        (
            "checkpoint-selection-predictions.jsonl",
            _prediction_rows(selected_rows, final_selection_logits, selection_targets),
        ),
        (
            "reporting-validation-predictions.jsonl",
            _prediction_rows(report_holdout_rows, holdout_logits, holdout_targets),
        ),
    )
    for filename, rows in prediction_sets:
        (output / filename).write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
            encoding="utf-8",
        )
    report = {
        "schema_version": 1,
        "status": "complete_clevr4_videoprism_frozen_image_probe",
        "claim_scope": "synthetic CLEVR-4 image taxonomy decisions; static image mode only",
        "seed": args.seed,
        "source_revision": DATASET_REVISION,
        "config_sha256": sha256_file(config_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "cache_report_sha256": sha256_file(cache_report_path),
        "training_image_count": len(train_rows),
        "training_unique_media_count": len({media_by_id[str(row['id'])] for row in train_rows}),
        "checkpoint_selection_validation_image_count": len(selected_rows),
        "checkpoint_selection_validation_unique_media_count": len(
            {media_by_id[str(row["id"])] for row in selected_rows}
        ),
        "one_time_reporting_validation_image_count": len(report_holdout_rows),
        "one_time_reporting_validation_unique_media_count": len(
            {media_by_id[str(row["id"])] for row in report_holdout_rows}
        ),
        "train_validation_id_overlap": 0,
        "train_validation_media_hash_overlap": 0,
        "selection_reporting_id_overlap": 0,
        "training_task_count": len(train_tasks),
        "checkpoint_selection_task_count": len(selection_tasks),
        "reporting_task_count": len(holdout_tasks),
        "encoder_revision": MODEL_REVISION,
        "encoder_checkpoint_sha256": CHECKPOINT_SHA256,
        "candidate_text_revision": TEXT_REVISION,
        "readout_parameters": parameter_count,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "epochs": args.epochs,
        "batch_questions": args.batch_questions,
        "history": history,
        "best_epoch": best_epoch,
        "checkpoint_selection_rule": "minimum NLL on prior 512-image validation subset",
        "checkpoint_selection_metrics": selected_metrics,
        "checkpoint_selection_by_taxonomy": selected_by_taxonomy,
        "one_time_reporting_validation_metrics": holdout_metrics,
        "one_time_reporting_validation_by_taxonomy": holdout_by_taxonomy,
        "training_seconds": training_seconds,
        "device": str(device),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "sealed_audit_loaded": False,
        "note": (
            "New encoder and full train coverage differ from the prior V-JEPA sample. "
            "The one-time validation subset is not a sealed audit."
        ),
        "readout_sha256": sha256_file(readout_path),
    }
    (output / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report["run_report_sha256"] = sha256_file(output / "run-report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
