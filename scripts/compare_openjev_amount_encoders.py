"""Compare two frozen text encoders on an internal Open-Jev train-family split.

This entry point reads only the publisher TRAIN parquet. It never discovers or
loads publisher validation, calibration, test, or OOD files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.openjev_amount_probe import (
    assert_split_integrity,
    format_pair,
    make_family_split,
    question_type_cells,
    validate_rows,
)

TRAIN_SHA256 = "dd09738e9463fd1efcdf87137600bea0c21270ee729c4475124d684b88d0cbb2"
DATASET_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_encoder(value: str) -> tuple[str, str, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("encoder must be NAME=MODEL_OR_DIR=REVISION")
    return parts[0], parts[1], parts[2]


def args_from_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder", action="append", type=parse_encoder, required=True)
    parser.add_argument("--expected-train-sha256", default=TRAIN_SHA256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


class CandidateScorer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(hidden_size, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x).squeeze(-1)


def _metrics(
    logits: Tensor, target: Tensor, mask: Tensor, records: list[dict[str, Any]]
) -> dict[str, Any]:
    logits = logits.float().masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    prediction = probs.argmax(dim=-1)
    chosen_target = target.gather(1, prediction[:, None]).squeeze(1)
    confidence = probs.max(dim=1).values
    ece = 0.0
    for bin_index in range(10):
        low, high = bin_index / 10.0, (bin_index + 1) / 10.0
        selected = (confidence >= low) & (
            (confidence < high) if bin_index < 9 else (confidence <= high)
        )
        if selected.any():
            ece += float(selected.float().mean()) * abs(
                float(confidence[selected].mean()) - float(chosen_target[selected].mean())
            )
    per_row = {
        "accuracy": (target.argmax(dim=1) == prediction).float().cpu().tolist(),
        "nll": (-(target * log_probs).sum(dim=1)).cpu().tolist(),
        "brier": ((probs - target).square().sum(dim=1)).cpu().tolist(),
        "ece_confidence": confidence.cpu().tolist(),
        "ece_expected_correctness": chosen_target.cpu().tolist(),
    }
    cells: dict[str, Any] = {}
    for question_id, indices in question_type_cells(records).items():
        cell: dict[str, float | int] = {"count": len(indices)}
        for metric in ("accuracy", "nll", "brier"):
            cell[metric] = float(np.mean([per_row[metric][i] for i in indices]))
        # ECE is computed within each question type cell, not pooled across tasks.
        cell_ece = 0.0
        for bin_index in range(10):
            low, high = bin_index / 10.0, (bin_index + 1) / 10.0
            chosen = [
                i
                for i in indices
                if per_row["ece_confidence"][i] >= low
                and (
                    per_row["ece_confidence"][i] < high
                    if bin_index < 9
                    else per_row["ece_confidence"][i] <= high
                )
            ]
            if chosen:
                cell_ece += (
                    len(chosen)
                    / len(indices)
                    * abs(
                        float(np.mean([per_row["ece_confidence"][i] for i in chosen]))
                        - float(np.mean([per_row["ece_expected_correctness"][i] for i in chosen]))
                    )
                )
        cell["ece_10bin"] = cell_ece
        cells[question_id] = cell
    macro = {
        key: float(np.mean([cell[key] for cell in cells.values()]))
        for key in ("accuracy", "nll", "brier", "ece_10bin")
    }
    predictions = []
    for index, row in enumerate(records):
        option_count = len(row["options"])
        predictions.append(
            {
                "id": row["id"],
                "family": row["family"],
                "question_id": row["question_id"],
                "target": row["target"],
                "prediction_index": int(prediction[index]),
                "probabilities": probs[index, :option_count].cpu().tolist(),
            }
        )
    return {"macro_over_question_id": macro, "by_question_id": cells, "predictions": predictions}


def _load_encoder(model_ref: str, revision: str) -> tuple[Any, nn.Module]:
    local = Path(model_ref).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref, revision=None if local else revision, local_files_only=local
    )
    try:
        model = AutoModel.from_pretrained(
            model_ref,
            revision=None if local else revision,
            local_files_only=local,
            add_pooling_layer=False,
        )
    except TypeError as error:
        if "unexpected keyword argument 'add_pooling_layer'" not in str(error):
            raise
        model = AutoModel.from_pretrained(
            model_ref, revision=None if local else revision, local_files_only=local
        )
    model = model.to("cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._tiny_omni_actual_revision = getattr(model.config, "_commit_hash", None) or revision
    return tokenizer, model


def _extract_features(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: nn.Module,
    *,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    hidden_size = int(model.config.hidden_size)
    max_options = max(len(row["options"]) for row in records)
    output = np.zeros((len(records), max_options, hidden_size), dtype=np.float16)
    flat: list[tuple[int, int, str, str]] = []
    for row_index, row in enumerate(records):
        for option_index, option in enumerate(row["options"]):
            first, second = format_pair(row, option)
            flat.append((row_index, option_index, first, second))
    truncated = 0
    max_observed = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            batch = flat[start : start + batch_size]
            firsts = [item[2] for item in batch]
            seconds = [item[3] for item in batch]
            lengths = tokenizer(firsts, seconds, truncation=False, padding=False)["input_ids"]
            lengths = [len(value) for value in lengths]
            truncated += sum(value > max_length for value in lengths)
            max_observed = max(max_observed, max(lengths, default=0))
            encoded = tokenizer(
                firsts,
                seconds,
                truncation="only_first",
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            result = model(**encoded).last_hidden_state.float()
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (result * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
            if not torch.isfinite(pooled).all():
                raise ValueError("encoder returned non-finite candidate features")
            for local_index, (row_index, option_index, _, _) in enumerate(batch):
                output[row_index, option_index] = (
                    pooled[local_index].cpu().numpy().astype(np.float16)
                )
            del encoded, result, attention_mask, pooled
    elapsed = time.perf_counter() - started
    return output, {
        "candidate_pairs": len(flat),
        "truncated_candidate_pairs": truncated,
        "truncated_candidate_fraction": truncated / max(1, len(flat)),
        "max_untruncated_tokens": max_observed,
        "feature_extraction_seconds": elapsed,
        "hidden_size": hidden_size,
        "encoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "resolved_revision": getattr(model, "_tiny_omni_actual_revision", "UNKNOWN"),
        "model_identifier": getattr(model.config, "_name_or_path", "UNKNOWN"),
        "all_encoder_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
    }


def _make_tensors(
    features: np.ndarray, records: list[dict[str, Any]], indices: list[int]
) -> tuple[Tensor, Tensor, Tensor]:
    max_options = features.shape[1]
    x = torch.from_numpy(features[indices].astype(np.float32))
    target = torch.zeros((len(indices), max_options), dtype=torch.float32)
    mask = torch.zeros((len(indices), max_options), dtype=torch.bool)
    for output_index, record_index in enumerate(indices):
        row = records[record_index]
        count = len(row["options"])
        target[output_index, :count] = torch.tensor(row["target"], dtype=torch.float32)
        mask[output_index, :count] = True
    return x, target, mask


def _evaluate(
    scorer: CandidateScorer, x: Tensor, y: Tensor, mask: Tensor, batch_size: int = 256
) -> Tensor:
    outputs = []
    scorer.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            outputs.append(
                scorer(x[start : start + batch_size]).masked_fill(
                    ~mask[start : start + batch_size], -1e9
                )
            )
    return torch.cat(outputs)


def _fit(
    train: tuple[Tensor, Tensor, Tensor],
    valid: tuple[Tensor, Tensor, Tensor],
    valid_records: list[dict[str, Any]],
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[CandidateScorer, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    x_train, y_train, train_mask = train
    x_valid, y_valid, valid_mask = valid
    scorer = CandidateScorer(x_train.shape[-1])
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        scorer.train()
        permutation = torch.randperm(len(x_train), generator=generator)
        losses = []
        for start in range(0, len(permutation), batch_size):
            selected = permutation[start : start + batch_size]
            logits = scorer(x_train[selected]).masked_fill(~train_mask[selected], -1e9)
            loss = -(y_train[selected] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite readout training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        valid_logits = _evaluate(scorer, x_valid, y_valid, valid_mask)
        valid_metrics = _metrics(valid_logits, y_valid, valid_mask, valid_records)
        selector_loss = valid_metrics["macro_over_question_id"]["nll"]
        history.append(
            {
                "epoch": epoch,
                "train_soft_cross_entropy": float(np.mean(losses)),
                "validation_macro_soft_nll": selector_loss,
                "validation_macro_accuracy": valid_metrics["macro_over_question_id"]["accuracy"],
            }
        )
        if selector_loss < best_loss - 1e-7:
            best_loss = selector_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise ValueError("no validation-selected checkpoint was produced")
    scorer.load_state_dict(best_state)
    return scorer, {"history": history, "best_epoch": best_epoch, "best_state": best_state}


def run_encoder(
    name: str,
    model_ref: str,
    revision: str,
    records: list[dict[str, Any]],
    split_by_family: dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    tokenizer, encoder = _load_encoder(model_ref, revision)
    if args.max_length > int(encoder.config.max_position_embeddings):
        raise ValueError(f"max_length exceeds position limit for {name}")
    features, feature_meta = _extract_features(
        records, tokenizer, encoder, batch_size=args.encode_batch_size, max_length=args.max_length
    )
    del tokenizer, encoder
    train_indices = [
        i for i, row in enumerate(records) if split_by_family[row["family"]] == "train"
    ]
    valid_indices = [
        i for i, row in enumerate(records) if split_by_family[row["family"]] == "validation"
    ]
    train = _make_tensors(features, records, train_indices)
    valid = _make_tensors(features, records, valid_indices)
    valid_records = [records[index] for index in valid_indices]
    training_started = time.perf_counter()
    model, fit = _fit(
        train,
        valid,
        valid_records,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    training_seconds = time.perf_counter() - training_started
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = model_dir / "best-readout.pt"
    torch.save(fit["best_state"], checkpoint_path)
    reloaded = CandidateScorer(features.shape[-1])
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    selected_logits = _evaluate(model, *valid)
    reloaded_logits = _evaluate(reloaded, *valid)
    reload_delta = float((selected_logits.softmax(-1) - reloaded_logits.softmax(-1)).abs().max())
    if reload_delta > 1e-7:
        raise ValueError("readout save/reload probability mismatch")
    report_metrics = _metrics(reloaded_logits, valid[1], valid[2], valid_records)
    (model_dir / "validation-predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in report_metrics["predictions"]
        ),
        encoding="utf-8",
    )
    np.savez_compressed(model_dir / "frozen-option-features.npz", features=features)
    result = {
        "encoder_name": name,
        "model_ref": model_ref,
        "revision": revision,
        "encoder_parameters": feature_meta["encoder_parameters"],
        "readout_parameters": sum(parameter.numel() for parameter in reloaded.parameters()),
        "total_parameters_before_quantization": feature_meta["encoder_parameters"]
        + sum(parameter.numel() for parameter in reloaded.parameters()),
        "feature_extraction": feature_meta,
        "split_rows": {"train": len(train_indices), "validation": len(valid_indices)},
        "split_families": {
            split: sum(value == split for value in split_by_family.values())
            for split in ("train", "validation")
        },
        "training": {
            "epochs_completed": len(fit["history"]),
            "best_epoch": fit["best_epoch"],
            "history": fit["history"],
        },
        "training_seconds": training_seconds,
        "metrics": {key: value for key, value in report_metrics.items() if key != "predictions"},
        "reload_verification": {"status": "pass", "max_probability_absolute_delta": reload_delta},
        "artifacts": {
            "readout": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            },
            "features": {
                "path": str(model_dir / "frozen-option-features.npz"),
                "sha256": sha256_file(model_dir / "frozen-option-features.npz"),
                "bytes": (model_dir / "frozen-option-features.npz").stat().st_size,
            },
        },
    }
    (model_dir / "encoder-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = args_from_cli()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.encode_batch_size < 1:
        raise ValueError("training and encoding settings must be positive")
    if len({name for name, _, _ in args.encoder}) != len(args.encoder):
        raise ValueError("encoder names must be unique")
    if args.expected_train_sha256 != TRAIN_SHA256:
        raise ValueError("this experiment is pinned to the audited Open-Jev train parquet hash")
    if sha256_file(args.train_parquet) != args.expected_train_sha256:
        raise ValueError("train parquet SHA-256 mismatch")
    torch.set_num_threads(args.cpu_threads)
    started_utc = datetime.now(UTC).isoformat()
    raw_rows = pq.read_table(args.train_parquet).to_pylist()
    records = validate_rows(raw_rows)
    split = make_family_split(
        raw_rows, seed=args.seed, validation_fraction=args.validation_fraction
    )
    assert_split_integrity(records, split)
    train_families = {family for family, part in split.items() if part == "train"}
    validation_families = {family for family, part in split.items() if part == "validation"}
    train_hashes = {row["input_hash"] for row in records if split[row["family"]] == "train"}
    validation_hashes = {
        row["input_hash"] for row in records if split[row["family"]] == "validation"
    }
    if train_families & validation_families or train_hashes & validation_hashes:
        raise ValueError("train/validation family or normalized-content overlap")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    split_manifest = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "group_key": "metadata.document_family_id",
        "assignment": split,
        "source_train_sha256": sha256_file(args.train_parquet),
        "rows_loaded": len(records),
        "train_families": sorted(train_families),
        "validation_families": sorted(validation_families),
        "train_ids": sorted(row["id"] for row in records if split[row["family"]] == "train"),
        "validation_ids": sorted(
            row["id"] for row in records if split[row["family"]] == "validation"
        ),
        "normalized_content_overlap": 0,
    }
    (args.output_dir / "internal-train-family-split.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    for name, model_ref, revision in args.encoder:
        results.append(
            run_encoder(name, model_ref, revision, records, split, args, args.output_dir)
        )
    summary = {
        "status": "complete_internal_validation_only",
        "started_at_utc": started_utc,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "repository": "ZefanCai/Open-Jev",
            "revision": DATASET_REVISION,
            "configuration": "amount-extraction-control-v1",
            "publisher_partition_loaded": "train only",
            "train_parquet_sha256": sha256_file(args.train_parquet),
            "official_validation_loaded": False,
            "calibration_test_ood_loaded": False,
            "row_count": len(records),
            "unique_families": len(split),
            "question_id_counts": {
                split_name: {
                    question_id: sum(
                        split[row["family"]] == split_name and row["question_id"] == question_id
                        for row in records
                    )
                    for question_id in sorted({row["question_id"] for row in records})
                }
                for split_name in ("train", "validation")
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pyarrow": pyarrow.__version__,
            "device": "cpu",
            "cpu_threads": args.cpu_threads,
        },
        "config": {
            "seed": args.seed,
            "validation_fraction_of_train_families": args.validation_fraction,
            "max_length": args.max_length,
            "encoding_batch_size": args.encode_batch_size,
            "readout": "shared 128-hidden tanh option scorer with scalar logits",
            "optimizer": "AdamW lr=0.002 weight_decay=0.0001",
            "epochs_max": args.epochs,
            "early_stopping_patience": args.patience,
            "selection": "macro soft NLL across question_id cells on internal validation families",
        },
        "results": results,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "limitations": [
            "Open-Jev amount-control is synthetic English invoice data; it is not real-world "
            "multimodal quality evidence.",
            "Labels are programmatically derived source targets; scores indicate performance "
            "on this controlled corpus.",
            "Internal validation is drawn from publisher TRAIN families and selects checkpoints; "
            "it is not a blind test.",
            "This text-only experiment makes no claim about Japanese, image, audio, or video."
            " quality.",
        ],
    }
    (args.output_dir / "run-report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
