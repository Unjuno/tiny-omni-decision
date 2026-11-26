"""Train a frozen MiniLM-L6 option scorer on the frozen Social IQa train split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Keep this diagnostic CPU-only even on machines where CUDA is installed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.socialiqa_readout import classification_metrics, validate_splits
from tiny_omni_decision.tensor_features import compact_cls_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]


class CandidateScorer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != 3:
            raise ValueError("features must have shape [batch, 3, hidden]")
        return self.network(features.reshape(-1, features.shape[-1])).reshape(-1, 3)


def _serialize(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (row["context"], f"{row['question']} Answer option: {option}") for option in row["options"]
    ]


def _extract_features(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    encoder: nn.Module,
    hidden_size: int,
    batch_questions: int,
    max_length: int,
    output_path: Path,
) -> dict[str, Any]:
    features = torch.empty((len(rows), 3, hidden_size), dtype=torch.float32, device="cpu")
    truncated = 0
    max_untruncated = 0
    started = time.perf_counter()
    encoder.eval()
    with torch.inference_mode():
        for row_start in range(0, len(rows), batch_questions):
            batch_rows = rows[row_start : row_start + batch_questions]
            pairs = [pair for row in batch_rows for pair in _serialize(row)]
            first = [pair[0] for pair in pairs]
            second = [pair[1] for pair in pairs]
            full = tokenizer(first, second, truncation=False, padding=False)
            lengths = [len(ids) for ids in full["input_ids"]]
            truncated += sum(length > max_length for length in lengths)
            max_untruncated = max(max_untruncated, max(lengths, default=0))
            encoded = tokenizer(
                first,
                second,
                truncation="only_first",
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            hidden = compact_cls_features(encoder(**encoded).last_hidden_state)
            if hidden.shape != (len(pairs), hidden_size) or not torch.isfinite(hidden).all():
                raise ValueError("encoder produced malformed or non-finite CLS features")
            features[row_start : row_start + len(batch_rows)] = hidden.reshape(
                len(batch_rows), 3, hidden_size
            )
            del full, encoded, hidden
    if truncated:
        raise ValueError(f"preflight mismatch: {truncated} candidate pairs were truncated")
    torch.save(features, output_path)
    result = {
        "questions": len(rows),
        "candidate_pairs": len(rows) * 3,
        "max_length": max_length,
        "truncated_pairs": truncated,
        "max_untruncated_tokens": max_untruncated,
        "feature_shape": list(features.shape),
        "feature_bytes": output_path.stat().st_size,
        "feature_sha256": _sha256(output_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del features
    return result


def _evaluate(
    model: CandidateScorer, features: Tensor, targets: Tensor
) -> tuple[dict[str, Any], Tensor, Tensor]:
    model.eval()
    with torch.inference_mode():
        logits = model(features)
    metrics = classification_metrics(logits, targets)
    return metrics, logits, torch.softmax(logits, dim=1)


def _train_readout(
    train_features: Tensor,
    train_targets: Tensor,
    validation_features: Tensor,
    validation_targets: Tensor,
    *,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    readout = config["readout"]
    training = config["training"]
    scorer = CandidateScorer(
        int(train_features.shape[-1]),
        int(readout["hidden_size"]),
        float(readout["dropout"]),
    ).cpu()
    trainable = sum(parameter.numel() for parameter in scorer.parameters())
    if trainable != int(readout["trainable_parameters"]):
        raise ValueError(f"readout parameter mismatch: {trainable}")
    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(training["seed"]))
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=int(training["batch_size_questions"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(training["max_epochs"]) + 1):
        scorer.train()
        train_loss_total = 0.0
        train_examples = 0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = scorer(batch_features)
            loss = F.cross_entropy(logits, batch_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            optimizer.step()
            count = int(batch_targets.shape[0])
            train_loss_total += float(loss.detach()) * count
            train_examples += count
        validation_metrics, _, _ = _evaluate(scorer, validation_features, validation_targets)
        train_loss = train_loss_total / train_examples
        record = {
            "epoch": epoch,
            "train_cross_entropy": train_loss,
            "validation": validation_metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record, separators=(",", ":")), flush=True)
        current_nll = float(validation_metrics["nll"])
        if current_nll < best_nll:
            best_nll = current_nll
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(training["patience"]):
                break
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint was selected")
    final_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
    torch.save(best_state, output_dir / "best-readout.pt")
    torch.save(final_state, output_dir / "final-readout.pt")
    scorer.load_state_dict(best_state, strict=True)
    selected_metrics, selected_logits, selected_probabilities = _evaluate(
        scorer, validation_features, validation_targets
    )
    _, _, reversed_probabilities = _evaluate(
        scorer, validation_features.flip(1), validation_targets
    )
    max_equivariance_error = float(
        (reversed_probabilities.flip(1) - selected_probabilities).abs().max()
    )
    if max_equivariance_error > 1e-6:
        raise ValueError(f"option permutation equivariance failed: {max_equivariance_error}")
    reloaded_scorer = CandidateScorer(
        int(train_features.shape[-1]),
        int(readout["hidden_size"]),
        float(readout["dropout"]),
    ).cpu()
    reloaded_scorer.load_state_dict(
        torch.load(output_dir / "best-readout.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    reloaded_metrics, reloaded_logits, reloaded_probabilities = _evaluate(
        reloaded_scorer, validation_features, validation_targets
    )
    reload_delta = max(
        float((reloaded_logits - selected_logits).abs().max()),
        float((reloaded_probabilities - selected_probabilities).abs().max()),
    )
    if reload_delta > 1e-6 or reloaded_metrics != selected_metrics:
        raise ValueError(f"best checkpoint reload mismatch: delta={reload_delta}")
    return {
        "readout_parameters": trainable,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "selected_validation_metrics": selected_metrics,
        "option_order_equivariance_max_probability_delta": max_equivariance_error,
        "best_checkpoint_reload_max_delta": reload_delta,
        "history": history,
        "selected_logits": selected_logits,
        "selected_probabilities": selected_probabilities,
        "selected_state": best_state,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    source = config["source"]
    model_config = config["model"]
    training = config["training"]
    if _sha256(args.train_jsonl) != source["train_file_sha256"]:
        raise ValueError("train JSONL hash mismatch")
    if _sha256(args.validation_jsonl) != source["validation_file_sha256"]:
        raise ValueError("validation JSONL hash mismatch")
    model_file = args.model_dir / "model.safetensors"
    if _sha256(model_file) != model_config["weights_sha256"]:
        raise ValueError("pretrained model weight hash mismatch")
    train_rows = _load_rows(args.train_jsonl)
    validation_rows = _load_rows(args.validation_jsonl)
    split_report = validate_splits(train_rows, validation_rows)
    args.output_dir.mkdir(parents=True)
    seed = int(training["seed"])
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    torch.set_num_threads(int(training["cpu_threads"]))
    started_at = datetime.now(UTC).isoformat()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    encoder = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to("cpu").eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    if encoder_parameters != int(model_config["encoder_parameters"]):
        raise ValueError(f"encoder parameter mismatch: {encoder_parameters}")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("frozen encoder acquired trainable parameters")
    feature_started = time.perf_counter()
    feature_train = _extract_features(
        train_rows,
        tokenizer=tokenizer,
        encoder=encoder,
        hidden_size=int(encoder.config.hidden_size),
        batch_questions=int(training["encoder_batch_size_questions"]),
        max_length=int(model_config["max_sequence_length"]),
        output_path=args.output_dir / "train-features.pt",
    )
    feature_validation = _extract_features(
        validation_rows,
        tokenizer=tokenizer,
        encoder=encoder,
        hidden_size=int(encoder.config.hidden_size),
        batch_questions=int(training["encoder_batch_size_questions"]),
        max_length=int(model_config["max_sequence_length"]),
        output_path=args.output_dir / "validation-features.pt",
    )
    feature_elapsed = time.perf_counter() - feature_started
    del encoder, tokenizer
    train_features = torch.load(
        args.output_dir / "train-features.pt", map_location="cpu", weights_only=True
    )
    validation_features = torch.load(
        args.output_dir / "validation-features.pt", map_location="cpu", weights_only=True
    )
    train_targets = torch.tensor([row["target"] for row in train_rows], dtype=torch.long)
    validation_targets = torch.tensor([row["target"] for row in validation_rows], dtype=torch.long)
    training_result = _train_readout(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        config=config,
        output_dir=args.output_dir,
    )
    prediction_path = args.output_dir / "validation-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(validation_rows):
            probabilities = training_result["selected_probabilities"][index].tolist()
            prediction = int(training_result["selected_logits"][index].argmax())
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "source_context_sha256": row["source_context_sha256"],
                        "target": row["target"],
                        "prediction": prediction,
                        "probabilities": probabilities,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    history_path = args.output_dir / "training-history.json"
    history_path.write_text(
        json.dumps(training_result["history"], indent=2) + "\n", encoding="utf-8"
    )
    artifact_files = [
        "best-readout.pt",
        "final-readout.pt",
        "train-features.pt",
        "validation-features.pt",
        "validation-predictions.jsonl",
        "training-history.json",
    ]
    report = {
        "status": "complete_train_only_grouped_validation_probe",
        "experiment_id": config["experiment_id"],
        "source_commit": os.environ.get("SOURCE_COMMIT", "UNKNOWN"),
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": _sha256(args.config),
        "dataset": {
            "train_sha256": _sha256(args.train_jsonl),
            "validation_sha256": _sha256(args.validation_jsonl),
            "official_dev_test_or_sealed_audit_opened": False,
            **split_report,
        },
        "model": {
            "id": model_config["id"],
            "revision": model_config["revision"],
            "weight_sha256": _sha256(model_file),
            "encoder_parameters": encoder_parameters,
            "readout_parameters": training_result["readout_parameters"],
            "total_parameters": encoder_parameters + training_result["readout_parameters"],
            "encoder_frozen": True,
        },
        "device": "cpu",
        "gpu_used": False,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "feature_extraction_seconds": feature_elapsed,
        "train_feature_extraction": feature_train,
        "validation_feature_extraction": feature_validation,
        "readout_training_seconds": training_result["elapsed_seconds"],
        "epochs_completed": training_result["epochs_completed"],
        "best_epoch": training_result["best_epoch"],
        "selection_metric": "minimum validation NLL",
        "selected_validation_metrics": training_result["selected_validation_metrics"],
        "option_order_equivariance_max_probability_delta": training_result[
            "option_order_equivariance_max_probability_delta"
        ],
        "best_checkpoint_reload_max_delta": training_result["best_checkpoint_reload_max_delta"],
        "test_policy": "No official dev/test or project sealed audit was opened.",
        "artifact_sha256": {name: _sha256(args.output_dir / name) for name in artifact_files},
        "artifact_bytes": {
            name: (args.output_dir / name).stat().st_size for name in artifact_files
        },
    }
    (args.output_dir / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "best_epoch": report["best_epoch"],
                "selected_validation_metrics": report["selected_validation_metrics"],
                "option_order_equivariance_max_probability_delta": report[
                    "option_order_equivariance_max_probability_delta"
                ],
                "output_dir": str(args.output_dir),
                "run_report_sha256": _sha256(args.output_dir / "run-report.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
