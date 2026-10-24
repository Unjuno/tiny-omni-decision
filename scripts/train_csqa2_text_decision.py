"""Fit the fixed Path A text Decision readout on CommonsenseQA 2.0 train/dev."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from train_frozen_text_probe import _extract_features, _metrics, _predict
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.dataset import iter_local_rows, normalize_jsonl, sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.schema import DatasetManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    return parser.parse_args()


def _load(path: Path, manifest_path: Path) -> list:
    manifest = DatasetManifest.model_validate(yaml.safe_load(manifest_path.read_text("utf-8")))
    return list(normalize_jsonl(iter_local_rows(path), manifest, "csqa2"))


def _normalized_question(example: object) -> str:
    return " ".join(example.question.casefold().split())


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.epochs != 8 or args.seed != 17:
        raise ValueError("the pre-registered run requires seed 17 and exactly 8 epochs")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    train, validation = _load(args.train, args.train_manifest), _load(
        args.validation, args.validation_manifest
    )
    train_ids = {example.source_record_id for example in train}
    validation_ids = {example.source_record_id for example in validation}
    train_questions = {_normalized_question(example) for example in train}
    validation_questions = {_normalized_question(example) for example in validation}
    if train_ids & validation_ids or train_questions & validation_questions:
        raise ValueError("train/dev ID or normalized-question overlap detected")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.model, local_files_only=True).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    feature_started = time.perf_counter()
    train_features, train_targets = _extract_features(
        train, tokenizer, encoder, device, args.feature_batch_size
    )
    validation_features, validation_targets = _extract_features(
        validation, tokenizer, encoder, device, args.feature_batch_size
    )
    feature_seconds = time.perf_counter() - feature_started
    feature_file = output / "frozen-features.pt"
    torch.save(
        {"train": train_features, "train_targets": train_targets,
         "validation": validation_features, "validation_targets": validation_targets},
        feature_file,
    )

    scorer = FrozenFeatureCandidateScorer(int(encoder.config.hidden_size) * 4).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    order_rng = random.Random(args.seed)
    history = []
    best_nll, best_state, best_epoch = float("inf"), None, None
    train_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_features)))
        order_rng.shuffle(order)
        losses = []
        scorer.train()
        for start in range(0, len(order), args.batch_questions):
            indices = order[start : start + args.batch_questions]
            rows = [train_features[index] for index in indices]
            targets = torch.tensor([train_targets[index] for index in indices])
            widths = [row.shape[0] for row in rows]
            scores = scorer(torch.cat(rows))
            padded = scores.new_full((len(rows), max(widths)), -1e4)
            offset = 0
            for row_index, width in enumerate(widths):
                padded[row_index, :width] = scores[offset : offset + width]
                offset += width
            loss = F.cross_entropy(padded, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        logits = _predict(scorer, validation_features, device)
        metrics = _metrics(logits, validation_targets)
        history.append({"epoch": epoch, "train_ce": sum(losses) / len(losses), **metrics})
        if metrics["nll"] < best_nll:
            best_nll, best_epoch = metrics["nll"], epoch
            best_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
    training_seconds = time.perf_counter() - train_started
    if best_state is None:
        raise RuntimeError("validation checkpoint selection produced no state")
    scorer.load_state_dict(best_state)
    logits = _predict(scorer, validation_features, device)
    metrics = _metrics(logits, validation_targets)
    predictions = []
    max_swap_difference = 0.0
    for example, features, logit in zip(validation, validation_features, logits, strict=True):
        swapped = _predict(scorer, [features.flip(0)], device)[0].flip(0)
        max_swap_difference = max(max_swap_difference, float((swapped - logit).abs().max()))
        probabilities = torch.softmax(logit.float(), dim=-1).tolist()
        predictions.append({
            "id": example.id,
            "source_record_id": example.source_record_id,
            "target": example.target,
            "prediction": example.options[int(logit.argmax())],
            "option_ids": example.options,
            "probabilities": dict(zip(example.options, probabilities, strict=True)),
        })
    if max_swap_difference > 1e-6:
        raise AssertionError("option permutation changed option-identity scores")

    torch.save(best_state, output / "best-readout.pt")
    prediction_path = output / "validation-predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    (output / "training-history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    model_weights = args.model / "model.safetensors"
    frozen_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    report = {
        "schema_version": 1,
        "status": "complete_validation_selected_text_decision_probe",
        "task": "CommonsenseQA 2.0 yes/no text-only decision, official train/dev",
        "scope_limit": (
            "Dev is used for checkpoint selection, not a blind audit; "
            "author groups unavailable."
        ),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_questions": args.batch_questions,
        "feature_batch_size": args.feature_batch_size,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "checkpoint_rule": "minimum official-dev NLL; no calibration or seed search",
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_label_counts": dict(Counter(example.target for example in train)),
        "validation_label_counts": dict(Counter(example.target for example in validation)),
        "train_dev_id_overlap": 0,
        "train_dev_normalized_question_overlap": 0,
        "train_unique_questions": len(train_questions),
        "validation_unique_questions": len(validation_questions),
        "train_file_sha256": sha256_file(args.train),
        "validation_file_sha256": sha256_file(args.validation),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
        "model_revision": args.revision,
        "model_weights_sha256": sha256_file(model_weights),
        "runner_sha256": sha256_file(Path(__file__)),
        "feature_cache_sha256": sha256_file(feature_file),
        "feature_cache_bytes": feature_file.stat().st_size,
        "device": "cpu",
        "encoder_parameters_frozen": frozen_parameters,
        "trainable_readout_parameters": trainable_parameters,
        "readout": "MLP([context, option, |difference|, product]->128 GELU->1); option softmax",
        "feature_extraction_seconds": feature_seconds,
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "history": history,
        "best_validation": metrics,
        "option_order_max_logit_delta": max_swap_difference,
        "prediction_file_sha256": sha256_file(prediction_path),
    }
    _write_json(output / "run-report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
