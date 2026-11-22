"""Compare frozen Japanese encoders on a train-only Typed Decisions split.

The publisher's public test split is never loaded by this entry point. Labels
are teacher distributions over translated states, so metrics measure agreement
with that teacher rather than factual correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.tensor_features import compact_cls_features
from tiny_omni_decision.token_contracts import (
    embedding_vocab_padding,
    validate_token_id_bounds,
)
from tiny_omni_decision.typed_decisions_ja import (
    TypedDecisionChoiceGroup,
    normalized_state_sha256,
    parse_typed_decisions_ja_row,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_encoder(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("encoder must be NAME=LOCAL_DIRECTORY")
    return name, Path(raw_path)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--encoder", action="append", type=_parse_encoder, required=True)
    parser.add_argument(
        "--trust-remote-tokenizer-code",
        action="store_true",
        help="Execute reviewed local custom tokenizer code; model weights remain local-only.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--smoke-states-per-workflow", type=int, default=0)
    return parser.parse_args()


def _rows_and_groups(
    parquet_path: Path, assignments_path: Path, smoke_states: int
) -> tuple[list[TypedDecisionChoiceGroup], dict[str, int]]:
    frame = pd.read_parquet(parquet_path)
    if set(frame["split"].astype(str)) != {"train"}:
        raise ValueError("source file contains a non-train row; refusing test access")
    assignments = [
        json.loads(line) for line in assignments_path.read_text(encoding="utf-8").splitlines()
    ]
    split_by_id = {str(item["id"]): str(item["split"]) for item in assignments}
    if len(split_by_id) != len(assignments) or len(split_by_id) != len(frame):
        raise ValueError("split assignments must cover every unique train row exactly once")
    all_groups: list[TypedDecisionChoiceGroup] = []
    seen_state_hashes: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        sample_id = str(row["id"])
        if sample_id not in split_by_id:
            raise ValueError("train row is missing a frozen split assignment")
        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
        state_hash = normalized_state_sha256(state)
        existing = seen_state_hashes.setdefault(state_hash, split_by_id[sample_id])
        if existing != split_by_id[sample_id]:
            raise ValueError("normalized state content crosses train/validation")
        all_groups.extend(parse_typed_decisions_ja_row(row))

    if smoke_states:
        selected_ids: set[str] = set()
        for workflow in sorted({group.workflow for group in all_groups}):
            for split in ("train", "validation"):
                limit = smoke_states if split == "train" else max(1, smoke_states // 4)
                ids = sorted(
                    {
                        group.sample_id
                        for group in all_groups
                        if group.workflow == workflow and split_by_id[group.sample_id] == split
                    }
                )
                selected_ids.update(ids[:limit])
        all_groups = [group for group in all_groups if group.sample_id in selected_ids]
    return all_groups, split_by_id


def _texts(group: TypedDecisionChoiceGroup) -> list[tuple[str, str]]:
    prompt = f"Workflow: {group.workflow}\nQuestion type: {group.question_type}\n"
    prompt += f"Question: {group.instructions}\nChoose the best candidate."
    return [
        (
            group.state_json,
            f"{prompt}\nCandidate {label}: {description}",
        )
        for label, description in zip(
            group.option_labels, group.option_descriptions, strict=True
        )
    ]


def _encode(
    encoder_dir: Path,
    tokenizer_dir: Path,
    groups: list[TypedDecisionChoiceGroup],
    *,
    encode_batch_size: int,
    max_length: int,
    cpu_threads: int,
    trust_remote_tokenizer_code: bool,
) -> tuple[Tensor, list[dict[str, Any]], dict[str, Any]]:
    torch.set_num_threads(cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=trust_remote_tokenizer_code,
    )
    try:
        encoder = AutoModel.from_pretrained(
            encoder_dir, local_files_only=True, add_pooling_layer=False
        )
    except TypeError as error:
        if "unexpected keyword argument 'add_pooling_layer'" not in str(error):
            raise
        # Some encoder families, including DistilBERT, do not implement a pooler.
        encoder = AutoModel.from_pretrained(encoder_dir, local_files_only=True)
    encoder = encoder.to("cpu").eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if max_length > int(encoder.config.max_position_embeddings):
        raise ValueError("max_length exceeds the encoder's position limit")
    vocab_size = int(encoder.config.vocab_size)
    tokenizer_vocab_size = int(tokenizer.vocab_size)
    unused_embedding_rows = embedding_vocab_padding(tokenizer_vocab_size, vocab_size)

    group_indices: list[list[int]] = []
    pair_texts: list[tuple[str, str]] = []
    cursor = 0
    for group in groups:
        pairs = _texts(group)
        group_indices.append(list(range(cursor, cursor + len(pairs))))
        pair_texts.extend(pairs)
        cursor += len(pairs)

    output: list[Tensor] = []
    truncated_pairs = 0
    max_untruncated_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(pair_texts), encode_batch_size):
            batch = pair_texts[start : start + encode_batch_size]
            first = [item[0] for item in batch]
            second = [item[1] for item in batch]
            full = tokenizer(first, second, truncation=False, padding=False)
            lengths = [len(ids) for ids in full["input_ids"]]
            truncated_pairs += sum(length > max_length for length in lengths)
            max_untruncated_tokens = max(max_untruncated_tokens, max(lengths, default=0))
            encoded = tokenizer(
                first,
                second,
                truncation="only_first",
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            token_ids = encoded["input_ids"]
            validate_token_id_bounds(int(token_ids.min()), int(token_ids.max()), vocab_size)
            last_hidden = encoder(**encoded).last_hidden_state
            # On CPU, a CLS slice is a view into the full [batch, seq, hidden]
            # allocation. Clone it so the feature cache cannot retain every
            # batch's full token activations.
            hidden = compact_cls_features(last_hidden)
            if not torch.isfinite(hidden).all():
                raise ValueError("encoder returned non-finite CLS features")
            output.append(hidden)
            del last_hidden, hidden, encoded, full
    elapsed = time.perf_counter() - started
    feature_rows = torch.cat(output)
    del output
    if feature_rows.shape[0] != len(pair_texts):
        raise ValueError("feature count does not match candidate count")
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise ValueError("frozen encoder unexpectedly accumulated gradients")
    metadata = {
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "hidden_size": int(encoder.config.hidden_size),
        "encoder_vocabulary_size": vocab_size,
        "tokenizer_vocabulary_size": tokenizer_vocab_size,
        "unused_embedding_rows": unused_embedding_rows,
        "max_position_embeddings": int(encoder.config.max_position_embeddings),
        "candidate_pairs": len(pair_texts),
        "truncated_candidate_pairs": truncated_pairs,
        "truncated_candidate_fraction": truncated_pairs / max(1, len(pair_texts)),
        "max_untruncated_tokens": max_untruncated_tokens,
        "feature_extraction_seconds": elapsed,
        "frozen_parameters": all(not parameter.requires_grad for parameter in encoder.parameters()),
        "tokenizer_class": type(tokenizer).__name__,
        "trust_remote_tokenizer_code": trust_remote_tokenizer_code,
    }
    del encoder, tokenizer
    return feature_rows, [{"candidate_indices": indices} for indices in group_indices], metadata


def _padded_arrays(
    features: Tensor, groups: list[TypedDecisionChoiceGroup], split_by_id: dict[str, str]
) -> dict[str, tuple[Tensor, Tensor, Tensor, list[TypedDecisionChoiceGroup]]]:
    max_options = max(len(group.option_labels) for group in groups)
    feature_size = int(features.shape[-1])
    result = {}
    for split in ("train", "validation"):
        selected = [group for group in groups if split_by_id[group.sample_id] == split]
        X = torch.zeros((len(selected), max_options, feature_size), dtype=torch.float32)
        Y = torch.zeros((len(selected), max_options), dtype=torch.float32)
        M = torch.zeros((len(selected), max_options), dtype=torch.bool)
        feature_cursor = 0
        for index, group in enumerate(groups):
            if split_by_id[group.sample_id] != split:
                feature_cursor += len(group.option_labels)
                continue
            n = len(group.option_labels)
            X[len([g for g in groups[:index] if split_by_id[g.sample_id] == split])] = features[
                feature_cursor : feature_cursor + n
            ]
            out_index = sum(
                split_by_id[g.sample_id] == split for g in groups[:index]
            )
            Y[out_index, :n] = torch.tensor(group.target_probabilities)
            M[out_index, :n] = True
            feature_cursor += n
        result[split] = (X, Y, M, selected)
    return result


class CandidateScorer(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(feature_size, 128)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(128, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(self.dropout(torch.tanh(self.hidden(features)))).squeeze(-1)


def _metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    groups: list[TypedDecisionChoiceGroup],
) -> dict[str, Any]:
    records = []
    cell_records: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for index, group in enumerate(groups):
        count = len(group.option_labels)
        log_probs = F.log_softmax(logits[index, :count].float(), dim=-1)
        probabilities = log_probs.exp()
        target = targets[index, :count].float()
        hard_index = group.option_labels.index(group.hard_target)
        prediction_index = int(probabilities.argmax())
        confidence = float(probabilities[prediction_index])
        expected_correctness = float(target[prediction_index])
        cell_records[(group.workflow, group.question_type)].append(
            {
                "accuracy": float(prediction_index == hard_index),
                "soft_nll": float(-(target * log_probs).sum()),
                "brier": float((probabilities - target).square().sum()),
                "confidence": confidence,
                "expected_correctness": expected_correctness,
                "kl": float(
                    (target * (torch.log(target.clamp_min(1e-12)) - log_probs)).sum()
                ),
            }
        )
        records.append(
            {
                "sample_id": group.sample_id,
                "workflow": group.workflow,
                "question_key": group.question_key,
                "question_type": group.question_type,
                "back_translation_match": group.back_translation_match,
                "target": group.hard_target,
                "prediction": group.option_labels[prediction_index],
                "gold_probabilities": dict(
                    zip(group.option_labels, group.target_probabilities, strict=True)
                ),
                "probabilities": dict(
                    zip(group.option_labels, probabilities.tolist(), strict=True)
                ),
            }
        )

    cells: dict[str, Any] = {}
    for (workflow, question_type), values in sorted(cell_records.items()):
        count = len(values)
        ece_bins: list[list[tuple[float, float]]] = [[] for _ in range(15)]
        for value in values:
            index = min(int(value["confidence"] * 15), 14)
            ece_bins[index].append(
                (value["confidence"], value["expected_correctness"])
            )
        ece = sum(
            len(items)
            / count
            * abs(
                sum(conf for conf, _ in items) / len(items)
                - sum(acc for _, acc in items) / len(items)
            )
            for items in ece_bins
            if items
        )
        cells[f"{workflow}/{question_type}"] = {
            "count": count,
            "accuracy_vs_teacher_mode": sum(v["accuracy"] for v in values) / count,
            "soft_nll_vs_teacher_distribution": sum(v["soft_nll"] for v in values) / count,
            "soft_brier_vs_teacher_distribution": sum(v["brier"] for v in values) / count,
            "soft_kl_vs_teacher_distribution": sum(v["kl"] for v in values) / count,
            "soft_expected_correctness_ece_15": ece,
        }
    macro = {
        key: sum(cell[key] for cell in cells.values()) / len(cells)
        for key in (
            "accuracy_vs_teacher_mode",
            "soft_nll_vs_teacher_distribution",
            "soft_brier_vs_teacher_distribution",
            "soft_kl_vs_teacher_distribution",
            "soft_expected_correctness_ece_15",
        )
    }
    return {
        "decision_count": len(groups),
        "cell_count": len(cells),
        "macro_over_workflow_question_type_cells": macro,
        "by_workflow_and_question_type": cells,
        "predictions": records,
    }


def _fit(
    arrays: dict[str, tuple[Tensor, Tensor, Tensor, list[TypedDecisionChoiceGroup]]],
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[CandidateScorer, dict[str, Any]]:
    train_x, train_y, train_mask, train_groups = arrays["train"]
    val_x, val_y, val_mask, val_groups = arrays["validation"]
    torch.manual_seed(seed)
    scorer = CandidateScorer(int(train_x.shape[-1]))
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    best_nll, best_epoch, best_state = float("inf"), 0, None
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_x)))
        rng.shuffle(order)
        scorer.train()
        train_losses = []
        for start in range(0, len(order), batch_size):
            indices = torch.tensor(order[start : start + batch_size], dtype=torch.long)
            logits = scorer(train_x[indices])
            logits = logits.masked_fill(~train_mask[indices], -1e9)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(train_y[indices] * log_probs).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        scorer.eval()
        with torch.inference_mode():
            val_logits = scorer(val_x).masked_fill(~val_mask, -1e9)
            val_metrics = _metrics(val_logits, val_y, val_mask, val_groups)
        val_nll = val_metrics["macro_over_workflow_question_type_cells"][
            "soft_nll_vs_teacher_distribution"
        ]
        history.append(
            {
                "epoch": epoch,
                "train_soft_ce": sum(train_losses) / len(train_losses),
                "validation_macro_soft_nll": val_nll,
                "validation_macro_accuracy": val_metrics[
                    "macro_over_workflow_question_type_cells"
                ]["accuracy_vs_teacher_mode"],
            }
        )
        if val_nll < best_nll - 1e-10:
            best_nll = val_nll
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("validation did not select a checkpoint")
    final_state = {key: value.detach().clone() for key, value in scorer.state_dict().items()}
    scorer.load_state_dict(best_state)
    scorer.eval()
    with torch.inference_mode():
        best_train_logits = scorer(train_x).masked_fill(~train_mask, -1e9)
        best_val_logits = scorer(val_x).masked_fill(~val_mask, -1e9)
        best_train_metrics = _metrics(best_train_logits, train_y, train_mask, train_groups)
        best_val_metrics = _metrics(best_val_logits, val_y, val_mask, val_groups)
    return scorer, {
        "best_epoch_by_validation_macro_soft_nll": best_epoch,
        "epochs_completed": len(history),
        "history": history,
        "best_train_metrics": best_train_metrics,
        "best_validation_metrics": best_val_metrics,
        "best_state": best_state,
        "final_state": final_state,
    }


def _model_checkpoint(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = model_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"no pinned model weight file in {model_dir}")


def _run_encoder(
    name: str,
    model_dir: Path,
    args: argparse.Namespace,
    groups: list[TypedDecisionChoiceGroup],
    split_by_id: dict[str, str],
    started_utc: str,
) -> dict[str, Any]:
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError(f"encoder must be a local, concrete directory: {model_dir}")
    model_dir = model_dir.resolve()
    weight_path = _model_checkpoint(model_dir)
    feature_started = time.perf_counter()
    features, group_feature_map, encoder_meta = _encode(
        model_dir,
        args.tokenizer_dir.resolve(),
        groups,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
        cpu_threads=args.cpu_threads,
        trust_remote_tokenizer_code=args.trust_remote_tokenizer_code,
    )
    feature_seconds = time.perf_counter() - feature_started
    rebuilt_groups = []
    feature_cursor = 0
    for group in groups:
        n = len(group.option_labels)
        rebuilt_groups.append(
            {
                "group": group,
                "candidate_indices": list(range(feature_cursor, feature_cursor + n)),
            }
        )
        feature_cursor += n
    if feature_cursor != len(features) or len(group_feature_map) != len(groups):
        raise ValueError("candidate feature mapping is inconsistent")
    max_options = max(len(group.option_labels) for group in groups)
    hidden_size = int(features.shape[-1])
    array_sets = {}
    for split in ("train", "validation"):
        items = [item for item in rebuilt_groups if split_by_id[item["group"].sample_id] == split]
        X = torch.zeros((len(items), max_options, hidden_size), dtype=torch.float32)
        Y = torch.zeros((len(items), max_options), dtype=torch.float32)
        M = torch.zeros((len(items), max_options), dtype=torch.bool)
        selected_groups = []
        for index, item in enumerate(items):
            group = item["group"]
            candidate_indices = item["candidate_indices"]
            n = len(candidate_indices)
            X[index, :n] = features[candidate_indices]
            Y[index, :n] = torch.tensor(group.target_probabilities)
            M[index, :n] = True
            selected_groups.append(group)
        array_sets[split] = (X, Y, M, selected_groups)

    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"features": features}, run_dir / "frozen-candidate-features.pt")
    training_started = time.perf_counter()
    scorer, fit = _fit(
        array_sets,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    training_seconds = time.perf_counter() - training_started
    torch.save(fit["best_state"], run_dir / "best-readout.pt")
    torch.save(fit["final_state"], run_dir / "final-readout.pt")

    # Reload the selected readout and prove exact probability replay from the frozen cache.
    reloaded = CandidateScorer(hidden_size)
    checkpoint = torch.load(
        run_dir / "best-readout.pt", map_location="cpu", weights_only=True
    )
    reloaded.load_state_dict(checkpoint)
    reloaded.eval()
    val_x, val_y, val_mask, val_groups = array_sets["validation"]
    with torch.inference_mode():
        before = scorer(val_x).masked_fill(~val_mask, -1e9).softmax(dim=-1)
        after = reloaded(val_x).masked_fill(~val_mask, -1e9).softmax(dim=-1)
    max_probability_delta = float((before - after).abs().max())
    if max_probability_delta > 1e-7:
        raise ValueError("best readout save/reload probability mismatch")
    val_metrics = _metrics(
        reloaded(val_x).masked_fill(~val_mask, -1e9), val_y, val_mask, val_groups
    )
    prediction_path = run_dir / "validation-predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in val_metrics["predictions"]
        ),
        encoding="utf-8",
    )
    head_params = sum(parameter.numel() for parameter in scorer.parameters())
    report = {
        "status": "complete_train_only_internal_development_validation",
        "started_at_utc": started_utc,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "model_name": name,
        "model_directory": str(model_dir),
        "model_weights_file": weight_path.name,
        "model_weights_bytes": weight_path.stat().st_size,
        "model_weights_sha256": sha256_file(weight_path),
        "encoder_parameters": encoder_meta["encoder_parameters"],
        "head_parameters": head_params,
        "total_parameters_before_quantization": encoder_meta["encoder_parameters"] + head_params,
        "head": "shared 128-hidden tanh candidate scorer with scalar logits",
        "encoder": "frozen AutoModel; CLS token output; pooler and MLM/SPLADE head omitted",
        "data": {
            "train_parquet_sha256": sha256_file(args.train_parquet),
            "split_assignments_sha256": sha256_file(args.split_assignments),
            "publisher_test_loaded": False,
            "train_state_groups": len(
                {g.sample_id for g in groups if split_by_id[g.sample_id] == "train"}
            ),
            "validation_state_groups": len(
                {g.sample_id for g in groups if split_by_id[g.sample_id] == "validation"}
            ),
            "train_decisions": len(array_sets["train"][3]),
            "validation_decisions": len(array_sets["validation"][3]),
            "split_by_workflow": {
                workflow: {
                    split: len(
                        {
                            g.sample_id
                            for g in groups
                            if g.workflow == workflow and split_by_id[g.sample_id] == split
                        }
                    )
                    for split in ("train", "validation")
                }
                for workflow in sorted({g.workflow for g in groups})
            },
            "translation_match_counts": {
                split: {
                    str(flag): sum(
                        g.back_translation_match
                        == (str(flag).lower() == "true")
                        for g in groups
                        if split_by_id[g.sample_id] == split
                    )
                    for flag in ("true", "false")
                }
                for split in ("train", "validation")
            },
        },
        "config": {
            "seed": args.seed,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "encode_batch_size": args.encode_batch_size,
            "max_length": args.max_length,
            "lr": 2e-3,
            "weight_decay": 1e-4,
            "selection": "validation macro soft NLL over workflow/question-type cells",
        },
        "runtime": {
            "device": "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "cpu_threads": args.cpu_threads,
            "feature_extraction_seconds": feature_seconds,
            "training_seconds": training_seconds,
            **encoder_meta,
        },
        "checkpoint_selection": {
            "best_epoch": fit["best_epoch_by_validation_macro_soft_nll"],
            "epochs_completed": fit["epochs_completed"],
            "history": fit["history"],
            "best_train_metrics": fit["best_train_metrics"],
            "best_validation_metrics": val_metrics,
        },
        "reload_verification": {
            "status": "pass",
            "max_probability_absolute_delta": max_probability_delta,
            "tolerance": 1e-7,
        },
        "caveats": [
            "Metrics are agreement with an English-state 4B-class teacher distribution, "
            "not human ground truth.",
            "Validation groups were drawn only from publisher train; public test was not loaded.",
            "All workflows and question schemas occur in both partitions; "
            "this is not workflow OOD.",
        ],
    }
    report_path = run_dir / "run-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.epochs < 1 or args.patience < 1 or args.max_length < 16:
        raise ValueError("epochs, patience, and max length must be positive")
    if len({name for name, _ in args.encoder}) != len(args.encoder):
        raise ValueError("encoder names must be unique")
    if args.smoke_states_per_workflow < 0:
        raise ValueError("smoke subset size cannot be negative")
    if args.dataset_revision != "01a4558e83bf5a748094239df843d658b33d2533":
        raise ValueError("dataset revision is not the audited pin")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    groups, split_by_id = _rows_and_groups(
        args.train_parquet,
        args.split_assignments,
        args.smoke_states_per_workflow,
    )
    if not groups or not any(split_by_id[group.sample_id] == "validation" for group in groups):
        raise ValueError("both fixed train and validation groups are required")
    train_ids = {g.sample_id for g in groups if split_by_id[g.sample_id] == "train"}
    valid_ids = {g.sample_id for g in groups if split_by_id[g.sample_id] == "validation"}
    if train_ids & valid_ids:
        raise ValueError("state id overlap across internal train and validation")
    args.output_dir.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    run_reports = []
    for name, model_dir in args.encoder:
        run_reports.append(
            _run_encoder(name, model_dir, args, groups, split_by_id, started)
        )
    summary = {
        "status": (
            "complete_train_only_smoke_comparison"
            if args.smoke_states_per_workflow
            else "complete_train_only_internal_development_comparison"
        ),
        "smoke_states_per_workflow": args.smoke_states_per_workflow,
        "dataset_revision": args.dataset_revision,
        "publisher_test_loaded": False,
        "encoders": [
            {
                "name": report["model_name"],
                "total_parameters_before_quantization": report[
                    "total_parameters_before_quantization"
                ],
                "validation_macro": report["checkpoint_selection"]["best_validation_metrics"][
                    "macro_over_workflow_question_type_cells"
                ],
            }
            for report in run_reports
        ],
    }
    (args.output_dir / "comparison-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
