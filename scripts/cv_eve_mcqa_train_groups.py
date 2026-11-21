"""Fixed TRAIN-only grouped CV for the pinned EVE-ESA MCQA task."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.train_frozen_text_probe import _extract_features, _metrics, _predict
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.schema import DatasetManifest
from tiny_omni_decision.text_probe_data import load_generic_text_examples

SOURCE_REVISION = "f99c5779ca51e6d3407eeebe8297d5081af8a66b"
SOURCE_SHA256 = "9f711235b688491f27a0b484f6fc31bd3fba1ede261ffc2d500a73f896833509"
SOURCE_BYTES = 1_082_818
SOURCE_ROWS = 1_261
MODEL_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
MODEL_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
SEED = 17
FOLDS = 5
EPOCHS = 8
STOP_FRACTION = 0.20


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _norm(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in text if char.isalnum())


def _source_examples(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(path) != SOURCE_SHA256 or path.stat().st_size != SOURCE_BYTES:
        raise ValueError("pinned EVE-ESA TRAIN shard hash or byte length mismatch")
    table = parquet.read_table(path)
    if table.num_rows != SOURCE_ROWS or set(table.column_names) != {
        "Question",
        "Choices",
        "Answer",
    }:
        raise ValueError("pinned EVE-ESA shard has an unexpected row count or schema")
    rows = table.to_pylist()
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    seen: dict[tuple[str, str], int] = {}
    answer_positions: collections.Counter[int] = collections.Counter()
    option_counts: collections.Counter[int] = collections.Counter()
    invalid: dict[int, list[str]] = {}
    row_identities: list[str] = []
    for index, row in enumerate(rows):
        question = row.get("Question")
        options = row.get("Choices")
        answer = row.get("Answer")
        problems = []
        if not isinstance(question, str) or not question.strip():
            problems.append("missing_question")
            qnorm = ""
        else:
            qnorm = _norm(question)
        if not isinstance(options, list) or not 2 <= len(options) <= 20:
            problems.append("invalid_choice_list")
            options = options if isinstance(options, list) else []
        normalized_options = [
            _norm(option) if isinstance(option, str) else "__null_choice__" for option in options
        ]
        if any(not isinstance(option, str) or not option.strip() for option in options):
            problems.append("null_or_empty_choice")
        if len(set(normalized_options)) != len(options):
            problems.append("duplicate_normalized_choices")
        if not isinstance(answer, str) or len(answer) != 1 or not answer.isalpha():
            problems.append("invalid_answer_label")
            target_index = -1
        else:
            target_index = ord(answer.upper()) - ord("A")
            if not 0 <= target_index < len(options):
                problems.append("answer_label_outside_choice_range")
            elif (
                target_index < len(options)
                and normalized_options[target_index] == "__null_choice__"
            ):
                problems.append("answer_target_choice_missing")
        if problems:
            invalid[index] = sorted(set(problems))
        else:
            answer_positions[target_index] += 1
            option_counts[len(options)] += 1

        content = json.dumps(
            {"question": qnorm, "choices": sorted(normalized_options)},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for key in (("question", qnorm), ("content", content)):
            if key[1]:
                previous = seen.setdefault(key, index)
                union(index, previous)
        row_identities.append(
            json.dumps(
                {"question": qnorm, "choices": sorted(normalized_options)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    roots: dict[int, list[int]] = collections.defaultdict(list)
    for index in range(len(rows)):
        roots[find(index)].append(index)
    group_for_row: dict[int, str] = {}
    for members in roots.values():
        identities = []
        for index in members:
            identities.append(row_identities[index])
        group_id = _sha_text("\n".join(sorted(identities)))
        for index in members:
            group_for_row[index] = group_id

    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in invalid:
            continue
        target_index = ord(row["Answer"].upper()) - ord("A")
        group_id = group_for_row[index]
        content_id = _sha_text(
            json.dumps(
                {
                    "question": _norm(row["Question"]),
                    "choices": sorted(_norm(x) for x in row["Choices"]),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        prepared.append(
            {
                "id": f"eve-train-{index:05d}-{content_id[:12]}",
                "source_record_id": f"{group_id}:row-{index:05d}",
                "state": "",
                "question": row["Question"],
                "options": row["Choices"],
                "target": row["Choices"][target_index],
                "split": "train",
                "license": "CC-BY-4.0",
                "commercial_use": True,
                "derivative_model_training_allowed": True,
                "redistribution_allowed": True,
                "attribution": "EVE-ESA MCQA single-answer benchmark; cite Atrio et al. (2026).",
            }
        )

    groups = sorted({group_for_row[index] for index in range(len(rows)) if index not in invalid})
    group_members: dict[str, list[int]] = collections.defaultdict(list)
    for index, group_id in group_for_row.items():
        if index in invalid:
            continue
        group_members[group_id].append(index)
    assignment_order = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"eve-esa-mcqa-train-groupcv-v1|seed={SEED}|group={group}".encode()
        ).hexdigest(),
    )
    outer_fold = {group: index % FOLDS for index, group in enumerate(assignment_order)}
    prepared_source_row_indices = [index for index in range(len(rows)) if index not in invalid]
    assignment_rows = sorted(
        (
            prepared[position]["id"],
            group_for_row[source_index],
            outer_fold[group_for_row[source_index]],
        )
        for position, source_index in enumerate(prepared_source_row_indices)
    )
    components = [len(members) for members in group_members.values()]
    receipt = {
        "schema": "eve_mcqa_train_groupcv_preflight_v1",
        "source_revision": SOURCE_REVISION,
        "source_train_sha256": SOURCE_SHA256,
        "source_bytes": SOURCE_BYTES,
        "source_rows": SOURCE_ROWS,
        "structurally_valid_rows": len(prepared),
        "structurally_invalid_rows": len(invalid),
        "structurally_invalid_row_audit": [
            {
                "source_row_index_zero_based": index,
                "row_fingerprint_sha256": _sha_text(
                    json.dumps(
                        rows[index], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                ),
                "reasons": invalid[index],
                "excluded_before_model_features_or_fit": True,
            }
            for index in sorted(invalid)
        ],
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_SHA256,
        "seed": SEED,
        "folds": FOLDS,
        "split_policy": "label-blind SHA-256 ordering; balanced round-robin folds",
        "group_policy": "transitive exact question or question-and-choice-content identity",
        "split_scope": "source TRAIN only; no other source split loaded",
        "group_count": len(groups),
        "largest_group_sizes": sorted(components, reverse=True)[:12],
        "outer_fold_rows": [
            sum(outer_fold[group_for_row[i]] == fold for i in range(len(rows)) if i not in invalid)
            for fold in range(FOLDS)
        ],
        "outer_fold_groups": [
            sum(outer_fold[group] == fold for group in groups) for fold in range(FOLDS)
        ],
        "option_count_distribution": dict(sorted(option_counts.items())),
        "target_position_counts_zero_based": dict(sorted(answer_positions.items())),
        "normalized_question_duplicate_rows": len(prepared)
        - len({_norm(row["question"]) for row in prepared}),
        "normalized_content_duplicate_rows": len(prepared)
        - len(
            {
                json.dumps(
                    {"q": _norm(row["question"]), "o": sorted(_norm(x) for x in row["options"])},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for row in prepared
            }
        ),
        "id_order_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in prepared).encode()
        ).hexdigest(),
        "group_assignment_sha256": _sha_text(json.dumps(assignment_rows, separators=(",", ":"))),
        "prepared_source_row_indices": prepared_source_row_indices,
        "outer_fold_assignment": outer_fold,
        "group_for_row": group_for_row,
    }
    return prepared, receipt


def _inner_split(groups: list[str], outer_fold: int) -> tuple[set[str], set[str]]:
    remaining = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"eve-esa-mcqa-train-groupcv-v1|seed={SEED}|outer={outer_fold}|inner={group}".encode()
        ).hexdigest(),
    )
    stop_count = max(1, round(len(remaining) * STOP_FRACTION))
    stop = set(remaining[:stop_count])
    fit = set(remaining[stop_count:])
    if not fit or not stop or fit & stop:
        raise ValueError("nested inner fit/stop split is invalid")
    return fit, stop


def _train_one(
    features: list[Tensor],
    targets: list[int],
    fit_indices: list[int],
    stop_indices: list[int],
    *,
    hidden: int,
    device: torch.device,
    fold: int,
) -> tuple[nn.Module, list[dict[str, float | int]], int]:
    random.seed(SEED + fold)
    torch.manual_seed(SEED + fold)
    scorer = FrozenFeatureCandidateScorer(hidden * 4).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=0.001, weight_decay=0.0001)
    order_rng = random.Random(SEED + fold)
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_nll = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        order = fit_indices.copy()
        order_rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), 64):
            batch_indices = order[start : start + 64]
            examples = [features[index].to(device) for index in batch_indices]
            batch_targets = torch.tensor([targets[index] for index in batch_indices], device=device)
            widths = [item.shape[0] for item in examples]
            raw_scores = scorer(torch.cat(examples, dim=0))
            padded = raw_scores.new_full((len(examples), max(widths)), -1e4)
            offset = 0
            for row_index, width in enumerate(widths):
                padded[row_index, :width] = raw_scores[offset : offset + width]
                offset += width
            loss = F.cross_entropy(padded, batch_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in outer fold {fold}, epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        stop_logits = _predict(scorer, [features[index] for index in stop_indices], device)
        stop_targets = [targets[index] for index in stop_indices]
        stop_metrics = _metrics(stop_logits, stop_targets)
        history.append(
            {
                "epoch": epoch,
                "train_ce": sum(losses) / len(losses),
                "stop_accuracy": stop_metrics["accuracy"],
                "stop_nll": stop_metrics["nll"],
                "stop_brier": stop_metrics["brier"],
                "stop_ece_15": stop_metrics["ece_15_bins"],
            }
        )
        if stop_metrics["nll"] < best_nll:
            best_nll = float(stop_metrics["nll"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError(f"no inner-stop-selected state for outer fold {fold}")
    scorer.load_state_dict(best_state)
    return scorer, history, best_epoch


def _metric_arrays(probabilities: list[list[float]], targets: list[int]) -> dict[str, Any]:
    logits = [
        torch.tensor(row, dtype=torch.float32).clamp_min(1e-12).log() for row in probabilities
    ]
    return _metrics(logits, targets)


def _bootstrap(
    probabilities: list[list[float]],
    targets: list[int],
    group_ids: list[str],
) -> dict[str, Any]:
    unique_groups = sorted(set(group_ids))
    indices_by_group = {
        group: [i for i, value in enumerate(group_ids) if value == group] for group in unique_groups
    }
    rng = np.random.default_rng(SEED)
    accuracy = np.empty(10_000, dtype=np.float64)
    nll = np.empty(10_000, dtype=np.float64)
    brier = np.empty(10_000, dtype=np.float64)
    for replicate in range(10_000):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample_indices = [index for group in sampled for index in indices_by_group[str(group)]]
        metric = _metric_arrays(
            [probabilities[i] for i in sample_indices], [targets[i] for i in sample_indices]
        )
        accuracy[replicate] = metric["accuracy"]
        nll[replicate] = metric["nll"]
        brier[replicate] = metric["brier"]

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "replicates": 10_000,
        "seed": SEED,
        "unit": "transitive normalized question/content groups",
        "accuracy_95pct_percentile_ci": interval(accuracy),
        "nll_95pct_percentile_ci": interval(nll),
        "brier_95pct_percentile_ci": interval(brier),
    }


def main() -> None:
    args = _args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    if sha256_file(model_dir / "model.safetensors") != MODEL_SHA256:
        raise ValueError("local frozen MiniLM weights do not match the pinned revision")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    records, preflight = _source_examples(args.source_parquet.resolve())
    if len(records) < 1_000 or preflight["group_count"] < 200:
        raise ValueError("EVE-ESA TRAIN-only grouped CV lacks the minimum row/group support")
    base_manifest = DatasetManifest.model_validate(
        {
            "schema_version": 1,
            "dataset_id": "eve-esa/mcqa-single-answer",
            "revision": SOURCE_REVISION,
            "subset": "default",
            "split": "train",
            "modalities": ["text"],
            "upstream_url": "https://huggingface.co/datasets/eve-esa/mcqa-single-answer",
            "license": "CC-BY-4.0",
            "commercial_use": True,
            "derivative_model_training_allowed": True,
            "redistribution_allowed": True,
            "attribution": "EVE-ESA MCQA single-answer benchmark; cite Atrio et al. (2026).",
            "source_citation": "https://arxiv.org/abs/2604.13071",
            "usage": "training",
            "trust_status": "trusted",
            "schema": {
                "state": "state",
                "question": "question",
                "options": "options",
                "target": "target",
            },
            "notes": {
                "split_scope": "TRAIN-only grouped cross-validation",
                "source_sha256": SOURCE_SHA256,
            },
        }
    )
    effective_config = {
        "seed": SEED,
        "outer_folds": FOLDS,
        "inner_stop_fraction": STOP_FRACTION,
        "epochs": EPOCHS,
        "batch_questions": 64,
        "feature_batch_size": 64,
        "optimizer": "AdamW(lr=0.001,weight_decay=0.0001)",
        "readout": "FrozenFeatureCandidateScorer(hidden=128, GELU)",
        "option_order_augmentation": False,
        "inputs": ["question", "candidate_option"],
        "device": str(device),
        "max_length": 256,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_SHA256,
        "checkpoint_selector": "earliest minimum inner-stop NLL",
    }
    preflight.update(
        {
            "effective_config": effective_config,
            "effective_config_sha256": _sha_text(
                json.dumps(effective_config, sort_keys=True, separators=(",", ":"))
            ),
            "sealed_audit_accessed": False,
            "public_validation_or_test_accessed": False,
            "fitting_started": False,
        }
    )
    preflight_path = output / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Load the already-prepared TRAIN rows through the repository's generic Decision adapter.
    train_path = output / "train-only.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    examples = load_generic_text_examples(train_path, base_manifest)
    group_ids = [example.source_record_id.split(":", 1)[0] for example in examples]
    if len(examples) != len(records):
        raise ValueError("generic adapter changed the usable TRAIN row count")
    if len(set(group_ids)) != preflight["group_count"]:
        raise ValueError("group identity changed during generic normalization")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("frozen text encoder has trainable parameters")
    encoder_parameters = int(sum(parameter.numel() for parameter in encoder.parameters()))
    feature_started = time.perf_counter()
    features, targets = _extract_features(examples, tokenizer, encoder, device, 64, "legacy", 256)
    feature_seconds = time.perf_counter() - feature_started
    feature_path = output / "frozen-features.pt"
    torch.save({"features": features, "targets": targets}, feature_path)
    hidden = int(encoder.config.hidden_size)
    del encoder
    preflight["fitting_started"] = True
    preflight["feature_extraction_seconds"] = feature_seconds
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fold_assignment = preflight["outer_fold_assignment"]
    source_group_for_row = preflight["group_for_row"]
    group_for_row = [
        source_group_for_row[index] for index in preflight["prepared_source_row_indices"]
    ]
    groups = sorted(set(group_ids))
    oof_probabilities: list[list[float] | None] = [None] * len(examples)
    oof_fold: list[int | None] = [None] * len(examples)
    histories = []
    fold_reports = []
    outer_started = time.perf_counter()
    for fold in range(FOLDS):
        outer_validation_groups = {group for group in groups if int(fold_assignment[group]) == fold}
        outer_train_groups = set(groups) - outer_validation_groups
        fit_groups, stop_groups = _inner_split(sorted(outer_train_groups), fold)
        if (
            fit_groups & stop_groups
            or fit_groups & outer_validation_groups
            or stop_groups & outer_validation_groups
        ):
            raise ValueError("nested fit/stop/outer groups overlap")
        fit_indices = [i for i, group in enumerate(group_ids) if group in fit_groups]
        stop_indices = [i for i, group in enumerate(group_ids) if group in stop_groups]
        outer_indices = [i for i, group in enumerate(group_ids) if group in outer_validation_groups]
        scorer, history, best_epoch = _train_one(
            features,
            targets,
            fit_indices,
            stop_indices,
            hidden=hidden,
            device=device,
            fold=fold,
        )
        outer_logits = _predict(scorer, [features[i] for i in outer_indices], device)
        outer_probs = [torch.softmax(logit.float(), dim=-1).tolist() for logit in outer_logits]
        outer_targets = [targets[i] for i in outer_indices]
        outer_metrics = _metric_arrays(outer_probs, outer_targets)
        for index, probability in zip(outer_indices, outer_probs, strict=True):
            oof_probabilities[index] = probability
            oof_fold[index] = fold
        fold_reports.append(
            {
                "outer_fold": fold,
                "fit_examples": len(fit_indices),
                "inner_stop_examples": len(stop_indices),
                "outer_score_examples": len(outer_indices),
                "fit_groups": len(fit_groups),
                "inner_stop_groups": len(stop_groups),
                "outer_score_groups": len(outer_validation_groups),
                "selected_epoch_by_inner_stop_nll": best_epoch,
                "outer_metrics": outer_metrics,
                "history": history,
            }
        )
        histories.extend({"outer_fold": fold, **row} for row in history)
        torch.save(scorer.state_dict(), output / f"fold-{fold}-selected-readout.pt")
        del scorer
    training_seconds = time.perf_counter() - outer_started

    if any(item is None for item in oof_probabilities) or any(item is None for item in oof_fold):
        raise RuntimeError("OOF predictions do not cover all source TRAIN rows")
    probabilities = [item for item in oof_probabilities if item is not None]
    targets_int = [int(item) for item in targets]
    pooled = _metric_arrays(probabilities, targets_int)
    uniform_probabilities = [
        [1.0 / len(example.options)] * len(example.options) for example in examples
    ]
    uniform = _metric_arrays(uniform_probabilities, targets_int)
    # Hard-majority answer-position reference, retained separately from uniform probabilities.
    position_majority = [[0.0] * len(examples[i].options) for i in range(len(examples))]
    for fold in range(FOLDS):
        outer_groups = {group for group in groups if int(fold_assignment[group]) == fold}
        train_indices = [i for i, group in enumerate(group_ids) if group not in outer_groups]
        by_width: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
        for index in train_indices:
            by_width[len(examples[index].options)][targets[index]] += 1
        for index, example in enumerate(examples):
            if int(fold_assignment[group_for_row[index]]) != fold:
                continue
            counts = by_width[len(example.options)]
            chosen = min(range(len(example.options)), key=lambda pos: (-counts[pos], pos))
            position_majority[index] = [
                1.0 if pos == chosen else 0.0 for pos in range(len(example.options))
            ]
    position_majority_metrics = _metric_arrays(position_majority, targets_int)
    bootstrap = _bootstrap(probabilities, targets_int, group_ids)
    prediction_path = output / "oof-predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": example.id,
                    "group_id": group_ids[index],
                    "outer_fold": oof_fold[index],
                    "options": example.options,
                    "target": example.target,
                    "probabilities": probabilities[index],
                    "prediction": example.options[int(np.argmax(probabilities[index]))],
                },
                ensure_ascii=False,
            )
            + "\n"
            for index, example in enumerate(examples)
        ),
        encoding="utf-8",
    )
    result = {
        "schema": "eve_mcqa_train_groupcv_results_v1",
        "status": "complete_train_only_grouped_cv",
        "task_claim": "narrow text-only EO MCQA; not broad or multimodal evidence",
        "source_revision": SOURCE_REVISION,
        "source_train_sha256": SOURCE_SHA256,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_SHA256,
        "seed": SEED,
        "preflight_sha256": sha256_file(preflight_path),
        "effective_config_sha256": preflight["effective_config_sha256"],
        "source_train_examples": len(examples),
        "unique_content_groups": len(groups),
        "frozen_encoder_parameters": encoder_parameters,
        "trainable_readout_parameters": sum(
            parameter.numel() for parameter in FrozenFeatureCandidateScorer(hidden * 4).parameters()
        ),
        "feature_extraction_seconds": feature_seconds,
        "outer_fit_and_scoring_seconds": training_seconds,
        "device": str(device),
        "folds": fold_reports,
        "pooled_oof_metrics": pooled,
        "uniform_reference_metrics": uniform,
        "fold-local_answer-position_majority_metrics": position_majority_metrics,
        "group_bootstrap": bootstrap,
        "oof_predictions_sha256": sha256_file(prediction_path),
        "frozen_features_sha256": sha256_file(feature_path),
        "sealed_audit_accessed": False,
        "public_validation_or_test_accessed": False,
        "interpretation": "TRAIN-only grouped CV; no product or broad-modality claim follows.",
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
