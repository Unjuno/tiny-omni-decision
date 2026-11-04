"""Extract frozen MiniLM TRAIN features and run fixed QASC evidence-context grouped CV."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from tiny_omni_decision.qasc import _normalized
from tiny_omni_decision.qasc_cv import (
    audit_qasc_train_folds,
    build_qasc_train_folds,
    qasc_train_component_ids,
)
from tiny_omni_decision.schema import DatasetManifest

EXPECTED_FEATURE_MODEL_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
SEED = 17
EPOCHS = 8
BATCH_QUESTIONS = 64
FEATURE_SIZE = 1536
HIDDEN_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fold_plan_matches(
    rows: list[dict[str, Any]],
    assignment: list[int],
    reference_plan: Path,
    reference_assignment: Path,
) -> bool:
    plan = json.loads(reference_plan.read_text(encoding="utf-8"))
    if plan.get("folds") != 5 or plan.get("seed") != SEED:
        raise ValueError(
            "question-only fold plan does not match the fixed five-fold seed-17 design"
        )
    expected_fold_counts = {
        int(row["fold"]): int(row["rows"]) for row in plan.get("folds_report", [])
    }
    if set(expected_fold_counts) != set(range(5)):
        raise ValueError("question-only fold plan has incomplete fold metadata")
    reference_rows = _read_jsonl(reference_assignment)
    reference_ids = [str(row["id"]) for row in reference_rows]
    source_ids = [str(row["id"]) for row in rows]
    if len(set(reference_ids)) != len(reference_ids) or set(reference_ids) != set(source_ids):
        raise ValueError("question-only fold assignment does not cover the same unique source IDs")
    reference_by_id = {str(row["id"]): int(row["fold"]) for row in reference_rows}
    current_by_id = {str(row["id"]): int(fold) for row, fold in zip(rows, assignment, strict=True)}
    if current_by_id != reference_by_id:
        raise ValueError("evidence-context fold membership differs from question-only CV")
    actual_fold_counts = {
        fold: sum(assigned == fold for assigned in assignment) for fold in range(5)
    }
    if actual_fold_counts != expected_fold_counts:
        raise ValueError("question-only fold counts disagree with its assignment report")
    return True


def _paired_group_bootstrap(
    rows: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    replicates: int = 10_000,
) -> dict[str, Any]:
    import numpy as np

    grouped: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    component_ids = qasc_train_component_ids(rows)
    for source, current, old, component_id in zip(
        rows, candidate, reference, component_ids, strict=True
    ):
        if current["id"] != source["id"] or old["id"] != source["id"]:
            raise ValueError("paired OOF predictions do not align to the fixed source IDs")
        target = int(current["target_index"])
        if target != int(old["target_index"]) or current["fold"] != old["fold"]:
            raise ValueError("paired OOF target or fold identity differs")
        for prediction in (current, old):
            probabilities = prediction["probabilities"]
            if (
                len(probabilities) != 8
                or any(
                    not math.isfinite(float(value)) or float(value) < 0 for value in probabilities
                )
                or not math.isclose(sum(float(value) for value in probabilities), 1.0, abs_tol=1e-5)
            ):
                raise ValueError(
                    "paired OOF probabilities must be finite eight-choice distributions"
                )
        counts = grouped[component_id]
        counts[0] += float(
            int(current["prediction_index"] == target) - int(old["prediction_index"] == target)
        )
        counts[1] += -math.log(max(float(current["probabilities"][target]), 1e-12))
        counts[2] += -math.log(max(float(old["probabilities"][target]), 1e-12))
        counts[3] += 1.0
    groups = list(grouped.values())
    values = np.asarray(groups, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    differences = np.empty((replicates, 2), dtype=np.float64)
    group_count = len(values)
    for replicate in range(replicates):
        sampled = values[rng.integers(0, group_count, size=group_count)].sum(axis=0)
        differences[replicate, 0] = sampled[0] / sampled[3]
        differences[replicate, 1] = (sampled[1] - sampled[2]) / sampled[3]
    return {
        "unit": "connected_component_fact_pair_question_content",
        "unique_groups": group_count,
        "replicates": replicates,
        "seed": SEED,
        "percentile_95_ci_candidate_minus_question_only_accuracy": [
            float(x) for x in np.quantile(differences[:, 0], [0.025, 0.975])
        ],
        "percentile_95_ci_candidate_minus_question_only_nll": [
            float(x) for x in np.quantile(differences[:, 1], [0.025, 0.975])
        ],
    }


def main() -> None:
    import torch
    from torch.nn import functional as F
    from transformers import AutoModel, AutoTokenizer

    from scripts.run_qasc_train_cv import (
        _baseline_position_probabilities,
        _metrics,
        _predict,
    )
    from scripts.train_frozen_text_probe import _extract_features, _format_context, _load_examples
    from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--reference-fold-plan", type=Path, required=True)
    parser.add_argument("--reference-fold-assignment", type=Path, required=True)
    parser.add_argument("--question-only-predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    train_path = args.train_jsonl.resolve(strict=True)
    prep_path = args.preparation_report.resolve(strict=True)
    reference_plan_path = args.reference_fold_plan.resolve(strict=True)
    reference_assignment_path = args.reference_fold_assignment.resolve(strict=True)
    reference_predictions_path = args.question_only_predictions.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    weights_path = model_path / "model.safetensors"
    if _sha256(weights_path) != EXPECTED_FEATURE_MODEL_SHA256:
        raise ValueError("frozen MiniLM weight hash mismatch")
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if _sha256(train_path) != prep["evidence_train_sha256"]:
        raise ValueError("evidence-context TRAIN hash differs from preparation receipt")
    if prep.get("validation_or_test_opened") is not False:
        raise ValueError("preparation receipt does not affirm train-only source access")
    rows = _read_jsonl(train_path)
    if len(rows) != 8134 or any(
        row.get("split") != "train" or not row.get("state") for row in rows
    ):
        raise ValueError("expected 8,134 nonempty-state TRAIN rows")
    assignment, split_report = build_qasc_train_folds(rows, folds=5, seed=SEED)
    fold_audits = audit_qasc_train_folds(rows, assignment, folds=5)
    _fold_plan_matches(rows, assignment, reference_plan_path, reference_assignment_path)
    reference_predictions = _read_jsonl(reference_predictions_path)
    if len(reference_predictions) != len(rows):
        raise ValueError("question-only predictions do not cover all TRAIN rows")
    current_fold_by_id = {row["id"]: int(fold) for row, fold in zip(rows, assignment, strict=True)}
    for row, prediction in zip(rows, reference_predictions, strict=True):
        if (
            prediction.get("id") != row["id"]
            or int(prediction.get("target_index", -1)) != row["options"].index(row["target"])
            or int(prediction.get("fold", -1)) != current_fold_by_id[row["id"]]
        ):
            raise ValueError(
                "question-only OOF predictions do not match source ID, target, or fold"
            )

    manifest = DatasetManifest.model_validate(
        yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    examples = _load_examples(train_path, manifest, adapter="generic")
    expected_example_ids = [f"{manifest.dataset_id}:{row['id']}" for row in rows]
    if [example.id for example in examples] != expected_example_ids:
        raise ValueError("generic Decision adapter changed normalized source ID order")
    for example, row in zip(examples, rows, strict=True):
        if (
            example.question != row["question"]
            or example.options != row["options"]
            or example.target != row["target"]
            or example.state != row["state"]
        ):
            raise ValueError("generic Decision adapter changed evidence-context task fields")
    context_token_lengths = [
        len(
            tokenizer(_format_context(example), add_special_tokens=True, truncation=False)[
                "input_ids"
            ]
        )
        for example in examples
    ]
    preflight = {
        "rows": len(rows),
        "fold_sizes": [sum(fold == index for fold in assignment) for index in range(5)],
        "fold_plan_matches_question_only": True,
        "max_context_tokens": max(context_token_lengths),
        "contexts_over_256_tokens": sum(length > 256 for length in context_token_lengths),
        "model_sha256": EXPECTED_FEATURE_MODEL_SHA256,
        "validation_loaded": False,
        "test_loaded": False,
    }
    if args.dry_run:
        print(json.dumps(preflight, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is set")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence-context output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    feature_started = time.perf_counter()
    train_features, targets = _extract_features(
        examples,
        tokenizer,
        encoder,
        device,
        args.feature_batch_size,
        context_policy="legacy",
        option_max_length=256,
    )
    feature_seconds = time.perf_counter() - feature_started
    del encoder
    if len(train_features) != len(rows) or len(targets) != len(rows):
        raise ValueError("frozen feature extraction changed TRAIN row count")
    feature_path = output / "frozen-features.pt"
    torch.save({"train_features": train_features, "train_targets": targets}, feature_path)
    _write_json(
        output / "effective-config.json",
        {
            "experiment_id": "qasc-path-a-evidence-context-group-cv-seed17",
            "seed": SEED,
            "folds": 5,
            "epochs": EPOCHS,
            "batch_questions": BATCH_QUESTIONS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "model_revision": (
                "sentence-transformers/paraphrase-MiniLM-L3-v2@"
                "4ca70771034acceecb2e72475f72050fcdde4ddc"
            ),
            "model_parameters": 17_389_824,
            "readout_parameters": 196_865,
            "context_policy": "legacy masked-mean; state contains two QASC support facts",
            "input_factor_changed": "empty state -> question-specific QASC fact1/fact2 state",
            "checkpoint_rule": "fixed epoch 8 endpoint; no held-fold selection",
            "device_for_feature_extraction": str(device),
            "readout_training_device": "cpu",
            "train_rows": len(rows),
            "validation_loaded": False,
            "test_loaded": False,
            "sealed_audit_loaded": False,
        },
    )
    _write_json(output / "fold-plan.json", {**split_report, "fold_audits": fold_audits})
    assignment_path = output / "fold-assignment.jsonl"
    assignment_path.write_text(
        "".join(
            json.dumps({"id": row["id"], "fold": fold}) + "\n"
            for row, fold in zip(rows, assignment, strict=True)
        ),
        encoding="utf-8",
    )

    target_in_fact_ids = set()
    for row in rows:
        evidence_lines = row["state"].splitlines()
        if len(evidence_lines) != 3 or evidence_lines[0] != "Evidence:":
            raise ValueError("expected exactly two line-preserving QASC support facts")
        target = _normalized(row["target"])
        if target and any(target in _normalized(fact) for fact in evidence_lines[1:]):
            target_in_fact_ids.add(row["id"])
    predictions: list[dict[str, Any] | None] = [None] * len(rows)
    fold_results = []
    train_seconds = 0.0
    eval_seconds = 0.0
    total_updates = 0
    started = time.perf_counter()
    last_scorer = None
    for held_fold in range(5):
        train_indices = [i for i, fold in enumerate(assignment) if fold != held_fold]
        held_indices = [i for i, fold in enumerate(assignment) if fold == held_fold]
        torch.manual_seed(SEED)
        scorer = FrozenFeatureCandidateScorer(FEATURE_SIZE, HIDDEN_SIZE)
        optimizer = torch.optim.AdamW(
            scorer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        rng = random.Random(SEED)
        history = []
        fold_started = time.perf_counter()
        updates = 0
        for epoch in range(1, EPOCHS + 1):
            order = train_indices.copy()
            rng.shuffle(order)
            scorer.train()
            losses = []
            for start in range(0, len(order), BATCH_QUESTIONS):
                batch = order[start : start + BATCH_QUESTIONS]
                feature_rows = [train_features[i] for i in batch]
                logits = scorer(torch.cat(feature_rows))
                padded = logits.new_full((len(batch), 8), -1e4)
                for row_index, row_logits in enumerate(logits.split(8)):
                    padded[row_index] = row_logits
                batch_targets = torch.tensor([targets[i] for i in batch], dtype=torch.long)
                loss = F.cross_entropy(padded, batch_targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss fold={held_fold} epoch={epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is None or not torch.isfinite(p.grad).all() for p in scorer.parameters()
                ):
                    raise FloatingPointError(f"invalid gradient fold={held_fold} epoch={epoch}")
                optimizer.step()
                losses.append(float(loss.detach()))
                updates += 1
            history.append(
                {"epoch": epoch, "online_train_ce": sum(losses) / len(losses), "updates": updates}
            )
        fold_train_seconds = time.perf_counter() - fold_started
        train_seconds += fold_train_seconds
        checkpoint = output / f"fold-{held_fold}-final.pt"
        torch.save(
            {key: value.detach().cpu() for key, value in scorer.state_dict().items()}, checkpoint
        )
        eval_started = time.perf_counter()
        logits, order_indices = _predict(scorer, train_features, held_indices)
        pre_reload = torch.softmax(logits, dim=-1)
        reloaded = FrozenFeatureCandidateScorer(FEATURE_SIZE, HIDDEN_SIZE)
        reloaded.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        reload_logits, reload_indices = _predict(reloaded, train_features, held_indices)
        probabilities = torch.softmax(reload_logits, dim=-1)
        if not torch.equal(order_indices, reload_indices) or not torch.equal(
            pre_reload, probabilities
        ):
            raise AssertionError(f"save/reload prediction mismatch in fold {held_fold}")
        held_targets = torch.tensor([targets[i] for i in held_indices], dtype=torch.long)
        metrics = _metrics(probabilities, held_targets)
        baseline = _metrics(
            _baseline_position_probabilities(train_indices, held_indices, targets), held_targets
        )
        for local, source_index in enumerate(held_indices):
            target = targets[source_index]
            prediction_index = int(probabilities[local].argmax())
            predictions[source_index] = {
                "id": rows[source_index]["id"],
                "fold": held_fold,
                "target_index": target,
                "prediction_index": prediction_index,
                "target_option": rows[source_index]["options"][target],
                "prediction_option": rows[source_index]["options"][prediction_index],
                "target_in_fact": rows[source_index]["id"] in target_in_fact_ids,
                "probabilities": [float(value) for value in probabilities[local]],
            }
        fold_eval_seconds = time.perf_counter() - eval_started
        eval_seconds += fold_eval_seconds
        total_updates += updates
        fold_results.append(
            {
                "fold": held_fold,
                "train_rows": len(train_indices),
                "held_rows": len(held_indices),
                "epoch_history": history,
                "out_of_fold_metrics": metrics,
                "train_answer_position_prior": baseline,
                "checkpoint_rule": "fixed final epoch 8",
                "checkpoint_sha256": _sha256(checkpoint),
                "reload_probabilities_exact": True,
                "training_seconds": fold_train_seconds,
                "evaluation_seconds": fold_eval_seconds,
            }
        )
        last_scorer = reloaded
        print(
            json.dumps({"event": "fold_complete", "fold": held_fold, "metrics": metrics}),
            flush=True,
        )

    if any(row is None for row in predictions):
        raise AssertionError("OOF predictions do not cover every TRAIN row")
    prediction_rows = [row for row in predictions if row is not None]
    prediction_path = output / "out-of-fold-predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in prediction_rows
        ),
        encoding="utf-8",
    )
    probabilities_all = torch.tensor(
        [row["probabilities"] for row in prediction_rows], dtype=torch.float32
    )
    targets_all = torch.tensor(targets, dtype=torch.long)
    overall = _metrics(probabilities_all, targets_all)
    reference_metrics = _metrics(
        torch.tensor([row["probabilities"] for row in reference_predictions], dtype=torch.float32),
        targets_all,
    )
    strata = {}
    for stratum in (True, False):
        indices = [i for i, row in enumerate(prediction_rows) if row["target_in_fact"] is stratum]
        if indices:
            strata[str(stratum).lower()] = _metrics(
                probabilities_all[indices], targets_all[indices]
            )
    old_path = output / "question-only-oof-reference.jsonl"
    old_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in reference_predictions
        ),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    paired_bootstrap = _paired_group_bootstrap(rows, prediction_rows, reference_predictions)
    report = {
        "schema_version": 1,
        "status": "complete_train_only_grouped_cross_validation",
        "claim_scope": (
            "QASC evidence-conditioned eight-choice TRAIN-only grouped OOF diagnostic; "
            "not broad text or multimodal quality"
        ),
        "seed": SEED,
        "fold_count": 5,
        "epochs_per_fold": EPOCHS,
        "train_rows": len(rows),
        "unique_fact_pair_groups": len({row["source_record_id"].split(":", 1)[0] for row in rows}),
        "unique_examples_seen": len(rows),
        "training_example_presentations": sum(
            len([i for i, fold in enumerate(assignment) if fold != f]) for f in range(5)
        )
        * EPOCHS,
        "optimizer_updates": total_updates,
        "out_of_fold_metrics": overall,
        "question_only_same_fold_reference_metrics": reference_metrics,
        "paired_group_bootstrap": paired_bootstrap,
        "target_fact_overlap_strata": strata,
        "folds": fold_results,
        "fold_integrity": fold_audits,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "sealed_audit_loaded": False,
        "feature_extraction_device": str(device),
        "readout_training_device": "cpu",
        "encoder_parameters_frozen": 17_389_824,
        "encoder_weights_sha256": EXPECTED_FEATURE_MODEL_SHA256,
        "trainable_readout_parameters": sum(
            parameter.numel() for parameter in last_scorer.parameters()
        ),
        "total_encoder_plus_readout_parameters": 17_389_824
        + sum(parameter.numel() for parameter in last_scorer.parameters()),
        "max_context_tokens": max(context_token_lengths),
        "contexts_over_256_tokens": sum(length > 256 for length in context_token_lengths),
        "feature_extraction_seconds": feature_seconds,
        "training_seconds": train_seconds,
        "evaluation_seconds_including_reload": eval_seconds,
        "wall_seconds": elapsed + feature_seconds,
        "predictions_sha256": _sha256(prediction_path),
        "question_only_reference_sha256": _sha256(old_path),
        "feature_cache_sha256": _sha256(feature_path),
        "evidence_train_sha256": prep["evidence_train_sha256"],
        "preparation_report_sha256": _sha256(prep_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "transformers": importlib.metadata.version("transformers"),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "all_folds_saved_and_reloaded_exactly": all(
            row["reload_probabilities_exact"] for row in fold_results
        ),
    }
    _write_json(output / "report.json", report)
    print(
        json.dumps(
            {"event": "run_complete", "metrics": overall, "wall_seconds": report["wall_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
