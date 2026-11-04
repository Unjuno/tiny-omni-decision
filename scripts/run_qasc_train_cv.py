"""Run a fixed grouped cross-validation fit using QASC TRAIN features only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.qasc_cv import audit_qasc_train_folds, build_qasc_train_folds

SEED = 17
FOLDS = 5
EPOCHS = 8
BATCH_QUESTIONS = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EXPECTED_TRAIN_SHA256 = "d2e220510f98e19a2df5e44f89d29e1b1bd5b218397264b08ff1102ea3903e51"
EXPECTED_FEATURE_SHA256 = "997029c92fe486a2279595da994847a2c0d5442649e667cbf5e4db6b56fcad16"
FEATURE_SIZE = 1536
HIDDEN_SIZE = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _metrics(probabilities: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> dict[str, Any]:
    probabilities = probabilities.float()
    targets = targets.long()
    if probabilities.ndim != 2 or probabilities.shape != (len(targets), 8):
        raise ValueError("QASC predictions must have shape [rows,8]")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("QASC probabilities must be finite and nonnegative")
    if not torch.allclose(probabilities.sum(-1), torch.ones(len(targets)), atol=1e-5):
        raise ValueError("QASC probabilities must sum to one")
    predictions = probabilities.argmax(-1)
    correct = predictions.eq(targets)
    nll = -probabilities.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-12).log()
    one_hot = F.one_hot(targets, num_classes=8).float()
    confidence = probabilities.max(-1).values
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index + 1 < bins else confidence <= high)
        if mask.any():
            ece += float(mask.float().mean()) * abs(
                float(confidence[mask].mean() - correct[mask].float().mean())
            )
    return {
        "examples": len(targets),
        "accuracy": float(correct.float().mean()),
        "nll": float(nll.mean()),
        "brier": float((probabilities - one_hot).square().sum(-1).mean()),
        "ece_15_bins": ece,
        "mean_confidence": float(confidence.mean()),
    }


def _predict(
    scorer: FrozenFeatureCandidateScorer,
    features: list[torch.Tensor],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    scorer.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_QUESTIONS):
            batch_indices = indices[start : start + BATCH_QUESTIONS]
            widths = [int(features[index].shape[0]) for index in batch_indices]
            flat = torch.cat([features[index] for index in batch_indices])
            logits = scorer(flat)
            offset = 0
            for width in widths:
                outputs.append(logits[offset : offset + width].float())
                offset += width
    return torch.stack(outputs), torch.tensor(indices, dtype=torch.long)


def _baseline_position_probabilities(
    train_indices: list[int], held_indices: list[int], targets: list[int]
) -> torch.Tensor:
    counts = torch.bincount(torch.tensor([targets[i] for i in train_indices]), minlength=8).float()
    prior = counts / counts.sum()
    return prior.unsqueeze(0).expand(len(held_indices), -1).clone()


def _preflight(train_path: Path, feature_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _sha256(train_path) != EXPECTED_TRAIN_SHA256:
        raise ValueError("prepared QASC TRAIN JSONL hash changed")
    if _sha256(feature_path) != EXPECTED_FEATURE_SHA256:
        raise ValueError("frozen QASC feature cache hash changed")
    rows = _read_jsonl(train_path)
    if len(rows) != 8134 or any(row.get("split") != "train" for row in rows):
        raise ValueError("expected exactly 8,134 prepared QASC TRAIN rows")
    assignment, split_report = build_qasc_train_folds(rows, folds=FOLDS, seed=SEED)
    fold_audits = audit_qasc_train_folds(rows, assignment, folds=FOLDS)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("QASC TRAIN IDs are not unique")
    split_report["fold_audits"] = fold_audits
    split_report["train_jsonl_sha256"] = EXPECTED_TRAIN_SHA256
    split_report["frozen_feature_cache_sha256"] = EXPECTED_FEATURE_SHA256
    split_report["validation_jsonl_opened"] = False
    split_report["official_test_shard_opened"] = False
    return rows, {"assignment": assignment, "report": split_report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--frozen-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_path = args.train_jsonl.resolve(strict=True)
    feature_path = args.frozen_features.resolve(strict=True)
    rows, split = _preflight(train_path, feature_path)
    if args.dry_run:
        print(json.dumps(split["report"], indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is set")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QASC CV output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    # The preserved cache also contains the original development rows. This runner
    # accesses only the TRAIN feature/target members and never indexes validation keys.
    cache = torch.load(feature_path, map_location="cpu", weights_only=True, mmap=True)
    train_features = cache["train_features"]
    cached_targets = [int(value) for value in cache["train_targets"]]
    if len(train_features) != len(rows) or len(cached_targets) != len(rows):
        raise ValueError("cached TRAIN features/targets do not match prepared TRAIN rows")
    for index, (feature, row, target) in enumerate(
        zip(train_features, rows, cached_targets, strict=True)
    ):
        if (
            feature.shape != (8, FEATURE_SIZE)
            or feature.dtype != torch.float32
            or not torch.isfinite(feature).all()
        ):
            raise ValueError(f"invalid cached TRAIN option features at row {index}")
        expected_target = row["options"].index(row["target"])
        if target != expected_target:
            raise ValueError(f"cached TRAIN target mismatch at row {index}")

    assignment = split["assignment"]
    split_report = split["report"]
    _write_json(
        output / "effective-config.json",
        {
            "experiment_id": "qasc-path-a-train-only-group-cv-seed17",
            "seed": SEED,
            "folds": FOLDS,
            "epochs": EPOCHS,
            "batch_questions": BATCH_QUESTIONS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "encoder_revision": (
                "sentence-transformers/paraphrase-MiniLM-L3-v2@"
                "4ca70771034acceecb2e72475f72050fcdde4ddc"
            ),
            "encoder_frozen_parameters": 17_389_824,
            "readout_parameters": 196_865,
            "feature_size": FEATURE_SIZE,
            "checkpoint_rule": "fixed epoch 8 endpoint; no held-fold selection",
            "device": "cpu",
            "training_rows": len(rows),
            "official_validation_read": False,
            "official_test_read": False,
            "sealed_audit_read": False,
        },
    )
    _write_json(output / "fold-plan.json", split_report)
    assignment_path = output / "fold-assignment.jsonl"
    with assignment_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, fold in zip(rows, assignment, strict=True):
            group = row["source_record_id"].split(":", 1)[0]
            stream.write(
                json.dumps({"id": row["id"], "fact_pair_group_sha256": group, "fold": fold}) + "\n"
            )

    torch.set_num_threads(4)
    all_predictions: list[dict[str, Any] | None] = [None] * len(rows)
    fold_results = []
    total_training_rows = 0
    total_updates = 0
    train_seconds = 0.0
    eval_seconds = 0.0
    started = time.perf_counter()
    for held_fold in range(FOLDS):
        train_indices = [i for i, fold in enumerate(assignment) if fold != held_fold]
        held_indices = [i for i, fold in enumerate(assignment) if fold == held_fold]
        total_training_rows += len(train_indices) * EPOCHS
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
            for start_index in range(0, len(order), BATCH_QUESTIONS):
                batch_indices = order[start_index : start_index + BATCH_QUESTIONS]
                feature_rows = [train_features[index] for index in batch_indices]
                widths = [int(features.shape[0]) for features in feature_rows]
                logits = scorer(torch.cat(feature_rows))
                padded = logits.new_full((len(batch_indices), 8), -1e4)
                offset = 0
                for row_index, width in enumerate(widths):
                    padded[row_index, :width] = logits[offset : offset + width]
                    offset += width
                targets = torch.tensor([cached_targets[index] for index in batch_indices])
                loss = F.cross_entropy(padded, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss in fold {held_fold}, epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for parameter in scorer.parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(
                            f"invalid gradient in fold {held_fold}, epoch {epoch}"
                        )
                optimizer.step()
                losses.append(float(loss.detach()))
                updates += 1
            history.append(
                {
                    "epoch": epoch,
                    "online_train_ce": sum(losses) / len(losses),
                    "optimizer_updates": updates,
                }
            )
        fold_train_seconds = time.perf_counter() - fold_started
        train_seconds += fold_train_seconds
        checkpoint = output / f"fold-{held_fold}-final.pt"
        torch.save(
            {key: value.detach().cpu() for key, value in scorer.state_dict().items()}, checkpoint
        )
        before_eval = time.perf_counter()
        logits, order_indices = _predict(scorer, train_features, held_indices)
        before_reload_probabilities = torch.softmax(logits, dim=-1)
        reloaded = FrozenFeatureCandidateScorer(FEATURE_SIZE, HIDDEN_SIZE)
        reloaded.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        reload_logits, reload_indices = _predict(reloaded, train_features, held_indices)
        probabilities = torch.softmax(reload_logits, dim=-1)
        if not torch.equal(order_indices, reload_indices) or not torch.equal(
            before_reload_probabilities, probabilities
        ):
            raise AssertionError(f"save/reload prediction parity failed in fold {held_fold}")
        targets_tensor = torch.tensor([cached_targets[i] for i in held_indices])
        metrics = _metrics(probabilities, targets_tensor)
        baseline_probs = _baseline_position_probabilities(
            train_indices, held_indices, cached_targets
        )
        baseline = _metrics(baseline_probs, targets_tensor)
        for local_index, source_index in enumerate(held_indices):
            row = rows[source_index]
            predicted = int(probabilities[local_index].argmax())
            all_predictions[source_index] = {
                "id": row["id"],
                "fold": held_fold,
                "target_index": cached_targets[source_index],
                "prediction_index": predicted,
                "target_option": row["options"][cached_targets[source_index]],
                "prediction_option": row["options"][predicted],
                "probabilities": [float(x) for x in probabilities[local_index]],
            }
        fold_eval_seconds = time.perf_counter() - before_eval
        eval_seconds += fold_eval_seconds
        total_updates += updates
        fold_results.append(
            {
                "fold": held_fold,
                "train_rows": len(train_indices),
                "held_rows": len(held_indices),
                "train_fact_pair_groups": len(
                    {rows[i]["source_record_id"].split(":", 1)[0] for i in train_indices}
                ),
                "held_fact_pair_groups": len(
                    {rows[i]["source_record_id"].split(":", 1)[0] for i in held_indices}
                ),
                "epoch_history": history,
                "out_of_fold_metrics": metrics,
                "train_answer_position_prior": baseline,
                "selected_checkpoint_rule": "fixed final epoch 8",
                "checkpoint_sha256": _sha256(checkpoint),
                "reload_probabilities_exact": True,
                "training_seconds": fold_train_seconds,
                "evaluation_and_reload_seconds": fold_eval_seconds,
            }
        )
        print(
            json.dumps({"event": "fold_complete", "fold": held_fold, "metrics": metrics}),
            flush=True,
        )

    if any(row is None for row in all_predictions):
        raise AssertionError("out-of-fold predictions do not cover all TRAIN rows")
    prediction_path = output / "out-of-fold-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in all_predictions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    probs_all = torch.tensor([row["probabilities"] for row in all_predictions], dtype=torch.float32)
    targets_all = torch.tensor(cached_targets)
    oof_metrics = _metrics(probs_all, targets_all)
    uniform = _metrics(torch.full((len(rows), 8), 1 / 8), targets_all)
    answer_positions = Counter(cached_targets)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "status": "complete_train_only_grouped_cross_validation",
        "claim_scope": (
            "QASC eight-choice text-only development cross-validation; "
            "not broad text or multimodal quality"
        ),
        "seed": SEED,
        "fold_count": FOLDS,
        "epochs_per_fold": EPOCHS,
        "train_rows": len(rows),
        "unique_fact_pair_groups": len({row["source_record_id"].split(":", 1)[0] for row in rows}),
        "unique_examples_seen": len(rows),
        "training_example_presentations": total_training_rows,
        "optimizer_updates": total_updates,
        "answer_position_counts": {str(i): answer_positions[i] for i in range(8)},
        "out_of_fold_metrics": oof_metrics,
        "uniform_baseline": uniform,
        "folds": fold_results,
        "fold_integrity": split_report["fold_audits"],
        "official_validation_read": False,
        "official_test_shard_read": False,
        "sealed_audit_read": False,
        "device": "cpu",
        "encoder_parameters_frozen": 17_389_824,
        "trainable_readout_parameters": sum(
            parameter.numel() for parameter in reloaded.parameters()
        ),
        "total_encoder_plus_readout_parameters": 17_389_824
        + sum(parameter.numel() for parameter in reloaded.parameters()),
        "training_seconds": train_seconds,
        "evaluation_seconds_including_reload": eval_seconds,
        "wall_seconds": elapsed,
        "predictions_sha256": _sha256(prediction_path),
        "feature_cache_sha256": EXPECTED_FEATURE_SHA256,
        "train_jsonl_sha256": EXPECTED_TRAIN_SHA256,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "transformers": importlib.metadata.version("transformers"),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "splitter_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "src" / "tiny_omni_decision" / "qasc_cv.py"
        ),
        "all_folds_saved_and_reloaded_exactly": all(
            row["reload_probabilities_exact"] for row in fold_results
        ),
    }
    _write_json(output / "report.json", report)
    print(
        json.dumps(
            {"event": "run_complete", "out_of_fold_metrics": oof_metrics, "wall_seconds": elapsed}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
