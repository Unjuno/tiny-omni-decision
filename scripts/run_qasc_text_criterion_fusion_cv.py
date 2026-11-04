"""Run one fixed, CPU-only QASC TRAIN grouped CV for text-criterion fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from tiny_omni_decision.fusion_decision import FusionDecisionScorer
from tiny_omni_decision.qasc import _normalized
from tiny_omni_decision.qasc_cv import audit_qasc_train_folds, build_qasc_train_folds
from tiny_omni_decision.qasc_text_criterion_fusion import (
    qasc_probability_metrics,
    qasc_text_roles,
)
from tiny_omni_decision.text_runtime import (
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    TEXT_FEATURE_SIZE,
    TypedTextDecisionRuntime,
)

SEED = 17
FOLDS = 5
EPOCHS = 8
BATCH_QUESTIONS = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WIDTH = 128
FUSION_PARAMETERS = 164_610
EXPECTED_CORPUS_SHA256 = "6ad3536ebba70a35f87bf078d0d2837df09fda652201a2b4a6fd7d7b01ed7f0b"
EXPECTED_FOLD_SHA256 = "72282c4b57fb2e5a56c4d2f15f75d11bf11f79071d8ac8a981b1a7311bcd0b67"
EXPECTED_ROWS = 8_134


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("QASC TRAIN JSONL rows must be objects")
    return rows


def _preflight(
    corpus_path: Path, fold_path: Path
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    if _sha256(corpus_path) != EXPECTED_CORPUS_SHA256:
        raise ValueError("QASC evidence TRAIN corpus hash mismatch")
    if _sha256(fold_path) != EXPECTED_FOLD_SHA256:
        raise ValueError("preserved QASC fold assignment hash mismatch")
    rows = _read_jsonl(corpus_path)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} QASC TRAIN rows, got {len(rows)}")
    for row in rows:
        if row.get("split") != "train" or row.get("license") != "CC-BY-4.0":
            raise ValueError("only CC-BY-4.0 QASC TRAIN rows are allowed")
        qasc_text_roles(row)
    assignment, report = build_qasc_train_folds(rows, folds=FOLDS, seed=SEED)
    audits = audit_qasc_train_folds(rows, assignment, folds=FOLDS)
    preserved = _read_jsonl(fold_path)
    if len(preserved) != len(rows):
        raise ValueError("preserved fold assignment row count mismatch")
    for row, prior, fold in zip(rows, preserved, assignment, strict=True):
        group_id = row["source_record_id"].split(":", 1)[0]
        if prior.get("id") != row["id"] or prior.get("fold") != fold:
            raise ValueError("recomputed QASC folds differ from the preserved assignment")
        if prior.get("fact_pair_group_sha256") != group_id:
            raise ValueError("QASC fact-pair group identity changed")
    group_has_answer = 0
    for row in rows:
        facts = row["state"].splitlines()[1:]
        normalized_facts = [_normalized(fact) for fact in facts]
        normalized_target = _normalized(row["target"])
        group_has_answer += int(any(normalized_target in fact for fact in normalized_facts))
    report.update(
        {
            "fold_audits": audits,
            "preserved_fold_assignment_sha256": EXPECTED_FOLD_SHA256,
            "preserved_assignment_exact_match": True,
            "target_string_in_evidence_count": group_has_answer,
            "target_string_in_evidence_fraction": group_has_answer / len(rows),
            "official_validation_read": False,
            "official_test_read": False,
            "sealed_audit_read": False,
        }
    )
    return rows, assignment, report


def _encode_roles(
    rows: list[dict[str, Any]], runtime: TypedTextDecisionRuntime
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    packed = torch.empty((len(rows), 10, TEXT_FEATURE_SIZE), dtype=torch.float32)
    text_digest = hashlib.sha256()
    batch_rows = 32
    for start in range(0, len(rows), batch_rows):
        section = rows[start : start + batch_rows]
        texts: list[str] = []
        for row in section:
            state, criterion, options = qasc_text_roles(row)
            texts.extend((state, criterion, *options))
        features = runtime.encode_texts(texts)
        if features.shape != (len(section) * 10, TEXT_FEATURE_SIZE):
            raise ValueError("MiniLM role features have an unexpected shape")
        shaped = features.reshape(len(section), 10, TEXT_FEATURE_SIZE).cpu().contiguous()
        packed[start : start + len(section)] = shaped
        text_digest.update(shaped.numpy().tobytes())
    return packed[:, 0], packed[:, 1], packed[:, 2:], text_digest.hexdigest()


def _predict(
    model: FusionDecisionScorer,
    states: torch.Tensor,
    criteria: torch.Tensor,
    options: torch.Tensor,
    indices: list[int],
    *,
    criterion_present: bool = True,
    permutation: list[int] | None = None,
) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_QUESTIONS):
            selected = indices[start : start + BATCH_QUESTIONS]
            index = torch.tensor(selected, dtype=torch.long)
            batch_states = states[index]
            batch_options = options[index]
            if permutation is not None:
                batch_options = batch_options[:, permutation]
            availability = torch.ones(len(selected), dtype=torch.bool)
            valid = torch.ones((len(selected), 8), dtype=torch.bool)
            if criterion_present:
                batch_criteria = criteria[index]
                present = torch.ones(len(selected), dtype=torch.bool)
            else:
                batch_criteria = None
                present = None
            logits = model(
                {"text_state": batch_states},
                {"text_state": availability},
                batch_options,
                valid,
                batch_criteria,
                present,
            )
            chunks.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(chunks) if chunks else torch.empty((0, 8))


def _fold_position_prior(
    train_indices: list[int], held_indices: list[int], target_indices: list[int]
) -> torch.Tensor:
    counts = torch.bincount(torch.tensor([target_indices[i] for i in train_indices]), minlength=8)
    prior = counts.float() / counts.sum()
    return prior[None].expand(len(held_indices), -1).clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--fold-assignment", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text-readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    corpus_path = args.train_jsonl.resolve(strict=True)
    fold_path = args.fold_assignment.resolve(strict=True)
    model_dir = args.model_dir.resolve(strict=True)
    readout_path = args.text_readout.resolve(strict=True)
    rows, assignment, fold_report = _preflight(corpus_path, fold_path)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root / "configs" / "pretrained_reuse" / "qasc_text_criterion_fusion_cv_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_config = config["readout"]
    if (
        config.get("experiment_id") != "qasc-text-criterion-fusion-cv-seed17-v1"
        or config.get("evidence_train_sha256") != EXPECTED_CORPUS_SHA256
        or config.get("fold_assignment_sha256") != EXPECTED_FOLD_SHA256
        or training_config.get("seed_per_fold") != SEED
        or training_config.get("fold_count") != FOLDS
        or training_config.get("epochs_per_fold") != EPOCHS
        or training_config.get("batch_questions") != BATCH_QUESTIONS
        or training_config.get("learning_rate") != LEARNING_RATE
        or training_config.get("weight_decay") != WEIGHT_DECAY
        or training_config.get("width") != WIDTH
        or training_config.get("trainable_parameters") != FUSION_PARAMETERS
        or training_config.get("device") != "cpu"
    ):
        raise ValueError("effective run constants differ from the frozen experiment config")
    config_sha256 = _sha256(config_path)
    if args.dry_run:
        print(json.dumps(fold_report, indent=2, ensure_ascii=False))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is set")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QASC criterion-fusion run: {output}")

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    runtime = TypedTextDecisionRuntime(model_dir, readout_path, device="cpu")
    if runtime.parameter_count != 17_586_689:
        raise ValueError("unexpected pinned Path A text runtime parameter count")
    if MODEL_REVISION != "4ca70771034acceecb2e72475f72050fcdde4ddc":
        raise ValueError("unexpected MiniLM revision")
    if MODEL_WEIGHTS_SHA256 != "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433":
        raise ValueError("unexpected MiniLM weight hash")

    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "experiment-config.json")
    started = time.perf_counter()
    states, criteria, options, feature_sha256 = _encode_roles(rows, runtime)
    target_indices = [row["options"].index(row["target"]) for row in rows]
    _write_json(
        output / "effective-config.json",
        {
            "experiment_id": "qasc-text-criterion-fusion-cv-seed17-v1",
            "experiment_config_sha256": config_sha256,
            "seed": SEED,
            "folds": FOLDS,
            "epochs_per_fold": EPOCHS,
            "batch_questions": BATCH_QUESTIONS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "device": "cpu",
            "max_length": 256,
            "model": "FusionDecisionScorer",
            "state_dimension": TEXT_FEATURE_SIZE,
            "criterion_dimension": TEXT_FEATURE_SIZE,
            "option_dimension": TEXT_FEATURE_SIZE,
            "hidden_width": WIDTH,
            "trainable_parameters": FUSION_PARAMETERS,
            "checkpoint_rule": "fixed epoch 8 endpoint; no held-fold checkpoint selection",
            "official_validation_read": False,
            "official_test_read": False,
            "sealed_audit_read": False,
        },
    )
    _write_json(output / "fold-plan.json", fold_report)
    with (output / "fold-assignment.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row, fold in zip(rows, assignment, strict=True):
            stream.write(
                json.dumps({"id": row["id"], "fold": fold}, ensure_ascii=False) + "\n"
            )

    oof_main: list[list[float] | None] = [None] * len(rows)
    oof_masked: list[list[float] | None] = [None] * len(rows)
    fold_results = []
    training_seconds = 0.0
    evaluation_seconds = 0.0
    total_updates = 0
    train_presentations = 0
    initial_model = FusionDecisionScorer(
        {"text_state": TEXT_FEATURE_SIZE},
        option_dim=TEXT_FEATURE_SIZE,
        criterion_dim=TEXT_FEATURE_SIZE,
        width=WIDTH,
    )
    trainable_parameters = sum(parameter.numel() for parameter in initial_model.parameters())
    del initial_model
    if trainable_parameters != FUSION_PARAMETERS:
        raise ValueError(f"unexpected FusionDecisionScorer size: {trainable_parameters}")

    for held_fold in range(FOLDS):
        train_indices = [i for i, fold in enumerate(assignment) if fold != held_fold]
        held_indices = [i for i, fold in enumerate(assignment) if fold == held_fold]
        torch.manual_seed(SEED)
        model = FusionDecisionScorer(
            {"text_state": TEXT_FEATURE_SIZE},
            option_dim=TEXT_FEATURE_SIZE,
            criterion_dim=TEXT_FEATURE_SIZE,
            width=WIDTH,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        rng = random.Random(SEED)
        history = []
        fold_start = time.perf_counter()
        updates = 0
        for epoch in range(1, EPOCHS + 1):
            order = train_indices.copy()
            rng.shuffle(order)
            model.train()
            losses = []
            for start in range(0, len(order), BATCH_QUESTIONS):
                selected = order[start : start + BATCH_QUESTIONS]
                index = torch.tensor(selected, dtype=torch.long)
                batch_size = len(selected)
                state = states[index]
                criterion = criteria[index]
                candidates = options[index]
                target = torch.tensor([target_indices[i] for i in selected])
                available = torch.ones(batch_size, dtype=torch.bool)
                candidate_mask = torch.ones((batch_size, 8), dtype=torch.bool)
                criterion_mask = torch.ones(batch_size, dtype=torch.bool)
                logits = model(
                    {"text_state": state},
                    {"text_state": available},
                    candidates,
                    candidate_mask,
                    criterion,
                    criterion_mask,
                )
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss in fold {held_fold}, epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for parameter in model.parameters():
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
                    "rolling_train_ce": sum(losses) / len(losses),
                    "optimizer_updates": updates,
                }
            )
        fold_training_seconds = time.perf_counter() - fold_start
        training_seconds += fold_training_seconds
        total_updates += updates
        train_presentations += len(train_indices) * EPOCHS

        checkpoint = output / f"fold-{held_fold}-epoch-8.pt"
        torch.save(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint
        )
        evaluation_start = time.perf_counter()
        probabilities = _predict(model, states, criteria, options, held_indices)
        criterion_masked = _predict(
            model, states, criteria, options, held_indices, criterion_present=False
        )
        reloaded = FusionDecisionScorer(
            {"text_state": TEXT_FEATURE_SIZE},
            option_dim=TEXT_FEATURE_SIZE,
            criterion_dim=TEXT_FEATURE_SIZE,
            width=WIDTH,
        ).eval()
        reloaded.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        reloaded_probabilities = _predict(reloaded, states, criteria, options, held_indices)
        if not torch.equal(probabilities, reloaded_probabilities):
            raise AssertionError(f"save/reload predictions differ in fold {held_fold}")

        permutation = [7, 6, 5, 4, 3, 2, 1, 0]
        probe_indices = held_indices[: min(32, len(held_indices))]
        probe = _predict(reloaded, states, criteria, options, probe_indices)
        permuted = _predict(
            reloaded, states, criteria, options, probe_indices, permutation=permutation
        )
        if not torch.allclose(permuted[:, permutation], probe, atol=1e-7, rtol=1e-7):
            raise AssertionError(f"option permutation invariance failed in fold {held_fold}")

        targets = torch.tensor([target_indices[i] for i in held_indices])
        masked_metrics = qasc_probability_metrics(criterion_masked, targets)
        metrics = qasc_probability_metrics(probabilities, targets)
        prior = _fold_position_prior(train_indices, held_indices, target_indices)
        prior_metrics = qasc_probability_metrics(prior, targets)
        for local_index, row_index in enumerate(held_indices):
            oof_main[row_index] = probabilities[local_index].tolist()
            oof_masked[row_index] = criterion_masked[local_index].tolist()
        fold_eval_seconds = time.perf_counter() - evaluation_start
        evaluation_seconds += fold_eval_seconds
        fold_results.append(
            {
                "fold": held_fold,
                "train_rows": len(train_indices),
                "held_rows": len(held_indices),
                "history": history,
                "criterion_present_metrics": metrics,
                "criterion_absent_same_checkpoint_metrics": masked_metrics,
                "answer_position_prior": prior_metrics,
                "checkpoint_sha256": _sha256(checkpoint),
                "save_reload_exact": True,
                "option_permutation_invariant": True,
                "training_seconds": fold_training_seconds,
                "evaluation_seconds": fold_eval_seconds,
            }
        )
        print(
            json.dumps(
                {
                    "event": "fold_complete",
                    "fold": held_fold,
                    "criterion_present": metrics,
                    "criterion_absent": masked_metrics,
                }
            ),
            flush=True,
        )

    if any(value is None for value in oof_main + oof_masked):
        raise AssertionError("OOF probabilities do not cover all TRAIN rows")
    main_tensor = torch.tensor(oof_main, dtype=torch.float32)
    masked_tensor = torch.tensor(oof_masked, dtype=torch.float32)
    targets_tensor = torch.tensor(target_indices)
    uniform = torch.full((len(rows), 8), 1.0 / 8)
    position_prior = torch.cat(
        [
            _fold_position_prior(
                [i for i, fold in enumerate(assignment) if fold != held_fold],
                [i for i, fold in enumerate(assignment) if fold == held_fold],
                target_indices,
            )
            for held_fold in range(FOLDS)
        ]
    )
    # Folds are concatenated by fold above; use an index-aligned prior for aggregate metrics.
    position_aligned = torch.empty_like(position_prior)
    offset = 0
    for held_fold in range(FOLDS):
        held_indices = [i for i, fold in enumerate(assignment) if fold == held_fold]
        position_aligned[held_indices] = position_prior[offset : offset + len(held_indices)]
        offset += len(held_indices)

    prediction_path = output / "out-of-fold-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            predicted = int(main_tensor[index].argmax())
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "source_record_id": row["source_record_id"],
                        "fold": assignment[index],
                        "target_index": target_indices[index],
                        "prediction_index": predicted,
                        "target_option": row["target"],
                        "prediction_option": row["options"][predicted],
                        "probabilities": main_tensor[index].tolist(),
                        "criterion_absent_probabilities": masked_tensor[index].tolist(),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "status": "complete_train_only_grouped_cross_validation",
        "claim_scope": (
            "QASC evidence-grounded text Decision only; "
            "not broad text or multimodal quality"
        ),
        "seed": SEED,
        "fold_count": FOLDS,
        "epochs_per_fold": EPOCHS,
        "training_rows": len(rows),
        "unique_examples_held_out_once": len(rows),
        "training_example_presentations": train_presentations,
        "optimizer_updates": total_updates,
        "encoder_revision": MODEL_REVISION,
        "encoder_weight_sha256": MODEL_WEIGHTS_SHA256,
        "encoder_parameters_frozen": 17_389_824,
        "encoder_runtime_parameter_count_including_unused_text_readout": runtime.parameter_count,
        "readout_class": "FusionDecisionScorer",
        "trainable_parameters": trainable_parameters,
        "role_feature_cache_sha256": feature_sha256,
        "experiment_config_sha256": config_sha256,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "helper_sha256": _sha256(
            repo_root / "src" / "tiny_omni_decision" / "qasc_text_criterion_fusion.py"
        ),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "out_of_fold_metrics": qasc_probability_metrics(main_tensor, targets_tensor),
        "criterion_absent_same_checkpoint_metrics": qasc_probability_metrics(
            masked_tensor, targets_tensor
        ),
        "uniform_baseline": qasc_probability_metrics(uniform, targets_tensor),
        "answer_position_prior": qasc_probability_metrics(position_aligned, targets_tensor),
        "folds": fold_results,
        "fold_integrity": fold_report,
        "save_reload_exact_all_folds": all(item["save_reload_exact"] for item in fold_results),
        "option_permutation_invariant_all_folds": all(
            item["option_permutation_invariant"] for item in fold_results
        ),
        "training_seconds": training_seconds,
        "evaluation_and_reload_seconds": evaluation_seconds,
        "wall_seconds": elapsed,
        "device": "cpu",
        "official_validation_read": False,
        "official_test_read": False,
        "sealed_audit_read": False,
    }
    _write_json(output / "report.json", report)
    print(json.dumps({"event": "run_complete", "report": report}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
