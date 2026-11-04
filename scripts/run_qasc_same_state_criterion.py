"""Run a single CPU QASC same-state/new-criterion development diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tiny_omni_decision.fusion_decision import FusionDecisionScorer
from tiny_omni_decision.qasc import _normalized
from tiny_omni_decision.qasc_same_state import (
    build_same_state_criterion_split,
    make_state_derangement,
)
from tiny_omni_decision.qasc_text_criterion_fusion import (
    qasc_probability_metrics,
    qasc_text_roles,
)
from tiny_omni_decision.text_runtime import (
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    READOUT_SHA256,
    TEXT_FEATURE_SIZE,
    TypedTextDecisionRuntime,
)

SEED = 17
BOOTSTRAP_REPETITIONS = 5_000
BOOTSTRAP_SEED = 1_718
DERANGEMENT_SEED = 18
EPOCHS = 8
BATCH_QUESTIONS = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WIDTH = 128
TRAINABLE_PARAMETERS = 164_610
EXPECTED_CORPUS_SHA256 = "6ad3536ebba70a35f87bf078d0d2837df09fda652201a2b4a6fd7d7b01ed7f0b"
EXPECTED_ENCODER_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
EXPECTED_ENCODER_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
EXPECTED_TRAIN_ROWS = 7_451
EXPECTED_HELD_ROWS = 683
EXPECTED_HELD_STATES = 666


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
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("QASC evidence TRAIN rows must be JSON objects")
    return rows


def _state_key(state: str) -> str:
    return hashlib.sha256(_normalized(state).encode("utf-8")).hexdigest()


def _preflight(corpus: Path, config_path: Path):
    if _sha256(corpus) != EXPECTED_CORPUS_SHA256:
        raise ValueError("QASC evidence TRAIN hash mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("experiment_id") != "qasc-same-state-new-criterion-seed17-v1"
        or config.get("source_revision") != "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"
        or config.get("license") != "CC-BY-4.0"
        or config.get("evidence_train_sha256") != EXPECTED_CORPUS_SHA256
        or config.get("train_rows") != EXPECTED_TRAIN_ROWS
        or config.get("held_rows") != EXPECTED_HELD_ROWS
        or config.get("held_state_groups") != EXPECTED_HELD_STATES
        or config.get("seed") != SEED
        or config.get("uncertainty", {}).get("repetitions") != BOOTSTRAP_REPETITIONS
        or config.get("uncertainty", {}).get("seed") != BOOTSTRAP_SEED
        or config.get("encoder", {}).get("revision") != EXPECTED_ENCODER_REVISION
        or config.get("encoder", {}).get("weight_sha256") != EXPECTED_ENCODER_SHA256
        or config.get("encoder", {}).get("max_sequence_length") != 256
        or config.get("readout", {}).get("class") != "FusionDecisionScorer"
        or config.get("readout", {}).get("modality_dimensions") != {"text_state": 384}
        or config.get("readout", {}).get("option_dim") != 384
        or config.get("readout", {}).get("criterion_dim") != 384
        or config.get("readout", {}).get("width") != WIDTH
        or config.get("readout", {}).get("trainable_parameters") != TRAINABLE_PARAMETERS
        or config.get("readout", {}).get("epochs") != EPOCHS
        or config.get("readout", {}).get("batch_questions") != BATCH_QUESTIONS
        or config.get("readout", {}).get("learning_rate") != LEARNING_RATE
        or config.get("readout", {}).get("weight_decay") != WEIGHT_DECAY
        or config.get("readout", {}).get("device") != "cpu"
        or config.get("readout", {}).get("torch_threads") != 4
    ):
        raise ValueError("config does not match the frozen same-state experiment")
    rows = _read_jsonl(corpus)
    if len(rows) != 8_134:
        raise ValueError(f"expected 8,134 QASC TRAIN rows, received {len(rows)}")
    for row in rows:
        qasc_text_roles(row)
    train_indices, held_indices, split_report = build_same_state_criterion_split(rows, seed=SEED)
    expected_hashes = config.get("expected_split_hashes", {})
    for name in ("train_ids_sha256", "held_ids_sha256", "held_question_order_sha256"):
        if expected_hashes.get(name) != split_report.get(name):
            raise ValueError(f"frozen split hash mismatch: {name}")
    return rows, train_indices, held_indices, split_report, config


def _encode_roles(rows: list[dict[str, Any]], runtime: TypedTextDecisionRuntime):
    packed = torch.empty((len(rows), 10, TEXT_FEATURE_SIZE), dtype=torch.float32)
    digest = hashlib.sha256()
    for start in range(0, len(rows), 32):
        section = rows[start : start + 32]
        texts: list[str] = []
        for row in section:
            state, criterion, options = qasc_text_roles(row)
            texts.extend((state, criterion, *options))
        encoded = runtime.encode_texts(texts)
        if encoded.shape != (len(section) * 10, TEXT_FEATURE_SIZE):
            raise ValueError("pinned MiniLM returned an unexpected role feature shape")
        shaped = encoded.reshape(len(section), 10, TEXT_FEATURE_SIZE).cpu().contiguous()
        packed[start : start + len(section)] = shaped
        digest.update(shaped.numpy().tobytes())
    return packed[:, 0], packed[:, 1], packed[:, 2:], digest.hexdigest()


def _predict_override(
    model: FusionDecisionScorer,
    state_override: torch.Tensor,
    criteria: torch.Tensor,
    options: torch.Tensor,
    *,
    criterion_present: bool = True,
) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(state_override), BATCH_QUESTIONS):
            end = start + BATCH_QUESTIONS
            batch_state = state_override[start:end]
            batch_options = options[start:end]
            size = len(batch_state)
            criterion = criteria[start:end] if criterion_present else None
            present = torch.ones(size, dtype=torch.bool) if criterion_present else None
            logits = model(
                {"text_state": batch_state},
                {"text_state": torch.ones(size, dtype=torch.bool)},
                batch_options,
                torch.ones((size, 8), dtype=torch.bool),
                criterion,
                present,
            )
            chunks.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(chunks)


def _row_metrics(probabilities: torch.Tensor, targets: torch.Tensor):
    targets = targets.long()
    correct = probabilities.argmax(-1).eq(targets).float()
    confidence = probabilities.max(-1).values
    nll = -probabilities.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-12).log()
    one_hot = F.one_hot(targets, num_classes=8).float()
    brier = (probabilities - one_hot).square().sum(-1)
    return {
        "accuracy": correct.tolist(),
        "nll": nll.tolist(),
        "brier": brier.tolist(),
        "confidence": confidence.tolist(),
    }


def _group_bootstrap(
    probabilities: dict[str, torch.Tensor],
    targets: torch.Tensor,
    group_keys: list[str],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    group_order = sorted(set(group_keys))
    rows_by_group: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        rows_by_group[key].append(index)
    all_rows = {name: _row_metrics(value, targets) for name, value in probabilities.items()}
    metric_names = ("accuracy", "nll", "brier")
    sums: dict[str, dict[str, list[float]]] = {
        name: {
            metric: [
                sum(all_rows[name][metric][index] for index in rows_by_group[group])
                for group in group_order
            ]
            for metric in metric_names
        }
        for name in probabilities
    }
    group_sizes = [len(rows_by_group[group]) for group in group_order]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(group_order), [1.0 / len(group_order)] * len(group_order), size=repetitions
    )
    denominators = counts @ np.asarray(group_sizes, dtype=np.float64)
    result = {}
    reference = "correct_state_and_criterion"
    for name in probabilities:
        if name == reference:
            continue
        result[name] = {}
        for metric in metric_names:
            ref_values = counts @ np.asarray(sums[reference][metric], dtype=np.float64)
            other_values = counts @ np.asarray(sums[name][metric], dtype=np.float64)
            delta = (ref_values - other_values) / denominators
            low, high = np.percentile(delta, [2.5, 97.5])
            result[name][metric] = {
                "reference_minus_control": float(np.mean(delta)),
                "percentile_95_ci": [float(low), float(high)],
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed,
                "cluster_unit": "held_state_group",
            }
    return result


def _target_in_evidence(row: dict[str, Any]) -> bool:
    target = _normalized(row["target"])
    if not target:
        return False
    return any(target in _normalized(fact) for fact in row["state"].splitlines()[1:])


def _position_prior(rows: list[dict[str, Any]], train_indices: list[int], held_indices: list[int]):
    counts = torch.bincount(
        torch.tensor(
            [rows[index]["options"].index(rows[index]["target"]) for index in train_indices]
        ),
        minlength=8,
    ).float()
    prior = counts / counts.sum()
    return prior[None].expand(len(held_indices), -1).clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text-readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    corpus = args.train_jsonl.resolve(strict=True)
    model_dir = args.model_dir.resolve(strict=True)
    readout_path = args.text_readout.resolve(strict=True)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root / "configs" / "pretrained_reuse" / "qasc_same_state_criterion_seed17_v1.json"
    )
    rows, train_indices, held_indices, split_report, config = _preflight(corpus, config_path)
    if args.dry_run:
        summary_keys = (
            "source_rows",
            "state_groups",
            "states_with_multiple_questions",
            "eligible_states",
            "training_rows",
            "held_rows",
            "held_unique_questions",
            "held_state_groups",
            "all_held_states_are_in_train",
            "normalized_question_overlap",
            "normalized_question_option_overlap",
            "train_ids_sha256",
            "held_ids_sha256",
            "held_target_in_evidence_count",
            "held_target_in_evidence_fraction",
        )
        print(json.dumps({key: split_report[key] for key in summary_keys}, indent=2))
        return
    if (
        len(train_indices) != EXPECTED_TRAIN_ROWS
        or len(held_indices) != EXPECTED_HELD_ROWS
        or split_report["held_state_groups"] != EXPECTED_HELD_STATES
        or not split_report["all_held_states_are_in_train"]
        or split_report["normalized_question_overlap"] != 0
        or split_report["normalized_question_option_overlap"] != 0
    ):
        raise ValueError("actual deterministic split differs from the frozen plan")
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is set")
    output = args.output_dir.resolve()
    if output.is_relative_to(repo_root):
        raise ValueError("experiment artifacts must be stored outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QASC same-state run: {output}")
    if shutil.disk_usage(output.parent).free < 64 * 1024 * 1024:
        raise OSError("less than 64 MiB free for the bounded CPU run receipt/checkpoint")

    if (
        MODEL_REVISION != EXPECTED_ENCODER_REVISION
        or MODEL_WEIGHTS_SHA256 != EXPECTED_ENCODER_SHA256
    ):
        raise ValueError("local text runtime does not match the pinned MiniLM checkpoint")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    runtime = TypedTextDecisionRuntime(model_dir, readout_path, device="cpu")
    if runtime.parameter_count != 17_586_689:
        raise ValueError("unexpected pinned Path A text runtime parameter count")

    output.mkdir(parents=True, exist_ok=False)
    config_sha = _sha256(config_path)
    shutil.copyfile(config_path, output / "experiment-config.json")
    started_at = datetime.now(UTC).isoformat()
    effective = {
        "experiment_id": config["experiment_id"],
        "experiment_config_sha256": config_sha,
        "source_corpus_sha256": _sha256(corpus),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "seed": SEED,
        "training_rows": len(train_indices),
        "held_rows": len(held_indices),
        "held_state_groups": split_report["held_state_groups"],
        "epochs": EPOCHS,
        "batch_questions": BATCH_QUESTIONS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "encoder_revision": MODEL_REVISION,
        "encoder_weight_sha256": MODEL_WEIGHTS_SHA256,
        "text_readout_sha256": READOUT_SHA256,
        "encoder_parameters_frozen": 17_389_824,
        "encoder_runtime_parameter_count_including_unused_path_a_readout": runtime.parameter_count,
        "readout": "FusionDecisionScorer",
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "checkpoint_rule": "fixed epoch-8 endpoint; no held evaluation during fit",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "state_derangement_seed": DERANGEMENT_SEED,
        "started_at_utc": started_at,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "torch_threads": 4,
        "official_validation_read": False,
        "official_test_read": False,
        "sealed_audit_read": False,
    }
    _write_json(output / "effective-config.json", effective)
    _write_json(output / "run-status.json", {"status": "running", "started_at_utc": started_at})
    _write_json(output / "split-plan.json", split_report)
    for filename, indices in (("train-ids.jsonl", train_indices), ("held-ids.jsonl", held_indices)):
        with (output / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for index in indices:
                stream.write(json.dumps({"id": rows[index]["id"]}, ensure_ascii=False) + "\n")

    started = time.perf_counter()
    torch.manual_seed(SEED)
    model = FusionDecisionScorer(
        {"text_state": TEXT_FEATURE_SIZE},
        option_dim=TEXT_FEATURE_SIZE,
        criterion_dim=TEXT_FEATURE_SIZE,
        width=WIDTH,
    )
    if sum(parameter.numel() for parameter in model.parameters()) != TRAINABLE_PARAMETERS:
        raise ValueError("FusionDecisionScorer parameter count mismatch")
    states, criteria, options, feature_sha = _encode_roles(rows, runtime)
    target_indices = [row["options"].index(row["target"]) for row in rows]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = random.Random(SEED)
    updates = 0
    presentations = 0
    training_seconds = 0.0
    history_path = output / "training-history.jsonl"
    state_group_by_row = [_state_key(row["state"]) for row in rows]
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = train_indices.copy()
        rng.shuffle(order)
        model.train()
        epoch_losses = []
        epoch_start = time.perf_counter()
        for start in range(0, len(order), BATCH_QUESTIONS):
            chosen = order[start : start + BATCH_QUESTIONS]
            indices = torch.tensor(chosen, dtype=torch.long)
            batch_size = len(chosen)
            batch_states = states[indices]
            batch_criteria = criteria[indices]
            batch_options = options[indices]
            targets = torch.tensor([target_indices[index] for index in chosen], dtype=torch.long)
            logits = model(
                {"text_state": batch_states},
                {"text_state": torch.ones(batch_size, dtype=torch.bool)},
                batch_options,
                torch.ones((batch_size, 8), dtype=torch.bool),
                batch_criteria,
                torch.ones(batch_size, dtype=torch.bool),
            )
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite CE at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"invalid gradient at epoch {epoch}")
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            updates += 1
        epoch_seconds = time.perf_counter() - epoch_start
        training_seconds += epoch_seconds
        presentations += len(order)
        record = {
            "epoch": epoch,
            "rolling_train_ce": sum(epoch_losses) / len(epoch_losses),
            "optimizer_updates": updates,
            "training_presentations": presentations,
            "epoch_seconds": epoch_seconds,
            "held_questions_evaluated": False,
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    checkpoint = output / "final-epoch-8.pt"
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
    held_state = states[torch.tensor(held_indices, dtype=torch.long)]
    held_criterion = criteria[torch.tensor(held_indices, dtype=torch.long)]
    held_options = options[torch.tensor(held_indices, dtype=torch.long)]
    held_targets = torch.tensor([target_indices[index] for index in held_indices], dtype=torch.long)
    held_rows = [rows[index] for index in held_indices]
    held_groups = [state_group_by_row[index] for index in held_indices]
    group_keys = sorted(set(held_groups))
    derangement = make_state_derangement(group_keys, seed=DERANGEMENT_SEED)
    first_row_by_group: dict[str, int] = {}
    for row_index, group in enumerate(state_group_by_row):
        first_row_by_group.setdefault(group, row_index)
    shuffled_state = torch.stack(
        [states[first_row_by_group[derangement[group]]] for group in held_groups]
    )
    zero_state = torch.zeros_like(held_state)

    evaluation_started = time.perf_counter()
    predictions = {
        "correct_state_and_criterion": _predict_override(
            model, held_state, held_criterion, held_options
        ),
        "correct_state_criterion_absent": _predict_override(
            model, held_state, held_criterion, held_options, criterion_present=False
        ),
        "zero_state_content_and_criterion": _predict_override(
            model, zero_state, held_criterion, held_options
        ),
        "different_deranged_state_and_criterion": _predict_override(
            model, shuffled_state, held_criterion, held_options
        ),
    }
    reloaded = FusionDecisionScorer(
        {"text_state": TEXT_FEATURE_SIZE},
        option_dim=TEXT_FEATURE_SIZE,
        criterion_dim=TEXT_FEATURE_SIZE,
        width=WIDTH,
    ).eval()
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    reloaded_predictions = _predict_override(reloaded, held_state, held_criterion, held_options)
    if not torch.equal(predictions["correct_state_and_criterion"], reloaded_predictions):
        raise AssertionError("save/reload probabilities changed")
    permutation = list(reversed(range(8)))
    with torch.inference_mode():
        permuted_logits = reloaded(
            {"text_state": held_state[:64]},
            {"text_state": torch.ones(min(64, len(held_state)), dtype=torch.bool)},
            held_options[:64, permutation],
            torch.ones((min(64, len(held_state)), 8), dtype=torch.bool),
            held_criterion[:64],
            torch.ones(min(64, len(held_state)), dtype=torch.bool),
        )
        permuted = torch.softmax(permuted_logits.float(), dim=-1)[:, permutation]
        original = predictions["correct_state_and_criterion"][:64]
        permutation_delta = float((permuted - original).abs().max())
        if not torch.allclose(permuted, original, atol=1e-6, rtol=1e-6):
            raise AssertionError("option permutation invariance failed")

    metrics = {
        name: qasc_probability_metrics(probability, held_targets)
        for name, probability in predictions.items()
    }
    uniform = torch.full_like(predictions["correct_state_and_criterion"], 1 / 8)
    position_counts = torch.bincount(
        torch.tensor(target_indices)[train_indices], minlength=8
    ).float()
    position_prior = (
        (position_counts / position_counts.sum())[None].expand(len(held_indices), -1).clone()
    )
    metrics["uniform_control"] = qasc_probability_metrics(uniform, held_targets)
    metrics["training_answer_position_prior"] = qasc_probability_metrics(
        position_prior, held_targets
    )
    grouped_intervals = _group_bootstrap(
        predictions,
        held_targets,
        held_groups,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
    )
    target_strata = {}
    in_evidence = torch.tensor([_target_in_evidence(row) for row in held_rows], dtype=torch.bool)
    for label, mask in (
        ("target_in_evidence", in_evidence),
        ("target_not_in_evidence", ~in_evidence),
    ):
        selected = torch.nonzero(mask, as_tuple=True)[0]
        target_strata[label] = {"examples": len(selected)}
        if len(selected):
            selected_targets = held_targets[selected]
            for name, probability in predictions.items():
                target_strata[label][name] = qasc_probability_metrics(
                    probability[selected], selected_targets
                )

    prediction_path = output / "held-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for local_index, (row_index, row) in enumerate(zip(held_indices, held_rows, strict=True)):
            target_index = target_indices[row_index]
            payload = {
                "id": row["id"],
                "source_record_id": row["source_record_id"],
                "state_group_sha256": held_groups[local_index],
                "question": row["question"],
                "target_index": target_index,
                "target_option": row["target"],
                "target_in_evidence": bool(in_evidence[local_index]),
            }
            for name, probability in predictions.items():
                chosen = int(probability[local_index].argmax())
                payload[name] = {
                    "prediction_index": chosen,
                    "prediction_option": row["options"][chosen],
                    "probabilities": probability[local_index].tolist(),
                }
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    training_elapsed = time.perf_counter() - started
    eval_seconds = time.perf_counter() - evaluation_started
    report = {
        "schema_version": 1,
        "status": "complete_internal_train_only_same_state_criterion_diagnostic",
        "claim_scope": config["claim_scope"],
        "start_time_utc": started_at,
        "finish_time_utc": datetime.now(UTC).isoformat(),
        "source_commit": effective["source_commit"],
        "effective_config_sha256": _sha256(output / "effective-config.json"),
        "experiment_config_sha256": config_sha,
        "corpus_sha256": _sha256(corpus),
        "role_feature_matrix_sha256": feature_sha,
        "final_checkpoint_sha256": _sha256(checkpoint),
        "prediction_file_sha256": _sha256(prediction_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "split_helper_sha256": _sha256(
            repo_root / "src" / "tiny_omni_decision" / "qasc_same_state.py"
        ),
        "seed": SEED,
        "train_rows": len(train_indices),
        "held_rows": len(held_indices),
        "held_state_groups": len(set(held_groups)),
        "training_presentations": presentations,
        "optimizer_updates": updates,
        "rolling_training_history": history,
        "training_seconds": training_seconds,
        "elapsed_through_final_eval_seconds": training_elapsed,
        "final_evaluation_seconds": eval_seconds,
        "encoder_revision": MODEL_REVISION,
        "encoder_weight_sha256": MODEL_WEIGHTS_SHA256,
        "text_readout_sha256": READOUT_SHA256,
        "encoder_parameters_frozen": 17_389_824,
        "encoder_runtime_parameters_including_unused_path_a_readout": runtime.parameter_count,
        "readout": "FusionDecisionScorer",
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "device": "cpu",
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "metrics": metrics,
        "paired_group_bootstrap": grouped_intervals,
        "target_in_evidence_strata": target_strata,
        "split_integrity": split_report,
        "state_derangement": {
            "seed": DERANGEMENT_SEED,
            "state_groups": len(derangement),
            "fixed_points": sum(
                source == destination for source, destination in derangement.items()
            ),
            "mapping_sha256": hashlib.sha256(
                json.dumps(derangement, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "save_reload_exact": True,
        "option_permutation_invariant": True,
        "option_permutation_max_abs_probability_delta": permutation_delta,
        "official_validation_read": False,
        "official_test_read": False,
        "sealed_audit_read": False,
        "project_level_independent_holdout": False,
    }
    _write_json(output / "report.json", report)
    _write_json(
        output / "run-status.json",
        {"status": "complete", "finished_at_utc": report["finish_time_utc"]},
    )
    print(json.dumps({"event": "run_complete", "report": report}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
