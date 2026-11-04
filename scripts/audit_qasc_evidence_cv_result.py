"""Recheck saved QASC evidence-context OOF predictions without loading a model."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.qasc import _normalized
from tiny_omni_decision.qasc_cv import qasc_train_component_ids

SEED = 17


def _probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    values = np.asarray([row["probabilities"] for row in rows], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("saved predictions must contain eight option probabilities")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("saved probabilities must be finite and nonnegative")
    if not np.allclose(values.sum(-1), np.ones(len(values)), atol=1e-5):
        raise ValueError("saved eight-choice probabilities must sum to one")
    return values


def audit_result(
    train_path: Path,
    preparation_path: Path,
    run_dir: Path,
    *,
    replicates: int = 10_000,
) -> dict[str, Any]:
    import torch

    from scripts.run_qasc_evidence_train_cv import _metrics, _read_jsonl, _sha256

    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prediction_path = run_dir / "out-of-fold-predictions.jsonl"
    reference_path = run_dir / "question-only-oof-reference.jsonl"
    if _sha256(train_path) != preparation["evidence_train_sha256"]:
        raise ValueError("evidence TRAIN corpus hash differs from its preparation receipt")
    if _sha256(train_path) != report["evidence_train_sha256"]:
        raise ValueError("evidence TRAIN corpus hash differs from the run report")
    if _sha256(preparation_path) != report["preparation_report_sha256"]:
        raise ValueError("preparation receipt hash differs from the run report")
    if _sha256(prediction_path) != report["predictions_sha256"]:
        raise ValueError("saved candidate prediction hash differs from the run report")
    if _sha256(reference_path) != report["question_only_reference_sha256"]:
        raise ValueError("saved question-only reference hash differs from the run report")
    rows = _read_jsonl(train_path)
    candidate = _read_jsonl(prediction_path)
    reference = _read_jsonl(reference_path)
    if not (len(rows) == len(candidate) == len(reference) == 8134):
        raise ValueError("expected exact QASC TRAIN row coverage from both saved OOF runs")
    targets = []
    target_in_facts = []
    for source, current, old in zip(rows, candidate, reference, strict=True):
        target_index = source["options"].index(source["target"])
        if (
            current["id"] != source["id"]
            or old["id"] != source["id"]
            or int(current["target_index"]) != target_index
            or int(old["target_index"]) != target_index
            or int(current["fold"]) != int(old["fold"])
        ):
            raise ValueError("candidate/reference OOF IDs, targets, or fold assignments differ")
        if int(current["prediction_index"]) != int(np.argmax(current["probabilities"])) or int(
            old["prediction_index"]
        ) != int(np.argmax(old["probabilities"])):
            raise ValueError("saved predicted option index disagrees with its probability vector")
        lines = source["state"].splitlines()
        if len(lines) != 3 or lines[0] != "Evidence:":
            raise ValueError("evidence state does not preserve exactly two support-fact lines")
        normalized_target = _normalized(source["target"])
        target_in_facts.append(
            bool(normalized_target)
            and any(normalized_target in _normalized(fact) for fact in lines[1:])
        )
        targets.append(target_index)
    candidate_probabilities = _probabilities(candidate)
    reference_probabilities = _probabilities(reference)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    candidate_tensor = torch.from_numpy(candidate_probabilities)
    reference_tensor = torch.from_numpy(reference_probabilities)
    candidate_metrics = _metrics(candidate_tensor, target_tensor)
    reference_metrics = _metrics(reference_tensor, target_tensor)
    for key, value in candidate_metrics.items():
        if isinstance(value, (int, float)) and not math.isclose(
            float(value), float(report["out_of_fold_metrics"][key]), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"recomputed candidate metric differs from run report: {key}")

    strata = {}
    for label in (True, False):
        indices = [index for index, included in enumerate(target_in_facts) if included is label]
        if indices:
            strata[str(label).lower()] = {
                "examples": len(indices),
                "metrics": _metrics(candidate_tensor[indices], target_tensor[indices]),
            }
    exact_fact_overlap_count = sum(target_in_facts)
    expected_overlap_count = int(
        preparation["target_option_exact_normalized_substring_in_either_fact_rows"]
    )
    if exact_fact_overlap_count != expected_overlap_count:
        raise ValueError("exact target/fact overlap count differs from preparation audit")

    component_ids = qasc_train_component_ids(rows)
    grouped: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for index, component_id in enumerate(component_ids):
        target = targets[index]
        counts = grouped[component_id]
        counts[0] += float(
            int(candidate[index]["prediction_index"] == target)
            - int(reference[index]["prediction_index"] == target)
        )
        counts[1] += -math.log(max(float(candidate[index]["probabilities"][target]), 1e-12))
        counts[2] += -math.log(max(float(reference[index]["probabilities"][target]), 1e-12))
        counts[3] += 1.0
    cluster_values = np.asarray(list(grouped.values()), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    differences = np.empty((replicates, 2), dtype=np.float64)
    for replicate in range(replicates):
        sampled = cluster_values[rng.integers(0, len(cluster_values), size=len(cluster_values))]
        total = sampled.sum(axis=0)
        differences[replicate, 0] = total[0] / total[3]
        differences[replicate, 1] = (total[1] - total[2]) / total[3]
    bootstrap = {
        "unit": "connected_component_fact_pair_question_normalized_content",
        "unique_groups": len(cluster_values),
        "replicates": replicates,
        "seed": SEED,
        "percentile_95_ci_candidate_minus_question_only_accuracy": [
            float(value) for value in np.quantile(differences[:, 0], [0.025, 0.975])
        ],
        "percentile_95_ci_candidate_minus_question_only_nll": [
            float(value) for value in np.quantile(differences[:, 1], [0.025, 0.975])
        ],
    }
    return {
        "schema_version": 1,
        "status": "saved_oof_analysis_rechecked_no_model_loaded",
        "supersedes_run_report_fields": [
            "target_fact_overlap_strata",
            "paired_group_bootstrap",
        ],
        "original_run_report_sha256": _sha256(report_path),
        "candidate_predictions_sha256": _sha256(prediction_path),
        "question_only_reference_sha256": _sha256(reference_path),
        "evidence_train_sha256": _sha256(train_path),
        "model_reloaded": False,
        "training_repeated": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "sealed_audit_loaded": False,
        "candidate_metrics_recomputed": candidate_metrics,
        "question_only_reference_metrics_recomputed": reference_metrics,
        "exact_target_in_either_support_fact_examples": exact_fact_overlap_count,
        "target_fact_overlap_fraction": exact_fact_overlap_count / len(rows),
        "target_fact_overlap_strata_exact": strata,
        "paired_component_bootstrap": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()
    output = args.run_dir.resolve(strict=True) / "analysis-corrigendum.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite saved analysis correction: {output}")
    result = audit_result(
        args.train_jsonl.resolve(strict=True),
        args.preparation_report.resolve(strict=True),
        args.run_dir.resolve(strict=True),
        replicates=args.replicates,
    )
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"analysis_correction": str(output), "bootstrap": result["paired_component_bootstrap"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
