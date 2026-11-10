"""Paired comparison of a fixed capacity-CV candidate and saved baseline outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tiny_omni_decision.robomnist_av_match_evaluation import classification_metrics  # noqa: E402
from tiny_omni_decision.robomnist_av_match_readout import (  # noqa: E402
    BilinearAVMatchScorer,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _paired_metric_bootstrap(
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    targets: np.ndarray,
    pair_ids: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 17,
    minimum_groups: int = 200,
) -> dict[str, Any]:
    """Compare saved predictions by resampling whole four-row source pairs."""
    candidate_probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    reference_probabilities = np.asarray(reference_probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    pair_ids = np.asarray(pair_ids)
    if (
        candidate_probabilities.ndim != 2
        or candidate_probabilities.shape[1] != 2
        or candidate_probabilities.shape != reference_probabilities.shape
        or candidate_probabilities.shape[0] != targets.size
        or targets.shape != pair_ids.shape
        or not np.isfinite(candidate_probabilities).all()
        or not np.isfinite(reference_probabilities).all()
        or not np.isin(targets, (0, 1)).all()
    ):
        raise ValueError("paired metric bootstrap inputs have invalid shapes or values")
    unique_pairs = np.unique(pair_ids)
    if len(unique_pairs) < minimum_groups:
        raise ValueError("too few independent pair units for the paired bootstrap")
    pair_rows = [np.flatnonzero(pair_ids == pair_id) for pair_id in unique_pairs]
    if any(len(indices) != 4 for indices in pair_rows):
        raise ValueError("every paired bootstrap unit must contain exactly four rows")
    if replicates < 1 or seed < 0:
        raise ValueError("replicates must be positive and seed must be non-negative")
    metrics = ("accuracy", "nll", "brier", "ece_15")
    candidate_point = classification_metrics(candidate_probabilities, targets)
    reference_point = classification_metrics(reference_probabilities, targets)
    point_delta = {
        key: float(candidate_point[key] - reference_point[key]) for key in metrics
    }
    samples = {key: np.empty(replicates, dtype=np.float64) for key in metrics}
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled_units = rng.integers(0, len(pair_rows), size=len(pair_rows))
        indices = np.concatenate([pair_rows[index] for index in sampled_units])
        candidate = classification_metrics(candidate_probabilities[indices], targets[indices])
        reference = classification_metrics(reference_probabilities[indices], targets[indices])
        for key in metrics:
            samples[key][replicate] = candidate[key] - reference[key]
    return {
        "unit": "four-row source pair unit sampled with replacement",
        "independent_pair_units": len(unique_pairs),
        "replicates": replicates,
        "seed": seed,
        "point_delta_candidate_minus_reference": point_delta,
        "percentile_95_ci": {
            key: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for key, values in samples.items()
        },
    }


def run(candidate_run: Path, reference_run: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite comparison output: {output}")
    candidate_meta = json.loads((candidate_run / "result.json").read_text(encoding="utf-8"))
    candidate_provenance = json.loads(
        (candidate_run / "run-implementation-provenance.json").read_text(encoding="utf-8")
    )
    reference_meta = json.loads((reference_run / "result.json").read_text(encoding="utf-8"))
    candidate_path = candidate_run / "validation-predictions.jsonl"
    reference_path = reference_run / "validation-predictions.jsonl"
    if (
        candidate_meta.get("status")
        != "LOWER_RANK_SELECTED_AND_EVALUATED_ON_DEVELOPMENT_VALIDATION"
        or candidate_meta.get("cv", {}).get("selected_rank") != 8
        or candidate_meta.get("sealed_audit_opened") is not False
        or candidate_meta.get("validation_predictions_sha256") != _sha256(candidate_path)
        or candidate_provenance.get("validation_predictions_sha256")
        != _sha256(candidate_path)
        or reference_meta.get("status") != "COMPLETE"
        or reference_meta.get("sealed_audit_opened") is not False
        or reference_meta.get("prediction_file_sha256") != _sha256(reference_path)
        or reference_meta.get("validation_count") != 2000
    ):
        raise ValueError("candidate or reference prediction integrity gate failed")
    candidate_rows = _load_jsonl(candidate_path)
    reference_rows = _load_jsonl(reference_path)
    if len(candidate_rows) != 2000 or len(reference_rows) != 2000:
        raise ValueError("both prediction files must contain all fixed validation rows")

    candidate_probability = []
    reference_probability = []
    targets = []
    pair_ids = []
    for index, (candidate, reference) in enumerate(
        zip(candidate_rows, reference_rows, strict=True)
    ):
        identity = ("pair_unit_id", "audio_group", "video_group")
        if (
            candidate.get("example_index") != index
            or reference.get("example_index") != index
            or any(candidate.get(key) != reference.get(key) for key in identity)
            or candidate.get("target")
            != (1 if reference.get("target_option_id") == "same" else 0)
        ):
            raise ValueError("saved predictions do not align to identical validation examples")
        candidate_probability.append(candidate["probabilities"])
        reference_probability.append(reference["predictions"]["joint"]["probabilities"])
        targets.append(int(candidate["target"]))
        pair_ids.append(candidate["pair_unit_id"])

    candidate_probability = np.asarray(candidate_probability, dtype=np.float64)
    reference_probability = np.asarray(reference_probability, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    pair_ids = np.asarray(pair_ids)
    candidate_metrics = classification_metrics(candidate_probability, targets)
    reference_metrics = classification_metrics(reference_probability, targets)
    candidate_parameters = BilinearAVMatchScorer(rank=8).trainable_parameters
    reference_parameters = BilinearAVMatchScorer(rank=32).trainable_parameters
    result = {
        "analysis_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "analysis_module_sha256": _sha256(
            REPO_ROOT / "src" / "tiny_omni_decision" / "robomnist_capacity_cv.py"
        ),
        "candidate_run": candidate_run.name,
        "reference_run": reference_run.name,
        "candidate_run_result_sha256": _sha256(candidate_run / "result.json"),
        "candidate_run_code_provenance_sha256": _sha256(
            candidate_run / "run-implementation-provenance.json"
        ),
        "candidate_predictions_sha256": _sha256(candidate_path),
        "reference_predictions_sha256": _sha256(reference_path),
        "same_validation_rows": len(candidate_rows),
        "same_validation_identity_order": True,
        "candidate_parameters": candidate_parameters,
        "reference_parameters": reference_parameters,
        "candidate_metrics": {key: float(value) for key, value in candidate_metrics.items()},
        "reference_metrics": {key: float(value) for key, value in reference_metrics.items()},
        "paired_bootstrap": _paired_metric_bootstrap(
            candidate_probability,
            reference_probability,
            targets,
            pair_ids,
            replicates=2000,
            seed=17,
            minimum_groups=200,
        ),
        "interpretation": (
            "Previously observed development validation; not blind and not used "
            "for candidate selection."
        ),
        "sealed_audit_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.candidate_run, args.reference_run, args.output), indent=2))


if __name__ == "__main__":
    main()
