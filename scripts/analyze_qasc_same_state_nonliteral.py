"""Post-hoc sensitivity analysis on saved QASC same-state predictions.

This reads only the already saved QASC TRAIN diagnostic predictions. It does
not load model weights, fit a model, access official validation/test, or read a
sealed audit. The target-not-in-evidence stratum is exploratory and is not used
for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from run_qasc_same_state_criterion import _group_bootstrap

from tiny_omni_decision.qasc_text_criterion_fusion import qasc_probability_metrics

EXPECTED_ROWS = 141
BOOTSTRAP_REPETITIONS = 5_000
BOOTSTRAP_SEED = 1_718
CONDITIONS = (
    "correct_state_and_criterion",
    "correct_state_criterion_absent",
    "zero_state_content_and_criterion",
    "different_deranged_state_and_criterion",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    predictions_path = run_dir / "held-predictions.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not str(report.get("status", "")).startswith("complete_"):
        raise ValueError("source diagnostic run is not complete")
    if report.get("official_validation_read") or report.get("official_test_read"):
        raise ValueError("source diagnostic touched a forbidden official split")
    if report.get("sealed_audit_read"):
        raise ValueError("source diagnostic touched the sealed audit")
    predictions_hash = _sha256(predictions_path)
    if predictions_hash != report.get("prediction_file_sha256"):
        raise ValueError("saved predictions do not match the source report hash")

    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subset = [row for row in rows if row.get("target_in_evidence") is False]
    if len(subset) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} nonliteral targets, got {len(subset)}")
    ids = [row.get("id") for row in subset]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prediction IDs in the sensitivity subset")
    groups = [row.get("state_group_sha256") for row in subset]
    if any(not isinstance(group, str) or len(group) != 64 for group in groups):
        raise ValueError("subset must carry hashed state-group identities")

    targets = torch.tensor([row["target_index"] for row in subset], dtype=torch.long)
    probabilities = {
        name: torch.tensor(
            [row[name]["probabilities"] for row in subset], dtype=torch.float64
        )
        for name in CONDITIONS
    }
    metrics = {
        name: qasc_probability_metrics(probability, targets)
        for name, probability in probabilities.items()
    }
    intervals = _group_bootstrap(
        probabilities,
        targets,
        groups,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "schema_version": 1,
        "analysis": "post-hoc target-not-in-evidence sensitivity analysis",
        "selection_use": False,
        "source_run_directory": str(run_dir),
        "source_report_sha256": _sha256(report_path),
        "source_prediction_sha256": predictions_hash,
        "source_prediction_sha256_expected": report["prediction_file_sha256"],
        "source_rows": len(rows),
        "subset_rows": len(subset),
        "unique_state_groups": len(set(groups)),
        "bootstrap": {
            "unit": "held state group",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "interval": "paired percentile 95%",
        },
        "metrics": metrics,
        "paired_differences_correct_minus_control": intervals,
        "official_validation_read": False,
        "official_test_read": False,
        "sealed_audit_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    output = args.run_dir / "target-not-in-evidence-analysis.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
