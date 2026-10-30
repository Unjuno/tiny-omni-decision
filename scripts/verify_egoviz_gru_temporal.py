"""Reload every fixed EgoViz GRU checkpoint and verify saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_egoviz_gru_temporal import (
    EVALUATION_STEPS,
    OrderedCrossModalGRUScorer,
    sha256_file,
)
from scripts.train_egoviz_vs_sync_readout import _load_samples, _tensor_data


def verify(corpus: Path, run_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result_path = run_dir / "results.json"
    status_path = run_dir / "status.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "FIXED_BUDGET_COMPLETE_EXPLORATORY_DEVELOPMENT":
        raise ValueError("training run is not complete")
    if status.get("results_sha256") != sha256_file(result_path):
        raise ValueError("results hash differs from completed status")
    if result.get("sealed_audit_accessed") is not False:
        raise ValueError("sealed audit isolation flag is not false")

    merge = json.loads((corpus / "merge-report.json").read_text(encoding="utf-8"))
    if merge.get("sealed_audit_accessed") is not False:
        raise ValueError("corpus sealed-audit gate failed")
    for name, expected in result["input_sha256"].items():
        if sha256_file(corpus / name) != expected:
            raise ValueError(f"frozen input changed: {name}")
    rows, video_by_id, sensors = _load_samples(
        corpus, corpus / "video-features.npz", corpus / "sensor-features.npz"
    )
    dev = _tensor_data(rows, video_by_id, sensors, "development")
    history = {int(row["update"]): row for row in result["history"]}
    if tuple(sorted(history)) != EVALUATION_STEPS:
        raise ValueError("fixed evaluation steps differ")

    checkpoint_checks = []
    model = OrderedCrossModalGRUScorer().cpu().eval()
    for step in EVALUATION_STEPS:
        row = history[step]
        state_path = run_dir / f"step-{step}.pt"
        predictions_path = run_dir / f"step-{step}-predictions.jsonl"
        if sha256_file(state_path) != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch at step {step}")
        if sha256_file(predictions_path) != row["prediction_sha256"]:
            raise ValueError(f"prediction hash mismatch at step {step}")
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
        with torch.inference_mode():
            logits = model(dev[0], dev[1]).numpy()
            permuted = model(dev[0], dev[1][:, [2, 0, 3, 1]]).numpy()
        saved = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(saved) != len(dev[4]):
            raise ValueError(f"saved prediction count mismatch at step {step}")
        for index, item in enumerate(saved):
            if (
                item["sample_id"] != dev[4][index]
                or item["group_id"] != str(dev[3][index])
                or int(item["target"]) != int(dev[2][index])
            ):
                raise ValueError(f"prediction identity mismatch at step {step}, row {index}")
            np.testing.assert_allclose(logits[index], item["logits"], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(permuted, logits[:, [2, 0, 3, 1]], atol=1e-6, rtol=1e-6)
        checkpoint_checks.append(
            {
                "update": step,
                "checkpoint_sha256": sha256_file(state_path),
                "prediction_sha256": sha256_file(predictions_path),
                "max_abs_logit_delta": float(
                    np.max(np.abs(logits - np.asarray([item["logits"] for item in saved])))
                ),
                "option_permutation_equivariance": "PASS",
                "sample_id_target_group_alignment": "PASS",
            }
        )

    verification = {
        "status": "PASS",
        "run_results_sha256": sha256_file(result_path),
        "development_ids": len(dev[4]),
        "sealed_audit_accessed": False,
        "checkpoints": checkpoint_checks,
    }
    output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.corpus, args.run_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
