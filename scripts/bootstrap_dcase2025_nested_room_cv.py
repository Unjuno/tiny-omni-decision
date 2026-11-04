"""Add paired group-bootstrap intervals to saved nested-room OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_dcase2025_av_readout import cluster_bootstrap_differences, sha256_file


def summarize(cv_dir: Path, *, replicates: int = 10_000, seed: int = 17) -> dict[str, object]:
    output = cv_dir / "paired-bootstrap.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap summary: {output}")
    result_path = cv_dir / "result.json"
    config_path = cv_dir / "effective-config.json"
    predictions_path = cv_dir / "oof-predictions.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS"
        or result.get("selection_used_outer_test") is not False
        or result.get("development_test_accessed") is not False
        or result.get("sealed_audit_accessed") is not False
        or config.get("outer_protocol")
        != "leave-one-room-out across 9 rooms; every sample tested once"
    ):
        raise ValueError("nested-room OOF integrity gate failed")
    if sha256_file(predictions_path) != result.get("predictions_sha256"):
        raise ValueError("OOF prediction hash mismatch")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != result.get("rows"):
        raise ValueError("OOF prediction row count mismatch")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate OOF prediction IDs")
    targets = np.asarray([row["target_onscreen"] for row in rows], dtype=np.int64)
    parent_ids = [str(row["parent_recording"]) for row in rows]
    room_ids = [str(row["room_id"]) for row in rows]
    probabilities = {
        mode: np.asarray([row["probabilities"][mode] for row in rows], dtype=np.float64)
        for mode in ("audio", "video", "joint")
    }
    if any(
        array.shape != (len(rows), 2) or not np.isfinite(array).all()
        for array in probabilities.values()
    ):
        raise ValueError("invalid OOF probability array")
    report = {
        "status": "PASS",
        "scope": (
            "posthoc paired bootstrap of nested dev-train OOF predictions; "
            "no refit or devtest access"
        ),
        "result_sha256": sha256_file(result_path),
        "config_sha256": sha256_file(config_path),
        "predictions_sha256": sha256_file(predictions_path),
        "rows": len(rows),
        "parent_groups": len(set(parent_ids)),
        "room_groups": len(set(room_ids)),
        "replicates": replicates,
        "seed": seed,
        "paired_group_differences": cluster_bootstrap_differences(
            targets,
            parent_ids,
            room_ids,
            probabilities,
            replicates=replicates,
            seed=seed,
        ),
        "interpretation_limit": (
            "Only 9 held-out rooms; room-cluster intervals are low-power and remain "
            "development evidence."
        ),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    report = summarize(args.cv_dir, replicates=args.replicates, seed=args.seed)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str((args.cv_dir / "paired-bootstrap.json").resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
