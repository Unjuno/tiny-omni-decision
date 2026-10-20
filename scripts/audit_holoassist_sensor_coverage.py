"""Audit zero sensor-coverage bins against fixed-window zero padding."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.audit_holoassist_metadata import sha256_file

BIN_COUNT = 30
COVERAGE_START = 18


def audit_split(rows: list[dict], sample_ids: np.ndarray, features: np.ndarray) -> dict:
    """Summarize padding, observed coverage, and labels without fitting a model."""
    if features.shape != (len(rows), BIN_COUNT, 21) or sample_ids.shape != (len(rows),):
        raise ValueError("window metadata and IMU feature shapes do not align")
    if sample_ids.astype(str).tolist() != [str(row["id"]) for row in rows]:
        raise ValueError("window metadata and IMU sample ID order differ")

    coverage = features[:, :, COVERAGE_START : COVERAGE_START + 3]
    if not np.array_equal(coverage[:, :, 0], coverage[:, :, 1]) or not np.array_equal(
        coverage[:, :, 0], coverage[:, :, 2]
    ):
        raise ValueError("sensor coverage channels unexpectedly differ")

    padding = np.zeros((len(rows), BIN_COUNT), dtype=bool)
    durations = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        start = float(row["window_start_seconds"])
        end = float(row["window_end_seconds"])
        duration = end - start
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or duration <= 0
            or duration > BIN_COUNT
            or abs(start - round(start)) > 1e-8
        ):
            raise ValueError(f"invalid temporal window boundaries for {row['id']}")
        durations[index] = duration
        valid_bins = min(BIN_COUNT, math.ceil(end - 1e-12) - int(start))
        padding[index, max(0, valid_bins) :] = True

    zero_coverage = np.all(coverage == 0, axis=2)
    valid = ~padding
    targets = np.asarray(
        [-1 if row.get("target") is None else int(row["target"]) for row in rows],
        dtype=np.int8,
    )
    final = np.asarray(
        [int(row["window_index"]) == int(row["window_count"]) - 1 for row in rows],
        dtype=bool,
    )

    def bin_summary(mask: np.ndarray) -> dict:
        count = int(mask.sum())
        zeros = int((mask & zero_coverage).sum())
        return {
            "bin_count": count,
            "zero_coverage_bin_count": zeros,
            "zero_coverage_rate": zeros / count if count else None,
        }

    def group_summary(mask: np.ndarray) -> dict:
        indices = np.flatnonzero(mask & (targets >= 0))
        if not len(indices):
            return {"labeled_window_count": 0}
        return {
            "labeled_window_count": int(len(indices)),
            "positive_rate": float(targets[indices].mean()),
            "mean_window_seconds": float(durations[indices].mean()),
            "mean_padded_bins": float(padding[indices].sum(axis=1).mean()),
            "mean_action_count": float(
                np.asarray([int(rows[index]["action_count"]) for index in indices]).mean()
            ),
            "incorrect_action_count": int(
                sum(int(rows[index]["incorrect_action_count"]) for index in indices)
            ),
            "total_action_count": int(
                sum(int(rows[index]["action_count"]) for index in indices)
            ),
            "incorrect_action_rate": float(
                sum(int(rows[index]["incorrect_action_count"]) for index in indices)
                / sum(int(rows[index]["action_count"]) for index in indices)
            ),
            "incorrect_actions_per_100_window_seconds": float(
                100.0
                * sum(int(rows[index]["incorrect_action_count"]) for index in indices)
                / sum(durations[index] for index in indices)
            ),
        }

    total_zero = int(zero_coverage.sum())
    pad_zero = int((zero_coverage & padding).sum())
    labeled = targets >= 0
    return {
        "window_count": len(rows),
        "labeled_window_count": int(labeled.sum()),
        "positive_rate": float(targets[labeled].mean()),
        "coverage_channels_identical": True,
        "coverage_bins": {
            "total": bin_summary(np.ones_like(padding, dtype=bool)),
            "valid_observation": bin_summary(valid),
            "zero_padding": bin_summary(padding),
            "zero_bins_explained_by_padding_percent": (
                100.0 * pad_zero / total_zero if total_zero else None
            ),
            "nonzero_coverage_bins_in_padding": int((~zero_coverage & padding).sum()),
        },
        "labeled_window_groups": {
            "final_window": group_summary(final),
            "nonfinal_window": group_summary(~final),
        },
        "windows_with_any_zero_coverage_in_valid_bins": int(
            np.any(zero_coverage & valid, axis=1).sum()
        ),
    }


def audit_dataset(windows_dir: Path, imu_dir: Path) -> dict:
    """Audit frozen train/validation windows against the aligned IMU archive."""
    result = {
        "schema_version": 1,
        "sealed_audit_loaded": False,
        "inputs": {},
        "splits": {},
    }
    for split in ("train", "validation"):
        windows_path = windows_dir / f"{split}.jsonl"
        features_path = imu_dir / f"{split}.npz"
        rows = [
            json.loads(line)
            for line in windows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with np.load(features_path, allow_pickle=False) as archive:
            stats = audit_split(rows, archive["sample_id"], archive["features"])
        result["inputs"][str(windows_path)] = sha256_file(windows_path)
        result["inputs"][str(features_path)] = sha256_file(features_path)
        result["splits"][split] = stats
    receipt_path = imu_dir / "receipt.json"
    result["inputs"][str(receipt_path)] = sha256_file(receipt_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--imu-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    result = audit_dataset(args.windows_dir, args.imu_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
