"""Paired group bootstrap of two frozen DCASE nested-CV prediction sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_dcase2025_nested_room_cv import room_folds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    result_path = path / "result.json"
    config_path = path / "effective-config.json"
    predictions_path = path / "oof-predictions.jsonl"
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
        raise ValueError(f"nested-room integrity gate failed for {path}")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    if (
        len(rows) != result.get("rows")
        or len({row["sample_id"] for row in rows}) != len(rows)
        or sha256_file(predictions_path) != result.get("predictions_sha256")
    ):
        raise ValueError(f"OOF prediction integrity failed for {path}")
    return result, config, rows


def paired_bootstrap(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    replicates: int = 10_000,
    seed: int = 17,
) -> dict[str, object]:
    output_path = candidate_dir / "architecture-comparison-bootstrap-exact-folds.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    ref_result, ref_config, ref_rows = _load(reference_dir)
    cand_result, cand_config, cand_rows = _load(candidate_dir)
    if ref_config.get(
        "query_id_order_sha256", ref_config.get("feature_identity", {}).get("query_ids_sha256")
    ) != cand_config.get(
        "query_id_order_sha256", cand_config.get("feature_identity", {}).get("query_ids_sha256")
    ):
        raise ValueError("nested-CV feature ID-order hashes differ")
    keys = ("sample_id", "room_id", "parent_recording", "target_onscreen")
    if len(ref_rows) != len(cand_rows) or any(
        any(left[key] != right[key] for key in keys)
        for left, right in zip(ref_rows, cand_rows, strict=True)
    ):
        raise ValueError("nested-CV prediction rows are not exactly paired")
    split_seed = cand_config.get("seed")
    if split_seed != ref_config.get("split_seed"):
        raise ValueError("nested-CV split seeds differ")
    generated_folds = room_folds(cand_rows, seed=split_seed)
    generated_layout = [
        {
            "test_room": fold["test_room"],
            "validation_room": fold["validation_room"],
            "train_rooms": fold["train_rooms"],
            "train_rows": len(fold["train_indices"]),
            "validation_rows": len(fold["validation_indices"]),
            "test_rows": len(fold["test_indices"]),
        }
        for fold in generated_folds
    ]
    if generated_layout != ref_config.get("folds"):
        raise ValueError("inner or outer room fold assignments differ")
    expected_fold_pairs = [
        (fold["test_room"], fold["validation_room"]) for fold in generated_layout
    ]
    for run_result in (ref_result, cand_result):
        actual_fold_pairs = [
            (fold["test_room"], fold["validation_room"])
            for fold in run_result.get("fold_results", [])
        ]
        if actual_fold_pairs != expected_fold_pairs:
            raise ValueError("saved run history differs from deterministic nested room folds")
    if replicates < 1 or seed < 0:
        raise ValueError("replicates must be positive and seed non-negative")
    targets = np.asarray([row["target_onscreen"] for row in ref_rows], dtype=np.int64)
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("invalid binary targets")
    comparisons = {}
    rng = np.random.default_rng(seed)
    for group_kind, field in (("parent_recording", "parent_recording"), ("room", "room_id")):
        groups = sorted({str(row[field]) for row in ref_rows})
        group_indexes = [
            np.flatnonzero(np.asarray([str(row[field]) for row in ref_rows], dtype=object) == group)
            for group in groups
        ]
        draws = rng.integers(0, len(groups), size=(replicates, len(groups)))
        denoms = np.asarray([len(indices) for indices in group_indexes], dtype=np.float64)[
            draws
        ].sum(1)
        mode_comparisons = {}
        for mode in ("audio", "video", "joint"):
            ref = np.asarray([row["probabilities"][mode] for row in ref_rows], dtype=np.float64)
            cand = np.asarray([row["probabilities"][mode] for row in cand_rows], dtype=np.float64)
            if (
                ref.shape != (len(targets), 2)
                or cand.shape != ref.shape
                or not np.isfinite(ref).all()
                or not np.isfinite(cand).all()
            ):
                raise ValueError(f"invalid probabilities for {mode}")
            deltas = []
            for indices in group_indexes:
                truth = targets[indices]
                stats = []
                for probabilities in (cand, ref):
                    selected = probabilities[indices]
                    prediction = selected.argmax(1)
                    nll = -np.log(np.clip(selected[np.arange(len(indices)), truth], 1e-15, 1.0))
                    onehot = np.eye(2)[truth]
                    brier = np.square(selected - onehot).sum(1)
                    stats.append(
                        np.asarray(
                            [
                                np.sum(prediction == truth),
                                nll.sum(),
                                brier.sum(),
                            ]
                        )
                    )
                deltas.append(stats[0] - stats[1])
            group_deltas = np.asarray(deltas)
            point = group_deltas.sum(0) / len(targets)
            sampled = group_deltas[draws].sum(1) / denoms[:, None]
            intervals = np.quantile(sampled, [0.025, 0.975], axis=0)
            mode_comparisons[mode] = {
                "candidate_minus_reference": {
                    "accuracy": float(point[0]),
                    "nll": float(point[1]),
                    "brier": float(point[2]),
                },
                "percentile_95_ci": {
                    "accuracy": [float(v) for v in intervals[:, 0]],
                    "nll": [float(v) for v in intervals[:, 1]],
                    "brier": [float(v) for v in intervals[:, 2]],
                },
            }
        comparisons[group_kind] = {
            "groups": len(groups),
            "replicates": replicates,
            "seed": seed,
            "modes": mode_comparisons,
        }
    report = {
        "status": "PASS",
        "scope": (
            "paired posthoc bootstrap of dev-train nested-CV predictions only; "
            "no refit or devtest access"
        ),
        "candidate_minus_reference": "linear affine readout minus width-64 GELU readout",
        "reference_result_sha256": sha256_file(reference_dir / "result.json"),
        "reference_predictions_sha256": ref_result["predictions_sha256"],
        "candidate_result_sha256": sha256_file(candidate_dir / "result.json"),
        "candidate_predictions_sha256": cand_result["predictions_sha256"],
        "sample_count": len(targets),
        "fold_layout_verified_against_reference": True,
        "fold_layout_verified_against_both_run_histories": True,
        "seed": seed,
        "paired_group_differences": comparisons,
        "limit": (
            "Only 78 parent recordings and 9 rooms; intervals are development "
            "diagnostics, not independent confirmation."
        ),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = paired_bootstrap(
        args.reference,
        args.candidate,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(json.dumps({"status": result["status"], "sample_count": result["sample_count"]}))


if __name__ == "__main__":
    main()
