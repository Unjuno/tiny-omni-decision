"""Analyze matched full-extent × unique-coverage factorial predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.native_fit import summarize
from scripts.semantic_state_diagnostic import log_probabilities

CELLS = ("center_repeat", "center_fresh", "full_repeat", "full_fresh")
METRICS = ("accuracy", "nll", "brier", "ece")
DEFAULT_ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
FIT_DIRS = {
    "center_repeat": "native-long-budget-fit-01",
    "center_fresh": "native-coverage-long-fresh-fit-01",
    "full_repeat": "native-full-extent-fit-01",
    "full_fresh": "native-full-extent-fresh-long-fit-01",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines()]


def inspect_conditions(root=DEFAULT_ROOT, step=32_768, fit_dirs=None):
    """Verify available fit receipts, prediction hashes, and identical validation IDs."""
    root = Path(root)
    fit_dirs = FIT_DIRS if fit_dirs is None else fit_dirs
    rows = _load_rows(root / "native-common-corpus-02/validation.jsonl")
    expected_ids = [
        {"id": row["id"], "option_ids": row["option_ids"], "target_index": row["target_index"]}
        for row in rows
    ]
    inventory, score_arrays = {}, {}
    first_ids = None
    for cell, dirname in fit_dirs.items():
        fit = root / dirname
        result_path = fit / "result.json"
        if not result_path.is_file():
            inventory[cell] = {"status": "MISSING", "fit_directory": str(fit)}
            continue
        result = _json(result_path)
        ids_path = fit / "validation-ids.json"
        scores_path = fit / f"validation-{step:05}.npy"
        expected_status = (
            "PASS_NATIVE_FULL_EXTENT_TRAIN"
            if cell.startswith("full_")
            else "PASS_NATIVE_LONG_BUDGET_TRAIN"
        )
        errors = []
        if result.get("status") != expected_status:
            errors.append("unexpected run status")
        if result.get("step") != step:
            errors.append("run endpoint does not match requested fixed step")
        if result.get("sealed_audit_loaded") is not False:
            errors.append("sealed audit must be explicitly recorded as not loaded")
        if not ids_path.is_file() or not scores_path.is_file():
            errors.append("validation IDs or endpoint predictions are missing")
            ids, scores = None, None
        else:
            ids = _json(ids_path)
            if ids != expected_ids:
                errors.append("validation IDs/order differ from the pinned common corpus")
            if first_ids is not None and ids != first_ids:
                errors.append("validation IDs/order differ across factorial cells")
            first_ids = ids if first_ids is None else first_ids
            expected_hashes = result.get("artifacts", {})
            if expected_hashes.get(ids_path.name) != sha256(ids_path):
                errors.append("validation ID hash mismatch")
            if expected_hashes.get(scores_path.name) != sha256(scores_path):
                errors.append("endpoint prediction hash mismatch")
            scores = np.load(scores_path, allow_pickle=False)
            if scores.ndim != 2 or len(scores) != len(rows) or not np.isfinite(scores).all():
                errors.append("endpoint predictions have invalid shape or values")
        inventory[cell] = {
            "status": "INVALID" if errors else "PASS",
            "fit_directory": str(fit),
            "run_status": result.get("status"),
            "step": result.get("step"),
            "best_step": result.get("best_step"),
            "source_commit": result.get("source_commit"),
            "config_sha256": result.get("config_sha256"),
            "result_sha256": sha256(result_path),
            "validation_ids_sha256": sha256(ids_path) if ids_path.is_file() else None,
            "prediction_sha256": sha256(scores_path) if scores_path.is_file() else None,
            "errors": errors,
        }
        if not errors and scores is not None:
            score_arrays[cell] = scores
    missing = [cell for cell, record in inventory.items() if record["status"] == "MISSING"]
    invalid = [cell for cell, record in inventory.items() if record["status"] == "INVALID"]
    return (
        {
            "status": "PASS" if not missing and not invalid else "BLOCKED" if missing else "FAIL",
            "analyzer_sha256": sha256(__file__),
            "goal_complete": False,
            "product_promoted": False,
            "blind_evaluation": False,
            "sealed_audit_loaded": False,
            "step": step,
            "validation_rows": len(rows),
            "missing_cells": missing,
            "invalid_cells": invalid,
            "conditions": inventory,
        },
        rows,
        score_arrays,
    )


def run(root=DEFAULT_ROOT, output=None, step=32_768, seed=17, resamples=10_000, fit_dirs=None):
    audit, rows, scores = inspect_conditions(root, step, fit_dirs=fit_dirs)
    if audit["status"] != "PASS":
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=False)
            (output / "preflight.json").write_text(
                json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf8"
            )
        return audit
    summaries = factorial_effects(rows, scores)
    bootstrap = paired_cluster_bootstrap(rows, scores, seed=seed, resamples=resamples)
    audit["metrics"] = summaries
    audit["paired_cluster_bootstrap"] = bootstrap
    audit["primary_comparison"] = "fixed final step across the four arms; no best-step selection"
    audit["input_hashes"] = {
        cell: {
            key: value[key]
            for key in ("result_sha256", "validation_ids_sha256", "prediction_sha256")
        }
        for cell, value in audit["conditions"].items()
    }
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "factorial-comparison.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--step", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--full-fresh-fit-dir",
        help="fit directory for the full-extent × fresh-coverage cell",
    )
    args = parser.parse_args()
    fit_dirs = dict(FIT_DIRS)
    if args.full_fresh_fit_dir:
        fit_dirs["full_fresh"] = args.full_fresh_fit_dir
    result = run(
        args.root,
        args.output,
        args.step,
        args.seed,
        args.resamples,
        fit_dirs=fit_dirs,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


def _validate(rows, scores_by_cell):
    if set(scores_by_cell) != set(CELLS):
        raise ValueError("exactly four factorial cells are required")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("unique, nonempty validation rows are required")
    modalities = {row["modality"] for row in rows}
    if modalities != {"text", "image", "audio", "video"}:
        raise ValueError("all four modalities must be present")
    arrays = {}
    for cell in CELLS:
        values = np.asarray(scores_by_cell[cell])
        if values.ndim != 2 or len(values) != len(rows):
            raise ValueError("all cells must have the same validation rows and order")
        # summarize enforces finite scores, valid targets, and option padding.
        summarize(rows, values)
        arrays[cell] = values
    return arrays


def factorial_effects(rows, scores_by_cell):
    """Return cell metrics and paired 2×2 simple effects/difference-in-differences."""
    arrays = _validate(rows, scores_by_cell)
    cells = {name: summarize(rows, arrays[name]) for name in CELLS}
    effects = {}
    for name, left, right in (
        ("coverage_effect_center", "center_fresh", "center_repeat"),
        ("coverage_effect_full", "full_fresh", "full_repeat"),
        ("extent_effect_repeat", "full_repeat", "center_repeat"),
        ("extent_effect_fresh", "full_fresh", "center_fresh"),
    ):
        effects[name] = _subtract_summaries(cells[left], cells[right])

    effects["interaction"] = _subtract_effects(
        effects["coverage_effect_full"], effects["coverage_effect_center"]
    )
    return {
        "cells": cells,
        **effects,
        "interpretation": {
            "coverage_effect_center": "center_fresh minus center_repeat",
            "coverage_effect_full": "full_fresh minus full_repeat",
            "extent_effect_repeat": "full_repeat minus center_repeat",
            "extent_effect_fresh": "full_fresh minus center_fresh",
            "interaction": "(full_fresh - full_repeat) - (center_fresh - center_repeat)",
        },
    }


def paired_cluster_bootstrap(rows, scores_by_cell, seed=17, resamples=10_000):
    """Bootstrap linear paired contrasts, clustering rows by modality and group."""
    if type(seed) is not int or seed < 0 or type(resamples) is not int or resamples < 1:
        raise ValueError("nonnegative integer seed and positive resample count required")
    arrays = _validate(rows, scores_by_cell)
    target = np.asarray([row["target_index"] for row in rows], dtype=np.int64)
    row_metrics = {}
    for cell, scores in arrays.items():
        logp = log_probabilities(scores)
        probabilities = np.exp(logp)
        truth = np.eye(scores.shape[1], dtype=np.float64)[target]
        row_metrics[cell] = {
            "accuracy": (scores.argmax(axis=1) == target).astype(np.float64),
            "nll": -logp[np.arange(len(rows)), target],
            "brier": ((probabilities - truth) ** 2).sum(axis=1),
        }

    contrast_cells = {
        "coverage_effect_center": (("center_fresh", 1), ("center_repeat", -1)),
        "coverage_effect_full": (("full_fresh", 1), ("full_repeat", -1)),
        "extent_effect_repeat": (("full_repeat", 1), ("center_repeat", -1)),
        "extent_effect_fresh": (("full_fresh", 1), ("center_fresh", -1)),
        "interaction": (
            ("full_fresh", 1),
            ("full_repeat", -1),
            ("center_fresh", -1),
            ("center_repeat", 1),
        ),
    }
    rng = np.random.default_rng(seed)
    bootstrap_draws = {}
    for modality in ("text", "image", "audio", "video"):
        modality_indices = [i for i, row in enumerate(rows) if row["modality"] == modality]
        groups = sorted({rows[i]["group"] for i in modality_indices})
        if len(groups) < 2:
            raise ValueError("at least two independent groups per modality required")
        bootstrap_draws[modality] = (
            [
                np.asarray(
                    [i for i in modality_indices if rows[i]["group"] == group],
                    dtype=np.int64,
                )
                for group in groups
            ],
            rng.integers(0, len(groups), size=(resamples, len(groups))),
        )
    result = {}
    for contrast, terms in contrast_cells.items():
        result[contrast] = {}
        macro_draws = {metric: [] for metric in ("accuracy", "nll", "brier")}
        modality_results = {}
        for modality in ("text", "image", "audio", "video"):
            indices = np.asarray(
                [i for i, row in enumerate(rows) if row["modality"] == modality],
                dtype=np.int64,
            )
            group_rows, draws = bootstrap_draws[modality]
            sizes = np.asarray([len(item) for item in group_rows], dtype=np.float64)
            denominators = sizes[draws].sum(axis=1)
            modality_results[modality] = {}
            for metric in ("accuracy", "nll", "brier"):
                delta = sum(sign * row_metrics[cell][metric] for cell, sign in terms)
                group_sums = np.asarray([delta[item].sum() for item in group_rows])
                sampled = group_sums[draws].sum(axis=1) / denominators
                point = float(delta[indices].mean())
                bounds = np.quantile(sampled, [0.025, 0.975], method="linear")
                modality_results[modality][metric] = {
                    "estimate": point,
                    "low": float(bounds[0]),
                    "high": float(bounds[1]),
                }
                macro_draws[metric].append(sampled)
        result[contrast]["modalities"] = modality_results
        result[contrast]["macro"] = {}
        for metric, modality_draw_list in macro_draws.items():
            sampled_macro = np.mean(np.stack(modality_draw_list, axis=0), axis=0)
            point_macro = float(
                np.mean([modality_results[m][metric]["estimate"] for m in modality_results])
            )
            bounds = np.quantile(sampled_macro, [0.025, 0.975], method="linear")
            result[contrast]["macro"][metric] = {
                "estimate": point_macro,
                "low": float(bounds[0]),
                "high": float(bounds[1]),
            }
    result["protocol"] = {
        "seed": seed,
        "resamples": resamples,
        "unit": "paired rows clustered by group within modality",
        "interval": "percentile 95%; NumPy linear quantile",
        "metrics": ["accuracy", "nll", "brier"],
        "ece": "reported as a point estimate only; non-linear calibration metric",
    }
    return result


def _subtract_summaries(left, right):
    return {
        "macro": {m: left["macro"][m] - right["macro"][m] for m in METRICS},
        "modalities": {
            modality: {
                metric: left["modality_macro"][modality][metric]
                - right["modality_macro"][modality][metric]
                for metric in METRICS
            }
            for modality in ("text", "image", "audio", "video")
        },
    }


def _subtract_effects(left, right):
    return {
        "macro": {m: left["macro"][m] - right["macro"][m] for m in METRICS},
        "modalities": {
            modality: {
                metric: left["modalities"][modality][metric] - right["modalities"][modality][metric]
                for metric in METRICS
            }
            for modality in ("text", "image", "audio", "video")
        },
    }


if __name__ == "__main__":
    main()
