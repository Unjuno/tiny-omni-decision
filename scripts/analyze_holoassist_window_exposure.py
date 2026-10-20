"""Post-hoc window-exposure strata for frozen HoloAssist validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_validation_windows(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid validation window JSON on line {line_number}") from error
            sample_id = row.get("id")
            session_id = row.get("session_id")
            start = row.get("window_start_seconds")
            end = row.get("window_end_seconds")
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id in by_id
                or row.get("split") != "validation"
                or not isinstance(row.get("loss_mask"), bool)
                or not isinstance(session_id, str)
                or not session_id
                or isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(start)
                or not math.isfinite(end)
                or end <= start
            ):
                raise ValueError(f"invalid or duplicate validation window on line {line_number}")
            if row["loss_mask"] and row.get("target") not in (0, 1):
                raise ValueError(f"labeled window has invalid binary target on line {line_number}")
            normalized = {
                "id": sample_id,
                "session_id": session_id,
                "target": int(row["target"]) if row["loss_mask"] else None,
                "duration_seconds": float(end - start),
                "loss_mask": row["loss_mask"],
            }
            ordered.append(normalized)
            by_id[sample_id] = normalized
    if not ordered:
        raise ValueError("validation windows must be nonempty")
    return ordered, by_id


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid predictions JSON: {path}") from error
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"prediction file must contain a nonempty JSON list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("id")
        probabilities = np.asarray(row.get("probabilities"), dtype=np.float64)
        session_id = row.get("session_id")
        target = row.get("target")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in result
            or not isinstance(session_id, str)
            or not session_id
            or target not in (0, 1)
            or probabilities.shape != (2,)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-6)
        ):
            raise ValueError(f"invalid or duplicate prediction row in {path}: {sample_id}")
        result[sample_id] = {
            "session_id": session_id,
            "target": int(target),
            "probabilities": probabilities,
        }
    return result


def _metrics(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = [row["id"] for row in rows]
    targets = np.asarray([predictions[sample_id]["target"] for sample_id in ids], dtype=np.int64)
    probabilities = np.stack([predictions[sample_id]["probabilities"] for sample_id in ids])
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    positive = targets == 1
    negative = targets == 0
    confidence = probabilities.max(axis=1)
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))
    one_hot = np.eye(2, dtype=np.float64)[targets]
    brier = np.square(probabilities - one_hot).sum(axis=1)
    ece = 0.0
    for index in range(15):
        lower, upper = index / 15, (index + 1) / 15
        selected = (confidence >= lower) & (
            (confidence < upper) if index < 14 else (confidence <= upper)
        )
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return {
        "n_windows": len(rows),
        "n_sessions": len({row["session_id"] for row in rows}),
        "positive_windows": int(positive.sum()),
        "positive_rate": float(positive.mean()),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": (
            float(((predicted[positive] == 1).mean() + (predicted[negative] == 0).mean()) / 2)
            if positive.any() and negative.any()
            else None
        ),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece_15": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def _paired_session_bootstrap(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_session: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row["id"])
    sessions = sorted(by_session)
    metric_names = ("accuracy", "balanced_accuracy", "nll", "brier", "ece_15")
    if len(sessions) < 2:
        return {
            "status": "not_estimable_fewer_than_two_sessions",
            "sessions": len(sessions),
            "replicates": replicates,
            "seed": seed,
            "intervals": None,
        }
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for _ in range(replicates):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        sampled_rows = [
            {"id": sample_id, "session_id": session}
            for session_value in sampled
            for session in [str(session_value)]
            for sample_id in by_session[session]
        ]
        av_metrics = _metrics(sampled_rows, predictions["av"])
        avs_metrics = _metrics(sampled_rows, predictions["avs"])
        for metric in metric_names:
            base, candidate = av_metrics[metric], avs_metrics[metric]
            if base is not None and candidate is not None:
                draws[metric].append(float(candidate - base))
    return {
        "status": "complete",
        "sessions": len(sessions),
        "replicates": replicates,
        "seed": seed,
        "sampling_unit": "session; keep all selected labeled windows",
        "interval_method": "paired percentile 95%",
        "intervals": {
            metric: (
                {
                    "lower_95": float(np.quantile(values, 0.025)),
                    "upper_95": float(np.quantile(values, 0.975)),
                }
                if values
                else None
            )
            for metric, values in draws.items()
        },
    }


def analyze_files(
    windows_path: Path,
    prediction_paths: dict[str, Path],
    *,
    full_window_seconds: float,
    replicates: int = 10_000,
    seed: int = 1701,
) -> dict[str, Any]:
    if (
        not math.isfinite(full_window_seconds)
        or full_window_seconds <= 0
        or replicates < 100
        or seed < 0
        or not {"av", "sensor_only", "avs"}.issubset(prediction_paths)
    ):
        raise ValueError(
            "positive window size, >=100 replicates, and av/sensor_only/avs inputs required"
        )
    windows_path = windows_path.resolve(strict=True)
    windows, windows_by_id = _load_validation_windows(windows_path)
    labeled = {row["id"] for row in windows if row["loss_mask"]}
    predictions = {
        name: _load_predictions(path.resolve(strict=True))
        for name, path in prediction_paths.items()
    }
    for name, rows in predictions.items():
        if set(rows) != labeled:
            raise ValueError(f"prediction IDs must match labeled windows for {name}")
        for sample_id, prediction in rows.items():
            window = windows_by_id[sample_id]
            if (
                prediction["session_id"] != window["session_id"]
                or prediction["target"] != window["target"]
            ):
                raise ValueError(f"prediction label/session mismatch for {name}: {sample_id}")

    stratum_rows: dict[str, list[dict[str, Any]]] = {"full": [], "tail": []}
    for row in windows:
        duration = row["duration_seconds"]
        if duration > full_window_seconds + 1e-6:
            raise ValueError(f"window exceeds configured duration: {row['id']}")
        if not row["loss_mask"]:
            continue
        name = "full" if math.isclose(duration, full_window_seconds, abs_tol=1e-6) else "tail"
        stratum_rows[name].append(row)
    if any(not rows for rows in stratum_rows.values()):
        raise ValueError("both full and tail labeled windows are required for this audit")

    strata: dict[str, Any] = {}
    for index, (name, rows) in enumerate(stratum_rows.items()):
        metrics = {arm: _metrics(rows, predictions[arm]) for arm in sorted(predictions)}
        av_metrics, avs_metrics = metrics["av"], metrics["avs"]
        comparison = {
            metric: (
                float(avs_metrics[metric] - av_metrics[metric])
                if av_metrics[metric] is not None and avs_metrics[metric] is not None
                else None
            )
            for metric in ("accuracy", "balanced_accuracy", "nll", "brier", "ece_15")
        }
        strata[name] = {
            "n_windows": len(rows),
            "n_sessions": len({row["session_id"] for row in rows}),
            "positive_windows": sum(row["target"] == 1 for row in rows),
            "positive_rate": float(np.mean([row["target"] for row in rows])),
            "metrics": metrics,
            "comparisons": {
                "avs_minus_av": {
                    **comparison,
                    "paired_session_bootstrap": _paired_session_bootstrap(
                        rows, predictions, replicates=replicates, seed=seed + index
                    ),
                }
            },
        }
    labeled_order = [row["id"] for row in windows if row["loss_mask"]]
    return {
        "status": "PASS_POSTHOC_WINDOW_EXPOSURE_STRATIFICATION",
        "purpose": (
            "Assess saved validation predictions by complete versus partial final-window exposure."
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_inference_performed": False,
        "weights_loaded": False,
        "sealed_audit_loaded": False,
        "official_validation_or_test_loaded": False,
        "checkpoint_selection_performed": False,
        "selection_bias_warning": (
            "Both checkpoints were selected on this same development validation split; stratified "
            "metrics and bootstrap intervals are post-hoc diagnostics, not independent "
            "confirmation."
        ),
        "inputs": {
            "validation_windows": {
                "path": str(windows_path),
                "sha256": sha256_file(windows_path),
                "bytes": windows_path.stat().st_size,
            },
            "predictions": {
                name: {
                    "path": str(path.resolve(strict=True)),
                    "sha256": sha256_file(path.resolve(strict=True)),
                    "bytes": path.stat().st_size,
                }
                for name, path in sorted(prediction_paths.items())
            },
        },
        "window_definition": {
            "full_window_seconds": full_window_seconds,
            "tolerance_seconds": 1e-6,
            "classification": (
                "full if duration equals configured size; otherwise shorter duration is tail"
            ),
        },
        "dataset": {
            "total_validation_windows": len(windows),
            "labeled_windows": len(labeled),
            "unlabeled_context_windows_excluded": sum(not row["loss_mask"] for row in windows),
            "labeled_id_order_sha256": hashlib.sha256(
                "\n".join(labeled_order).encode()
            ).hexdigest(),
        },
        "strata": strata,
        "limitations": [
            "One previously observed development validation split; no independent generalization "
            "claim.",
            "Window-level metrics are descriptive; confidence intervals resample sessions, not "
            "windows.",
            "Window duration is associated with target prevalence in this corpus and remains a "
            "task-design confound.",
            "This analysis cannot establish that sensor data are semantically necessary for the "
            "label.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, metavar="ARM=PATH")
    parser.add_argument("--full-window-seconds", type=float, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit result: {args.output}")
    paths: dict[str, Path] = {}
    for item in args.prediction:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in paths:
            raise ValueError("--prediction values must be unique ARM=PATH pairs")
        paths[name] = Path(path)
    result = analyze_files(
        args.windows,
        paths,
        full_window_seconds=args.full_window_seconds,
        replicates=args.replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
