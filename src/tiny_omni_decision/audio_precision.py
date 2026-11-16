"""Pure request projection and numeric-comparison helpers for Path A audio."""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np

CONDITIONS = ("random_distractors", "hard_distractors")
PARITY_THRESHOLD = 1e-5

def _request_fields(source: dict[str, Any]) -> dict[str, Any]:
    """Project the fixed artifact onto inference inputs; ignore all labels."""
    options_by_condition = source["options_by_condition"]
    if set(options_by_condition) != set(CONDITIONS):
        raise ValueError("unexpected LibriSpeech distractor conditions")
    projected_options = {
        condition: [str(option) for option in options_by_condition[condition]]
        for condition in CONDITIONS
    }
    if any(len(options) != 4 or len(set(options)) != 4 for options in projected_options.values()):
        raise ValueError("each condition must contain four unique options")
    return {
        "sample_id": str(source["id"]),
        "audio_path": str(source["audio_path"]),
        "audio_sha256": str(source["audio_sha256"]),
        "duration_seconds": float(source["duration_seconds"]),
        "options_by_condition": projected_options,
    }


def _compare(cpu: dict[str, Any], cuda: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        cpu_rows = cpu["prediction_data"][condition]
        cuda_rows = cuda["prediction_data"][condition]
        if len(cpu_rows) != len(cuda_rows):
            raise ValueError("CPU/CUDA prediction row counts differ")
        deltas: list[float] = []
        decision_maxima: list[float] = []
        changed = 0
        for left, right in zip(cpu_rows, cuda_rows, strict=True):
            if (
                left["sample_id"] != right["sample_id"]
                or left["audio_sha256"] != right["audio_sha256"]
            ):
                raise ValueError("CPU/CUDA request order or audio identity differs")
            a = np.asarray(left["probabilities"], dtype=np.float64)
            b = np.asarray(right["probabilities"], dtype=np.float64)
            if a.shape != (4,) or b.shape != (4,):
                raise ValueError("expected four option probabilities")
            row_delta = np.abs(a - b)
            deltas.extend(row_delta.tolist())
            decision_maxima.append(float(row_delta.max()))
            changed += int(a.argmax() != b.argmax())
        ordered = sorted(decision_maxima)
        result[condition] = {
            "decision_count": len(decision_maxima),
            "probability_count": len(deltas),
            "max_abs_probability_delta": max(deltas),
            "mean_abs_probability_delta": statistics.fmean(deltas),
            "p50_decision_max_delta": ordered[len(ordered) // 2],
            "p95_decision_max_delta": ordered[int(0.95 * (len(ordered) - 1))],
            "p99_decision_max_delta": ordered[int(0.99 * (len(ordered) - 1))],
            "decisions_over_1e-5": sum(value > PARITY_THRESHOLD for value in decision_maxima),
            "argmax_changed_decisions": changed,
            "strict_probability_parity_passed": max(deltas) <= PARITY_THRESHOLD,
        }
    return result

