"""Compare frozen AV event readout predictions with position-neutral controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.audit_perception_av_correspondence import sha256_file


def score_rows(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot score an empty prediction set")
    correct, losses, briers, confidences = [], [], [], []
    for row in rows:
        probs = np.asarray(row["probabilities"], dtype=np.float64)
        target = row["target_index"]
        if probs.ndim != 1 or len(probs) < 2 or not 0 <= target < len(probs):
            raise ValueError("invalid candidate probability vector or target")
        if (
            not np.isfinite(probs).all()
            or np.any(probs < 0)
            or not np.isclose(probs.sum(), 1.0, atol=1e-5)
        ):
            raise ValueError("candidate probabilities must be finite, nonnegative, and sum to one")
        correct.append(int(int(probs.argmax()) == target))
        losses.append(float(-np.log(max(probs[target], np.finfo(np.float64).tiny))))
        briers.append(float(np.square(probs - np.eye(1, len(probs), target)[0]).sum()))
        confidences.append(float(probs.max()))
    correct_array = np.asarray(correct, dtype=np.float64)
    confidence_array = np.asarray(confidences, dtype=np.float64)
    ece = 0.0
    for lower in np.arange(0.0, 1.0, 0.1):
        mask = (confidence_array >= lower) & (
            confidence_array < lower + 0.1 if lower < 0.9 else confidence_array <= 1.0
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(confidence_array[mask].mean() - correct_array[mask].mean())
            )
    return {
        "n": len(rows),
        "accuracy": float(correct_array.mean()),
        "nll": float(np.mean(losses)),
        "brier": float(np.mean(briers)),
        "ece": float(ece),
    }


def uniform_predictions(reference_rows: list[dict]) -> list[dict]:
    return [
        {**row, "probabilities": [1.0 / row["candidate_count"]] * row["candidate_count"]}
        for row in reference_rows
    ]


def label_prior_predictions(reference_rows: list[dict], fit_rows: list[dict]):
    counts: dict[int, np.ndarray] = {}
    for row in fit_rows:
        size = row["candidate_count"]
        counts.setdefault(size, np.ones(size, dtype=np.float64))[row["target_index"]] += 1.0
    if not counts:
        raise ValueError("fit rows are required for the label-position prior")
    result = []
    for row in reference_rows:
        size = row["candidate_count"]
        probs = counts.get(size, np.ones(size, dtype=np.float64))
        probs = probs / probs.sum()
        result.append({**row, "probabilities": probs.tolist()})
    return result


def _read_predictions(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("prediction IDs must be unique")
    return rows


def _assert_matched(left: list[dict], right: list[dict]) -> None:
    a = {
        row["id"]: (
            row["video_id"], row["target_index"], row["candidate_count"],
            row["candidate_order_sha256"],
        )
        for row in left
    }
    b = {
        row["id"]: (
            row["video_id"], row["target_index"], row["candidate_count"],
            row["candidate_order_sha256"],
        )
        for row in right
    }
    if a != b:
        raise ValueError("comparison predictions do not match event IDs, labels, and videos")


def paired_video_bootstrap(left: list[dict], right: list[dict], *, draws=10_000, seed=17):
    """Bootstrap the per-event accuracy and NLL differences by video cluster."""
    _assert_matched(left, right)
    left_by_id, right_by_id = ({row["id"]: row for row in rows} for rows in (left, right))
    groups: dict[str, list[str]] = defaultdict(list)
    for row in left:
        groups[row["video_id"]].append(row["id"])
    videos = sorted(groups)
    if len(videos) < 2 or draws < 1:
        raise ValueError("cluster bootstrap needs at least two videos and one draw")
    per_video = {}
    for video, ids in groups.items():
        lrows = [left_by_id[row_id] for row_id in ids]
        rrows = [right_by_id[row_id] for row_id in ids]
        lmetrics, rmetrics = [], []
        for rows, destination in ((lrows, lmetrics), (rrows, rmetrics)):
            correct = 0
            losses = 0.0
            for row in rows:
                probs = np.asarray(row["probabilities"], dtype=np.float64)
                target = row["target_index"]
                correct += int(int(probs.argmax()) == target)
                losses -= float(np.log(max(probs[target], np.finfo(np.float64).tiny)))
            destination.extend((correct, losses))
        per_video[video] = (lmetrics[0], rmetrics[0], lmetrics[1], rmetrics[1], len(ids))
    aggregate = np.asarray([per_video[video] for video in videos], dtype=np.float64)
    rng = np.random.default_rng(seed)
    diffs = {key: np.empty(draws, dtype=np.float64) for key in ("accuracy", "nll")}
    for index in range(draws):
        sampled = aggregate[rng.integers(0, len(videos), size=len(videos))].sum(axis=0)
        diffs["accuracy"][index] = (sampled[0] - sampled[1]) / sampled[4]
        diffs["nll"][index] = (sampled[2] - sampled[3]) / sampled[4]
    return {
        metric: {
            "left_minus_right_point": score_rows(left)[metric] - score_rows(right)[metric],
            "ci_95_percentile": np.quantile(values, [0.025, 0.975]).tolist(),
            "draws": draws,
            "seed": seed,
            "cluster_unit": "video_id",
        }
        for metric, values in diffs.items()
    }


def analyze(joint_path: Path, video_only_path: Path, features_path: Path, output_path: Path):
    if output_path.exists():
        raise ValueError("analysis output must be new")
    joint = _read_predictions(joint_path)
    video_only = _read_predictions(video_only_path)
    _assert_matched(joint, video_only)
    # Keep this analysis module importable in the standard CPU-only [dev] CI
    # environment; the fixed-model runner (and its optional Torch dependency)
    # is needed only when the comparison is actually executed.
    from scripts.train_av_event_source import read_rows

    feature_rows = read_rows(features_path)
    fit_rows = [row for row in feature_rows if row["split"] == "fit"]
    dev = {row["id"]: row for row in feature_rows if row["split"] == "development_validation"}
    if set(dev) != {row["id"] for row in joint}:
        raise ValueError("development predictions do not match pinned feature rows")
    reference = [dev[row["id"]] for row in joint]
    uniform = uniform_predictions(reference)
    prior = label_prior_predictions(reference, fit_rows)
    by_count = {}
    joint_by_id = {row["id"]: row for row in joint}
    video_only_by_id = {row["id"]: row for row in video_only}
    for size in sorted({row["candidate_count"] for row in reference}):
        subset = [row for row in reference if row["candidate_count"] == size]
        ids = [row["id"] for row in subset]
        by_count[str(size)] = {
            "n": len(subset),
            "joint": score_rows([joint_by_id[row_id] for row_id in ids]),
            "video_only": score_rows([video_only_by_id[row_id] for row_id in ids]),
            "uniform": score_rows(uniform_predictions(subset)),
            "fit_label_position_prior": score_rows(label_prior_predictions(subset, fit_rows)),
        }
    result = {
        "status": "PASS_FIXED_AV_EVENT_READOUT_ANALYSIS",
        "source_hashes": {
            "features": sha256_file(features_path),
            "joint_predictions": sha256_file(joint_path),
            "video_only_predictions": sha256_file(video_only_path),
        },
        "development_validation": {
            "joint": score_rows(joint),
            "video_only": score_rows(video_only),
            "uniform": score_rows(uniform),
            "fit_label_position_prior": score_rows(prior),
            "joint_minus_video_only_video_bootstrap": paired_video_bootstrap(joint, video_only),
            "strata_by_candidate_count": by_count,
        },
        "test_opened": False,
        "checkpoint_selection": False,
        "tuning_on_development_validation": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-predictions", type=Path, required=True)
    parser.add_argument("--video-only-predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.joint_predictions,
        args.video_only_predictions,
        args.features,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
