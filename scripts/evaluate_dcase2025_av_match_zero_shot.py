"""Evaluate direct cosine alignment of frozen DCASE audio/video features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(targets), 4)
        or not len(targets)
        or not np.isfinite(probabilities).all()
        or not np.allclose(probabilities.sum(1), 1.0, atol=1e-7)
        or np.any(targets < 0)
        or np.any(targets > 3)
    ):
        raise ValueError("expected normalized four-choice probabilities and valid targets")
    predicted = probabilities.argmax(1)
    confidence = probabilities.max(1)
    onehot = np.eye(4, dtype=np.float64)[targets]
    ece = 0.0
    for bucket in range(15):
        lower = bucket / 15
        upper = (bucket + 1) / 15
        mask = (confidence >= lower) & (confidence <= upper if bucket == 14 else confidence < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float((predicted[mask] == targets[mask]).mean()) - float(confidence[mask].mean())
            )
    return {
        "count": len(targets),
        "accuracy": float((predicted == targets).mean()),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-15, 1)).mean()
        ),
        "brier": float(np.square(probabilities - onehot).sum(1).mean()),
        "ece_15": ece,
        "mean_confidence": float(confidence.mean()),
    }


def cosine_probabilities(audio: np.ndarray, video_candidates: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    video_candidates = np.asarray(video_candidates, dtype=np.float64)
    if (
        audio.ndim != 2
        or video_candidates.ndim != 3
        or video_candidates.shape[0] != len(audio)
        or video_candidates.shape[1] != 4
        or video_candidates.shape[2] != audio.shape[1]
        or not np.isfinite(audio).all()
        or not np.isfinite(video_candidates).all()
    ):
        raise ValueError("audio [N,D] and video candidates [N,4,D] must be finite and aligned")
    audio_norm = np.linalg.norm(audio, axis=1, keepdims=True)
    video_norm = np.linalg.norm(video_candidates, axis=2, keepdims=True)
    audio_unit = audio / np.maximum(audio_norm, 1e-12)
    video_unit = video_candidates / np.maximum(video_norm, 1e-12)
    logits = np.einsum("nd,nkd->nk", audio_unit, video_unit)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _row_sums(probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    prediction = probabilities.argmax(1)
    nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-15, 1.0))
    brier = np.square(probabilities - np.eye(4)[targets]).sum(1)
    return np.column_stack((prediction == targets, nll, brier)).astype(np.float64)


def paired_bootstrap(
    new_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    targets: np.ndarray,
    parents: list[str],
    rooms: list[str],
    *,
    replicates: int = 10_000,
    seed: int = 17,
) -> dict[str, object]:
    if replicates < 1 or seed < 0 or not (len(parents) == len(rooms) == len(targets)):
        raise ValueError("invalid bootstrap arguments")
    rng = np.random.default_rng(seed)
    output: dict[str, object] = {}
    for name, group_ids in (("parent_recording", parents), ("room", rooms)):
        groups = sorted(set(group_ids))
        membership = np.asarray(group_ids, dtype=object)
        indices = [np.flatnonzero(membership == group) for group in groups]
        draws = rng.integers(0, len(groups), size=(replicates, len(groups)))
        denominators = np.asarray([len(index) for index in indices])[draws].sum(1)
        new_sums = np.stack(
            [_row_sums(new_probabilities[index], targets[index]).sum(0) for index in indices]
        )
        ref_sums = np.stack(
            [_row_sums(reference_probabilities[index], targets[index]).sum(0) for index in indices]
        )
        delta = new_sums - ref_sums
        point = delta.sum(0) / len(targets)
        sampled = delta[draws].sum(1) / denominators[:, None]
        interval = np.quantile(sampled, [0.025, 0.975], axis=0)
        output[name] = {
            "groups": len(groups),
            "replicates": replicates,
            "seed": seed,
            "point": {
                metric: float(point[index])
                for index, metric in enumerate(("accuracy", "nll", "brier"))
            },
            "percentile_95_ci": {
                metric: [float(value) for value in interval[:, index]]
                for index, metric in enumerate(("accuracy", "nll", "brier"))
            },
        }
    return output


def run(
    features_dir: Path,
    task_dir: Path,
    reference_run_dir: Path,
    output_dir: Path,
    *,
    replicates: int = 10_000,
    seed: int = 17,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    task_config_path = task_dir / "effective-config.json"
    task_config = json.loads(task_config_path.read_text(encoding="utf-8"))
    task_manifest_path = task_dir / "task-manifest.jsonl"
    task_preflight = json.loads((task_dir / "task-preflight.json").read_text(encoding="utf-8"))
    if (
        task_preflight.get("status") != "PASS"
        or task_preflight.get("positive_matches_query_parent") is not True
        or task_preflight.get("candidate_parent_unique_per_query") is not True
        or task_config.get("devtest_accessed") is not False
        or task_config.get("sealed_audit_accessed") is not False
        or task_config.get("source_split") != "DCASE official dev-train frozen features only"
        or sha256_file(task_config_path) != task_preflight.get("effective_config_sha256")
        or sha256_file(task_manifest_path) != task_preflight.get("task_manifest_sha256")
    ):
        raise ValueError("frozen task preflight failed")
    source_manifest_path = features_dir / "query-manifest.jsonl"
    feature_preflight_path = features_dir / "preflight.json"
    audio_path = features_dir / "features_audio.npy"
    video_path = features_dir / "features_video.npy"
    for path, expected in (
        (source_manifest_path, task_config.get("source_manifest_sha256")),
        (feature_preflight_path, task_config.get("source_preflight_sha256")),
        (audio_path, task_config.get("audio_features_sha256")),
        (video_path, task_config.get("video_features_sha256")),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen feature hash mismatch: {path.name}")

    examples = [
        json.loads(line) for line in task_manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    source_rows = [
        json.loads(line) for line in source_manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    feature_progress = json.loads((features_dir / "progress.json").read_text(encoding="utf-8"))
    if feature_progress.get("identity") != task_config.get("feature_identity"):
        raise ValueError("frozen feature identity differs from the task configuration")
    reference_path = reference_run_dir / "oof-predictions.jsonl"
    reference_result = json.loads((reference_run_dir / "result.json").read_text(encoding="utf-8"))
    if (
        reference_result.get("status") != "PASS"
        or reference_result.get("selection_used_outer_test") is not False
        or reference_result.get("development_test_accessed") is not False
        or reference_result.get("sealed_audit_accessed") is not False
        or sha256_file(reference_path) != reference_result.get("predictions_sha256")
    ):
        raise ValueError("reference CV predictions failed integrity checks")
    reference_rows = [
        json.loads(line) for line in reference_path.read_text(encoding="utf-8").splitlines()
    ]
    if (
        not examples
        or len(reference_rows) != len(examples)
        or len({row["query_id"] for row in examples}) != len(examples)
        or [row["query_id"] for row in reference_rows] != [row["query_id"] for row in examples]
    ):
        raise ValueError("task and reference predictions are incomplete or misaligned")
    audio_features = np.load(audio_path, mmap_mode="r")
    video_features = np.load(video_path, mmap_mode="r")
    audio_rows = []
    video_rows = []
    targets = []
    parents = []
    rooms = []
    for example, reference in zip(examples, reference_rows, strict=True):
        query_row = source_rows[int(example["query_row"])]
        if (
            query_row.get("sample_id") != example.get("query_sample_id")
            or query_row.get("parent_recording") != example.get("query_parent")
            or query_row.get("room_id") != example.get("room_id")
            or reference.get("query_parent") != example.get("query_parent")
            or reference.get("room_id") != example.get("room_id")
            or reference.get("target_index") != example.get("target_index")
        ):
            raise ValueError("task, feature, and reference row identities disagree")
        options = example.get("options", [])
        target_index = int(example["target_index"])
        if (
            len(options) != 4
            or len({option.get("parent_recording") for option in options}) != 4
            or not 0 <= target_index < 4
            or options[target_index].get("sample_id") != example.get("query_sample_id")
        ):
            raise ValueError("each query must have four options")
        for option in options:
            option_row = source_rows[int(option["feature_row"])]
            if (
                option_row.get("sample_id") != option.get("sample_id")
                or option_row.get("parent_recording") != option.get("parent_recording")
                or option_row.get("room_id") != example.get("room_id")
                or option_row.get("event_class") != example.get("event_class")
            ):
                raise ValueError("candidate feature row does not match its frozen task identity")
        audio_rows.append(np.asarray(audio_features[int(example["query_row"])]))
        video_rows.append(
            np.stack([np.asarray(video_features[int(option["feature_row"])]) for option in options])
        )
        targets.append(int(example["target_index"]))
        parents.append(str(example["query_parent"]))
        rooms.append(str(example["room_id"]))
    audio_array = np.stack(audio_rows)
    video_array = np.stack(video_rows)
    target_array = np.asarray(targets, dtype=np.int64)
    probabilities = cosine_probabilities(audio_array, video_array)
    trained_probabilities = np.asarray(
        [row["probabilities"]["joint"] for row in reference_rows], dtype=np.float64
    )
    uniform = np.full((len(target_array), 4), 0.25, dtype=np.float64)

    output_dir.mkdir(parents=True)
    config = {
        "method": (
            "independent L2 normalization then direct cosine(audio, each video option), "
            "softmax temperature 1"
        ),
        "fitting": "none; no learned projection, calibration, or label-dependent parameter",
        "seed": seed,
        "bootstrap_replicates": replicates,
        "features": str(features_dir),
        "task_manifest_sha256": sha256_file(task_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_preflight_sha256": sha256_file(feature_preflight_path),
        "audio_features_sha256": sha256_file(audio_path),
        "video_features_sha256": sha256_file(video_path),
        "development_test_accessed": False,
        "sealed_audit_accessed": False,
    }
    (output_dir / "effective-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prediction_path = output_dir / "predictions.jsonl"
    rows = [
        {
            "query_id": example["query_id"],
            "query_parent": parents[index],
            "room_id": rooms[index],
            "target_index": int(target_array[index]),
            "probabilities": probabilities[index].tolist(),
        }
        for index, example in enumerate(examples)
    ]
    prediction_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "status": "PASS",
        "queries": len(rows),
        "rooms": len(set(rooms)),
        "unique_query_parents": len(set(parents)),
        "metrics": {
            "raw_cosine": metrics(probabilities, target_array),
            "uniform": metrics(uniform, target_array),
            "trained_nested_cv": metrics(trained_probabilities, target_array),
        },
        "paired_raw_cosine_minus_uniform": paired_bootstrap(
            probabilities, uniform, target_array, parents, rooms, replicates=replicates, seed=seed
        ),
        "paired_raw_cosine_minus_trained_cv": paired_bootstrap(
            probabilities,
            trained_probabilities,
            target_array,
            parents,
            rooms,
            replicates=replicates,
            seed=seed,
        ),
        "prediction_sha256": sha256_file(prediction_path),
        "development_test_accessed": False,
        "sealed_audit_accessed": False,
        "interpretation_limit": (
            "same exposed dev-train task; no independent test, only 78 parents and 9 rooms"
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = run(
        args.features,
        args.task,
        args.reference_run,
        args.output,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(json.dumps({"status": result["status"], "queries": result["queries"]}))


if __name__ == "__main__":
    main()
