"""Evaluate frozen DCASE AV readouts once on the post-freeze room-held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from tiny_omni_decision.dcase2025_readout import DCASE2025Readout, binary_probability_metrics

BOOTSTRAP_SEED = 17
BOOTSTRAP_REPLICATES = 10_000
MODES = ("audio", "video", "joint")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_readout(path: Path, mode: str) -> tuple[DCASE2025Readout, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    model = DCASE2025Readout(mode, hidden_width=64)
    state = {
        name.removeprefix("model."): torch.from_numpy(value)
        for name, value in arrays.items()
        if name.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    for name in ("mean_audio", "scale_audio", "mean_video", "scale_video"):
        if (
            name not in arrays
            or arrays[name].shape != (768,)
            or not np.isfinite(arrays[name]).all()
        ):
            raise ValueError(f"readout checkpoint has invalid {name}")
    return model, arrays


def standardize(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = (np.asarray(features, dtype=np.float32) - mean.astype(np.float32)) / scale.astype(
        np.float32
    )
    if not np.isfinite(values).all():
        raise ValueError("non-finite standardized feature")
    return values


def parent_derangement(parent_ids: list[str], seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    groups = sorted(set(parent_ids))
    if len(groups) < 2:
        raise ValueError("cross-modal shuffle requires at least two parent recordings")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(groups))
    while np.any(permutation == np.arange(len(groups))):
        permutation = rng.permutation(len(groups))
    mapping = {source: groups[int(permutation[index])] for index, source in enumerate(groups)}
    indices_by_parent = {
        parent: np.flatnonzero(np.asarray(parent_ids, dtype=object) == parent) for parent in groups
    }
    donor_indices = np.empty(len(parent_ids), dtype=np.int64)
    for parent in groups:
        rows = indices_by_parent[parent]
        donors = indices_by_parent[mapping[parent]]
        positions = np.floor((np.arange(len(rows)) + 0.5) * len(donors) / len(rows)).astype(int)
        donor_indices[rows] = donors[np.minimum(positions, len(donors) - 1)]
    if any(parent_ids[index] == parent_ids[donor] for index, donor in enumerate(donor_indices)):
        raise AssertionError("parent-level shuffle retained a same-recording pair")
    return donor_indices


def cluster_bootstrap_differences(
    targets: np.ndarray,
    parent_ids: list[str],
    room_ids: list[str],
    probabilities: dict[str, np.ndarray],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Paired percentile intervals for joint-minus-unimodal metric differences."""
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) != len(parent_ids) or len(targets) != len(room_ids) or not len(targets):
        raise ValueError("bootstrap metadata must match non-empty targets")
    rng = np.random.default_rng(seed)
    results: dict[str, object] = {}
    for group_name, group_ids in (("parent_recording", parent_ids), ("room", room_ids)):
        groups = sorted(set(group_ids))
        positions = np.asarray(group_ids, dtype=object)
        counts = np.asarray([np.sum(positions == group) for group in groups], dtype=np.float64)
        if np.any(counts == 0):
            raise AssertionError("empty bootstrap group")
        draws = rng.integers(0, len(groups), size=(replicates, len(groups)))
        denominator = counts[draws].sum(axis=1)
        group_sums: dict[str, np.ndarray] = {}
        for name, probs in probabilities.items():
            sums = []
            for group in groups:
                mask = positions == group
                selected = probs[mask]
                truth = targets[mask]
                prediction = selected.argmax(axis=1)
                nll = -np.log(np.clip(selected[np.arange(len(truth)), truth], 1e-15, 1.0))
                one_hot = np.eye(2, dtype=np.float64)[truth]
                brier = np.square(selected - one_hot).sum(axis=1)
                sums.append(
                    [
                        float(np.sum(prediction == truth)),
                        float(nll.sum()),
                        float(brier.sum()),
                    ]
                )
            group_sums[name] = np.asarray(sums, dtype=np.float64)
        comparisons = {}
        for baseline in ("audio", "video"):
            delta_sums = group_sums["joint"] - group_sums[baseline]
            draws_sums = delta_sums[draws].sum(axis=1) / denominator[:, None]
            point = delta_sums.sum(axis=0) / counts.sum()
            intervals = np.quantile(draws_sums, [0.025, 0.975], axis=0)
            comparisons[f"joint_minus_{baseline}"] = {
                "accuracy_difference": float(point[0]),
                "nll_difference": float(point[1]),
                "brier_difference": float(point[2]),
                "percentile_95_ci": {
                    "accuracy_difference": [float(x) for x in intervals[:, 0]],
                    "nll_difference": [float(x) for x in intervals[:, 1]],
                    "brier_difference": [float(x) for x in intervals[:, 2]],
                },
            }
        results[group_name] = {
            "independent_groups": len(groups),
            "low_power_under_200_groups": len(groups) < 200,
            "replicates": replicates,
            "seed": seed,
            "comparisons": comparisons,
        }
    return results


def summarize_by(values: list[str] | np.ndarray, targets: np.ndarray, probs: np.ndarray):
    values = np.asarray(values, dtype=object)
    report = {}
    for value in sorted(set(values.tolist())):
        mask = values == value
        report[str(value)] = binary_probability_metrics(probs[mask], targets[mask])
    return report


def _predict(model: DCASE2025Readout, audio: np.ndarray, video: np.ndarray, event: np.ndarray):
    with torch.inference_mode():
        output = model(
            torch.from_numpy(np.asarray(audio, dtype=np.float32)),
            torch.from_numpy(np.asarray(video, dtype=np.float32)),
            torch.from_numpy(np.asarray(event, dtype=np.int64)),
        ).softmax(-1)
    values = output.numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError("readout produced non-finite probabilities")
    return values


def evaluate_once(
    feature_dir: Path,
    training_dir: Path,
    candidate_freeze: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to rerun or overwrite a DCASE dev-test evaluation: {output_dir}"
        )
    freeze = json.loads(candidate_freeze.read_text(encoding="utf-8"))
    training_result_path = training_dir / "result.json"
    training_config_path = training_dir / "effective-config.json"
    training_result = json.loads(training_result_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_DEVTEST" or freeze.get(
        "devtest_annotation_content_opened"
    ) is not False:
        raise ValueError("candidate was not frozen before dev-test annotation access")
    if sha256_file(training_result_path) != freeze.get("training_result_sha256"):
        raise ValueError("frozen training-result hash mismatch")
    if sha256_file(training_config_path) != freeze.get("effective_config_sha256"):
        raise ValueError("frozen training-config hash mismatch")
    progress = json.loads((feature_dir / "progress.json").read_text(encoding="utf-8"))
    preflight = json.loads((feature_dir / "preflight.json").read_text(encoding="utf-8"))
    if (
        progress.get("status") != "PASS"
        or progress.get("identity", {}).get("split") != "devtest"
        or progress.get("completed_rows") != progress.get("total_queries")
        or progress.get("valid_rows") != progress.get("total_queries")
        or preflight.get("status") != "PASS"
        or preflight.get("candidate_freeze_sha256") != sha256_file(candidate_freeze)
        or preflight.get("sealed_audit_accessed") is not False
    ):
        raise ValueError("dev-test feature or post-freeze preflight gate failed")
    rows = [
        json.loads(line)
        for line in (feature_dir / "query-manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ids_hash = hashlib.sha256("".join(f"{row['sample_id']}\n" for row in rows).encode()).hexdigest()
    if (
        len(rows) != progress["total_queries"]
        or ids_hash != progress["identity"]["query_ids_sha256"]
    ):
        raise ValueError("dev-test query manifest differs from its frozen row-order hash")
    targets = np.asarray([row["target_onscreen"] for row in rows], dtype=np.int64)
    events = np.asarray([row["event_class"] for row in rows], dtype=np.int64)
    parent_ids = [str(row["parent_recording"]) for row in rows]
    room_ids = [str(row["room_id"]) for row in rows]
    if not np.isin(targets, (0, 1)).all() or not np.isin(events, np.arange(13)).all():
        raise ValueError("unexpected dev-test labels")
    audio_raw = np.load(feature_dir / "features_audio.npy", mmap_mode="r")
    video_raw = np.load(feature_dir / "features_video.npy", mmap_mode="r")
    if audio_raw.shape != (len(rows), 768) or video_raw.shape != audio_raw.shape:
        raise ValueError("dev-test feature array shape mismatch")
    masks = np.load(feature_dir / "valid_mask.npy", mmap_mode="r")
    processed = np.load(feature_dir / "processed_mask.npy", mmap_mode="r")
    if not masks.all() or not processed.all():
        raise ValueError("dev-test feature arrays contain incomplete rows")
    if len(set(parent_ids)) != preflight.get("unique_parent_recordings"):
        raise ValueError("dev-test parent count differs from post-freeze preflight")

    checkpoint_paths = {mode: training_dir / f"{mode}-readout.npz" for mode in MODES}
    checkpoint_hashes = {mode: sha256_file(path) for mode, path in checkpoint_paths.items()}
    if checkpoint_hashes != freeze.get("candidate_checkpoint_sha256"):
        raise ValueError("frozen candidate checkpoint hashes changed")
    checkpoint_probabilities = {}
    checkpoint_states = {}
    normalized_audio = None
    normalized_video = None
    for mode in MODES:
        model, state = load_frozen_readout(checkpoint_paths[mode], mode)
        checkpoint_states[mode] = (model, state)
        if mode == "audio":
            normalized_audio = standardize(audio_raw, state["mean_audio"], state["scale_audio"])
            checkpoint_probabilities["audio"] = _predict(
                model, normalized_audio, np.zeros_like(normalized_audio), events
            )
        elif mode == "video":
            normalized_video = standardize(video_raw, state["mean_video"], state["scale_video"])
            checkpoint_probabilities["video"] = _predict(
                model, np.zeros_like(normalized_video), normalized_video, events
            )
        else:
            normalized_audio = standardize(audio_raw, state["mean_audio"], state["scale_audio"])
            normalized_video = standardize(video_raw, state["mean_video"], state["scale_video"])
            checkpoint_probabilities["joint"] = _predict(
                model, normalized_audio, normalized_video, events
            )

    assert normalized_audio is not None and normalized_video is not None
    joint_model = checkpoint_states["joint"][0]
    predictions = dict(checkpoint_probabilities)
    predictions["joint_missing_audio"] = _predict(
        joint_model, np.zeros_like(normalized_audio), normalized_video, events
    )
    predictions["joint_missing_video"] = _predict(
        joint_model, normalized_audio, np.zeros_like(normalized_video), events
    )
    donors = parent_derangement(parent_ids)
    predictions["joint_shuffle_audio_parent"] = _predict(
        joint_model, normalized_audio[donors], normalized_video, events
    )
    predictions["joint_shuffle_video_parent"] = _predict(
        joint_model, normalized_audio, normalized_video[donors], events
    )

    train_targets = training_result["train_targets"]
    train_total = train_targets["offscreen"] + train_targets["onscreen"]
    majority_probs = np.tile(
        [train_targets["offscreen"] / train_total, train_targets["onscreen"] / train_total],
        (len(targets), 1),
    )
    metrics = {
        mode: binary_probability_metrics(probabilities, targets)
        for mode, probabilities in predictions.items()
    }
    metrics["train_prevalence_baseline"] = binary_probability_metrics(majority_probs, targets)
    per_room = {
        mode: summarize_by(room_ids, targets, probs) for mode, probs in predictions.items()
    }
    per_event_class = {
        mode: summarize_by(events, targets, probs) for mode, probs in predictions.items()
    }
    paired_bootstrap = cluster_bootstrap_differences(
        targets,
        parent_ids,
        room_ids,
        checkpoint_probabilities,
        replicates=bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    output_dir.mkdir(parents=True)
    prediction_path = output_dir / "devtest-predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            result = {
                "sample_id": row["sample_id"],
                "parent_recording": parent_ids[index],
                "room_id": room_ids[index],
                "event_class": int(events[index]),
                "target_onscreen": int(targets[index]),
                "probabilities": {
                    name: values[index].tolist() for name, values in predictions.items()
                },
            }
            stream.write(json.dumps(result, sort_keys=True) + "\n")
    result = {
        "status": "PASS",
        "evaluation_type": "single post-freeze DCASE public dev-test evaluation; not sealed audit",
        "started_at_utc": progress.get("last_started_at"),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_freeze_sha256": sha256_file(candidate_freeze),
        "training_result_sha256": sha256_file(training_result_path),
        "training_config_sha256": sha256_file(training_config_path),
        "feature_run_identity": progress["identity"],
        "feature_progress_sha256": sha256_file(feature_dir / "progress.json"),
        "query_manifest_sha256": sha256_file(feature_dir / "query-manifest.jsonl"),
        "audio_features_sha256": sha256_file(feature_dir / "features_audio.npy"),
        "video_features_sha256": sha256_file(feature_dir / "features_video.npy"),
        "candidate_checkpoint_sha256": checkpoint_hashes,
        "prediction_sha256": sha256_file(prediction_path),
        "rows": len(rows),
        "parent_recordings": len(set(parent_ids)),
        "rooms": len(set(room_ids)),
        "event_classes": len(set(events.tolist())),
        "target_counts": {str(label): int((targets == label).sum()) for label in (0, 1)},
        "source_or_stratum_metrics": {"room": per_room, "event_class_id": per_event_class},
        "overall_metrics": metrics,
        "paired_group_bootstrap_joint_vs_unimodal": paired_bootstrap,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "low_power_note": (
            "Parent-recording intervals use 78 clusters and room intervals use 7 clusters; "
            "both are low-power (<200). They do not establish broad AV fusion capability."
        ),
        "interventions": {
            "missing_audio_or_video": "replace standardized modality features with train-mean zero",
            "shuffle_audio_or_video": (
                "fixed seed-17 parent-recording derangement; keep query and other signal fixed"
            ),
        },
        "selection_used_devtest_metrics": False,
        "sealed_audit_accessed": False,
        "limitation": (
            "The readouts were fit for 30 epochs on all dev-train rows without validation; "
            "near-perfect train metrics indicate substantial overfit risk. This frozen readout "
            "is evaluated once on room-held-out dev-test."
        ),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--candidate-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_once(args.features, args.training, args.candidate_freeze, args.output_dir)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
