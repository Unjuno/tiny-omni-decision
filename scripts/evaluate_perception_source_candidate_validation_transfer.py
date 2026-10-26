"""Evaluate one fixed readout checkpoint on the sealed-to-iteration official-val cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from tiny_omni_decision.perception_source_candidate_readout import (
    CandidateFeatureExample,
    pool_audio_features,
    pool_video_candidate_features,
)

VIDEOPRISM_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
VIDEOPRISM_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
EFFICIENTAT_WEIGHTS_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
EFFICIENTAT_SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
READOUT_CHECKPOINT_SHA256 = "63390ba7532348bbd600f64ed796e92a3daf13b1f44ba0b18e83b1fd304780bf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def video_group_bootstrap_accuracy_interval(predictions, *, repetitions=10_000, seed=17):
    """Percentile bootstrap top-1 accuracy for one event per independent video."""
    if not predictions or repetitions < 1:
        raise ValueError("bootstrap requires predictions and a positive repetition count")
    video_ids = [row["video_id"] for row in predictions]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("video-group bootstrap requires one scored event per video")
    correctness = np.asarray([row["top1_correct"] for row in predictions], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(correctness), size=(repetitions, len(correctness)))
    means = correctness[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return {
        "method": "percentile bootstrap resampling video groups with replacement",
        "video_groups": len(video_ids),
        "repetitions": repetitions,
        "seed": seed,
        "confidence_level": 0.95,
        "top1_set_accuracy_interval": [float(lower), float(upper)],
    }


def paired_video_group_bootstrap_differences(
    predictions_by_condition, *, reference="joint", repetitions=10_000, seed=17
):
    """Bootstrap paired Accuracy/NLL/Brier differences over the same video groups."""
    if reference not in predictions_by_condition or repetitions < 1:
        raise ValueError("paired bootstrap requires a reference and positive repetitions")
    ref_rows = predictions_by_condition[reference]
    ref_ids = [row["record_id"] for row in ref_rows]
    ref_videos = [row["video_id"] for row in ref_rows]
    if not ref_rows or len(set(ref_videos)) != len(ref_videos):
        raise ValueError("paired bootstrap requires one prediction per unique video")

    def row_metrics(row):
        candidate_ids = row["candidate_ids"]
        targets = set(row["target_candidate_ids"])
        probabilities = [row["candidate_probabilities"][key] for key in candidate_ids]
        probabilities.append(row["none_visible_probability"])
        positives = [key in targets for key in candidate_ids] + [row["none_visible"]]
        positive_count = sum(positives)
        if positive_count < 1:
            raise ValueError(f"prediction has no positive target: {row['record_id']}")
        distribution = np.asarray(positives, dtype=np.float64) / positive_count
        brier = float(np.square(np.asarray(probabilities, dtype=np.float64) - distribution).sum())
        return np.asarray([row["top1_correct"], row["nll"], brier], dtype=np.float64)

    reference_values = np.stack([row_metrics(row) for row in ref_rows])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ref_rows), size=(repetitions, len(ref_rows)))
    results = {}
    for condition, rows in predictions_by_condition.items():
        if condition == reference:
            continue
        if [row["record_id"] for row in rows] != ref_ids or [
            row["video_id"] for row in rows
        ] != ref_videos:
            raise ValueError(f"paired predictions do not match reference IDs: {condition}")
        values = np.stack([row_metrics(row) for row in rows])
        differences = values - reference_values
        sampled_means = differences[indices].mean(axis=1)
        lower, upper = np.quantile(sampled_means, [0.025, 0.975], axis=0, method="linear")
        point = differences.mean(axis=0)
        results[condition] = {
            "reference": reference,
            "comparison": condition,
            "video_groups": len(ref_rows),
            "repetitions": repetitions,
            "seed": seed,
            "confidence_level": 0.95,
            "delta_comparison_minus_reference": {
                "top1_set_accuracy": float(point[0]),
                "mean_set_nll": float(point[1]),
                "mean_multiclass_brier": float(point[2]),
            },
            "intervals": {
                "top1_set_accuracy": [float(lower[0]), float(upper[0])],
                "mean_set_nll": [float(lower[1]), float(upper[1])],
                "mean_multiclass_brier": [float(lower[2]), float(upper[2])],
            },
        }
    return {
        "method": "paired percentile bootstrap resampling video groups with replacement",
        "reference": reference,
        "video_groups": len(ref_rows),
        "repetitions": repetitions,
        "seed": seed,
        "comparisons": results,
    }


def build_validation_examples(plan, labels, video_root: Path, audio_root: Path):
    planned = plan["validation"]["records"]
    plan_ids = {row["record_id"] for row in planned}
    if len(plan_ids) != len(planned) or any(
        "supervision" in row or "target_candidate_ids" in row for row in planned
    ):
        raise ValueError("validation feature plan is duplicated or contains target labels")
    labels_by_id = {row["record_id"]: row for row in labels}
    if len(labels_by_id) != len(labels) or set(labels_by_id) != plan_ids:
        raise ValueError("validation labels differ from the fixed label-free plan")
    if any(
        row.get("status") != "candidate"
        or row.get("split") != "validation"
        or row.get("provenance", {}).get("source_split") != "official_validation"
        for row in labels
    ):
        raise ValueError("validation labels are not from the pinned official validation source")

    video_index = read_jsonl(video_root / "feature_index.jsonl")
    video_values = np.load(video_root / "candidate_frame_roi_features.npy", mmap_mode="r")
    audio_index = read_jsonl(audio_root / "record_window_references.jsonl")
    audio_values = np.load(audio_root / "audio_window_features.npy", mmap_mode="r")
    visual = pool_video_candidate_features(planned, video_index, video_values)
    audio = pool_audio_features(audio_index, audio_values, variant="source")
    if set(audio) != plan_ids or set(visual) != plan_ids:
        raise ValueError("validation feature IDs differ from the fixed plan")

    examples = []
    for row in planned:
        record_id = row["record_id"]
        label = labels_by_id[record_id]
        candidate_ids = tuple(candidate["candidate_id"] for candidate in row["candidates"])
        if set(visual[record_id]) != set(candidate_ids):
            raise ValueError(f"candidate feature IDs differ from plan: {record_id}")
        targets = frozenset(label["supervision"]["target_candidate_ids"])
        none_visible = bool(label["supervision"]["none_visible"])
        if none_visible and targets:
            raise ValueError(f"none-visible example has positive candidate labels: {record_id}")
        if not none_visible and (not targets or not targets.issubset(candidate_ids)):
            raise ValueError(f"positive labels do not match candidates: {record_id}")
        examples.append(
            CandidateFeatureExample(
                record_id=record_id,
                video_id=row["video_id"],
                candidate_ids=candidate_ids,
                visual_mean_delta=np.stack([visual[record_id][key] for key in candidate_ids]),
                audio=audio[record_id],
                offset_audio=None,
                donor_record_id="",
                target_candidate_ids=targets,
                none_visible=none_visible,
            )
        )
    return examples


def validate_feature_artifact(root, plan, report_hash, split_name, modality):
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if modality == "video":
        feature_path = root / "candidate_frame_roi_features.npy"
        index_path = root / "feature_index.jsonl"
        valid = (
            result.get("status") == "PASS_SOURCE_CANDIDATE_VALIDATION_VIDEO_FEATURES"
            and result.get("checkpoint_sha256") == VIDEOPRISM_CHECKPOINT_SHA256
            and result.get("source_commit") == VIDEOPRISM_SOURCE_COMMIT
            and result.get("model_parameters") == 114_365_184
            and result.get("feature_sha256") == sha256_file(feature_path)
            and result.get("feature_index_sha256") == sha256_file(index_path)
            and result.get("feature_shape") == list(np.load(feature_path, mmap_mode="r").shape)
            and not result.get("labels_loaded")
            and not result.get("metrics_computed")
            and not result.get("gpu_used")
        )
    else:
        feature_path = root / "audio_window_features.npy"
        index_path = root / "record_window_references.jsonl"
        unique_path = root / "unique_window_index.jsonl"
        valid = (
            result.get("status") == "PASS_SOURCE_CANDIDATE_VALIDATION_AUDIO_FEATURES"
            and result.get("weights_sha256") == EFFICIENTAT_WEIGHTS_SHA256
            and result.get("source_commit") == EFFICIENTAT_SOURCE_COMMIT
            and result.get("model_parameters") == 1_961_239
            and result.get("feature_sha256") == sha256_file(feature_path)
            and result.get("feature_shape") == list(np.load(feature_path, mmap_mode="r").shape)
            and result.get("reference_index_sha256") == sha256_file(index_path)
            and result.get("unique_index_sha256") == sha256_file(unique_path)
            and result.get("model_state_sha256_before") == result.get("model_state_sha256_after")
            and bool(result.get("gpu_gate", {}).get("TINY_OMNI_GPU_GATE_RESERVED_MIB"))
            and not result.get("labels_loaded")
            and not result.get("metrics_computed")
        )
    if not valid or run.get("status") != result.get("status"):
        raise ValueError(f"{modality} feature artifact failed frozen provenance checks: {root}")
    expected_report_key = (
        "validation_media_report_sha256"
        if split_name == "validation"
        else f"{split_name}_media_report_sha256"
    )
    if (
        result.get("split") != split_name
        or result.get("plan_sha256") != sha256_file(plan["_path"])
        or result.get("media_report_sha256") != report_hash
        or result.get("acquisition_manifest_sha256", run.get("acquisition_manifest_sha256"))
        != plan["acquisition_manifest_sha256"]
        or plan.get(expected_report_key) != report_hash
    ):
        raise ValueError(f"{modality} feature artifact does not match the fixed validation plan")
    return {"run": run, "result": result, "feature_sha256": sha256_file(feature_path)}


def evaluate_controls(model, examples):
    from scripts.train_perception_source_candidate_readout import evaluate, metrics_from_predictions

    controls = {}
    conditions = {
        "joint": (examples, "source", "joint", None),
        "video_only_zero_audio": (examples, "zero", "joint", None),
        "audio_only_zero_video": (examples, "source", "zero", None),
        "reverse_video_temporal_delta": (examples, "source", "reverse_temporal", None),
    }
    donor_indices = np.roll(np.arange(len(examples)), 1)
    if len(examples) < 2 or np.any(donor_indices == np.arange(len(examples))):
        raise ValueError("fixed audio-swap control requires at least two distinct videos")
    donor_ids = {
        examples[index].record_id: examples[int(donor)].record_id
        for index, donor in enumerate(donor_indices)
    }
    donor_audio = {example.record_id: example.audio for example in examples}
    shuffled_examples = [
        replace(example, donor_record_id=donor_ids[example.record_id]) for example in examples
    ]
    conditions["mismatched_audio"] = (
        shuffled_examples,
        "shuffle",
        "joint",
        donor_audio,
    )
    for name, (rows, audio_mode, video_mode, donors) in conditions.items():
        result = evaluate(
            model,
            rows,
            audio_mode=audio_mode,
            video_mode=video_mode,
            donor_audio=donors,
        )
        predictions = result.pop("predictions")
        result["metrics"] = metrics_from_predictions(predictions)
        result["video_group_bootstrap"] = video_group_bootstrap_accuracy_interval(predictions)
        result["predictions"] = predictions
        result["audio_mode"] = audio_mode
        result["video_mode"] = video_mode
        if name == "mismatched_audio":
            result["audio_donor_assignment_sha256"] = hashlib.sha256(
                json.dumps(donor_ids, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        controls[name] = result
    return controls


def run(args):
    import torch

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {args.output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan["_path"] = args.plan.resolve()
    corpus_rows = read_jsonl(args.corpus)
    labels = [row for row in corpus_rows if row.get("status") == "candidate"]
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != READOUT_CHECKPOINT_SHA256:
        raise ValueError("fixed readout checkpoint SHA-256 mismatch")
    if plan.get("format") != "perception_test_source_candidate_validation_feature_plan_v1":
        raise ValueError("unexpected validation feature plan format")
    if plan.get("validation", {}).get("record_count") != len(labels) or len(labels) != 138:
        raise ValueError("validation corpus count differs from frozen 138-record cohort")
    report_hash = plan["validation_media_report_sha256"]
    video_artifact = validate_feature_artifact(
        args.video_features, plan, report_hash, "validation", "video"
    )
    audio_artifact = validate_feature_artifact(
        args.audio_features, plan, report_hash, "validation", "audio"
    )
    examples = build_validation_examples(plan, labels, args.video_features, args.audio_features)
    readout = json.loads((args.readout_provenance / "result.json").read_text(encoding="utf-8"))
    if readout.get("best_readout_state_sha256") != checkpoint_hash:
        raise ValueError("fixed readout checkpoint differs from selected training artifact")
    if readout.get("validation_is_blind") is not False or readout.get("sealed_audit_read"):
        raise ValueError(
            "readout provenance does not support the declared non-blind validation scope"
        )
    from scripts.train_perception_source_candidate_readout import CrossModalCandidateScorer

    model = CrossModalCandidateScorer().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("validation evaluation model unexpectedly has trainable parameters")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.controls:
        control_results = evaluate_controls(model, examples)
        prediction_hashes = {}
        for name, result in control_results.items():
            prediction_path = args.output / f"predictions-{name}.jsonl"
            with prediction_path.open("x", encoding="utf-8", newline="\n") as stream:
                for row in result.pop("predictions"):
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            prediction_hashes[name] = sha256_file(prediction_path)
        metrics = {name: result for name, result in control_results.items()}
        summary_status = "PASS_FIXED_CHECKPOINT_NONBLIND_OFFICIAL_VALIDATION_CONTROLS"
    else:
        from scripts.train_perception_source_candidate_readout import (
            evaluate,
            metrics_from_predictions,
        )

        evaluation = evaluate(model, examples, audio_mode="source", video_mode="joint")
        predictions = evaluation.pop("predictions")
        prediction_path = args.output / "predictions.jsonl"
        with prediction_path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in predictions:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        prediction_hashes = {"joint": sha256_file(prediction_path)}
        metrics = {
            "joint": {
                **metrics_from_predictions(predictions),
                "video_group_bootstrap": video_group_bootstrap_accuracy_interval(predictions),
            }
        }
        summary_status = "PASS_FIXED_CHECKPOINT_NONBLIND_OFFICIAL_VALIDATION_TRANSFER"
    summary = {
        "status": summary_status,
        "scope": "fixed-checkpoint diagnostic on non-blind validation; not a sealed audit",
        "checkpoint_sha256": checkpoint_hash,
        "corpus_sha256": sha256_file(args.corpus),
        "plan_sha256": sha256_file(args.plan),
        "media_report_sha256": report_hash,
        "video_feature_sha256": video_artifact["feature_sha256"],
        "audio_feature_sha256": audio_artifact["feature_sha256"],
        "candidate_count_distribution": dict(
            sorted(Counter(len(example.candidate_ids) for example in examples).items())
        ),
        "metrics_by_condition": metrics,
        "predictions_sha256_by_condition": prediction_hashes,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "calibration_performed": False,
        "sealed_audit_opened": False,
    }
    summary_path = args.output / "result.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--video-features", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--readout-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
