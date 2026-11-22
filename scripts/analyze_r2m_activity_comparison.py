"""Paired diagnostics for audio-only versus gated audio+vibration outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> dict[str, dict[str, object]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        identifier = str(row["sample_id"])
        if identifier in rows:
            raise ValueError(f"duplicate prediction ID: {identifier}")
        rows[identifier] = row
    return rows


def rows_to_arrays(rows, vocabulary):
    ordered = [rows[key] for key in sorted(rows)]
    ids = [str(row["sample_id"]) for row in ordered]
    targets = np.asarray([vocabulary.index(str(row["target"])) for row in ordered])
    probabilities = np.asarray([
        [float(row["probabilities"][label]) for label in vocabulary]
        for row in ordered
    ], dtype=np.float64)
    participants = np.asarray([int(row["participant_id"]) for row in ordered])
    return ids, targets, probabilities, participants


def basic_metrics(targets, probabilities, vocabulary):
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    chosen = np.maximum(probabilities[np.arange(len(targets)), targets], 1e-15)
    one_hot = np.eye(len(vocabulary))[targets]
    confidence = probabilities.max(axis=1)
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for index, lower in enumerate(edges[:-1]):
        upper = edges[index + 1]
        upper_ok = confidence <= upper if index == 14 else confidence < upper
        mask = (confidence >= lower) & upper_ok
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    by_class = {}
    for class_index, label in enumerate(vocabulary):
        mask = targets == class_index
        if mask.any():
            by_class[label] = {
                "examples": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
                "nll": float(-np.log(chosen[mask]).mean()),
                "brier": float(((probabilities[mask] - one_hot[mask]) ** 2)
                                .sum(axis=1).mean()),
            }
    return {
        "examples": int(len(targets)),
        "accuracy": float(correct.mean()),
        "macro_accuracy_present_classes": float(np.mean([
            row["accuracy"] for row in by_class.values()
        ])),
        "macro_nll_present_classes": float(np.mean([
            row["nll"] for row in by_class.values()
        ])),
        "macro_brier_present_classes": float(np.mean([
            row["brier"] for row in by_class.values()
        ])),
        "nll": float(-np.log(chosen).mean()),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_15bin": ece,
        "mean_confidence": float(confidence.mean()),
        "per_activity": by_class,
    }


def paired_bootstrap(
    targets, audio_probabilities, fused_probabilities, participants, *,
    vocabulary, repetitions=10000, seed=17, batch_repetitions=250,
):
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (participant, target) in enumerate(zip(participants, targets, strict=True)):
        strata[(int(participant), int(target))].append(index)
    if len({participant for participant, _ in strata}) < 2:
        raise ValueError("paired validation must include at least two held-out participants")
    rng = np.random.default_rng(seed)
    deltas = {name: [] for name in ("accuracy", "macro_accuracy", "macro_nll", "macro_brier")}
    for start in range(0, repetitions, batch_repetitions):
        count = min(batch_repetitions, repetitions - start)
        sampled_parts = [
            rng.choice(indices, size=(count, len(indices)), replace=True)
            for indices in strata.values()
        ]
        sampled = np.concatenate(sampled_parts, axis=1)
        sampled_targets = targets[sampled]
        class_masks = [sampled_targets == index for index in range(len(vocabulary))]
        audio_pred = audio_probabilities[sampled].argmax(axis=2)
        fused_pred = fused_probabilities[sampled].argmax(axis=2)
        audio_correct = audio_pred == sampled_targets
        fused_correct = fused_pred == sampled_targets
        audio_chosen = np.take_along_axis(
            audio_probabilities[sampled], sampled_targets[..., None], axis=2
        )[..., 0]
        fused_chosen = np.take_along_axis(
            fused_probabilities[sampled], sampled_targets[..., None], axis=2
        )[..., 0]
        one_hot = np.eye(len(vocabulary))[sampled_targets]
        audio_brier = ((audio_probabilities[sampled] - one_hot) ** 2).sum(axis=2)
        fused_brier = ((fused_probabilities[sampled] - one_hot) ** 2).sum(axis=2)
        deltas["accuracy"].extend((fused_correct.mean(axis=1)
                                   - audio_correct.mean(axis=1)).tolist())
        class_accuracy_delta = []
        class_nll_delta = []
        class_brier_delta = []
        for class_mask in class_masks:
            class_accuracy_delta.append(
                (fused_correct * class_mask).sum(axis=1) / class_mask.sum(axis=1)
                - (audio_correct * class_mask).sum(axis=1) / class_mask.sum(axis=1)
            )
            class_nll_delta.append(
                ((-np.log(np.maximum(fused_chosen, 1e-15)) * class_mask)
                 .sum(axis=1) / class_mask.sum(axis=1))
                - ((-np.log(np.maximum(audio_chosen, 1e-15)) * class_mask)
                   .sum(axis=1) / class_mask.sum(axis=1))
            )
            class_brier_delta.append(
                (fused_brier * class_mask).sum(axis=1) / class_mask.sum(axis=1)
                - (audio_brier * class_mask).sum(axis=1) / class_mask.sum(axis=1)
            )
        deltas["macro_accuracy"].extend(np.mean(class_accuracy_delta, axis=0).tolist())
        deltas["macro_nll"].extend(np.mean(class_nll_delta, axis=0).tolist())
        deltas["macro_brier"].extend(np.mean(class_brier_delta, axis=0).tolist())
    return {
        name: {
            "mean_delta_fused_minus_audio": float(np.mean(values)),
            "percentile_95_interval": [
                float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
            ],
        }
        for name, values in deltas.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.repetitions < 100:
        raise SystemExit("use at least 100 paired bootstrap repetitions")
    results = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    vocabulary = tuple(json.loads(
        (args.run_dir / "effective-config.json").read_text(encoding="utf-8")
    )["choices"])
    audio = read_jsonl(args.run_dir / "audio_only-validation-predictions.jsonl")
    fused = read_jsonl(args.run_dir / "audio_sensor_gated-validation-predictions.jsonl")
    if set(audio) != set(fused):
        raise SystemExit("audio and fusion prediction sample IDs differ")
    ids, targets, audio_probabilities, participants = rows_to_arrays(audio, vocabulary)
    fused_ids, fused_targets, fused_probabilities, fused_participants = rows_to_arrays(
        fused, vocabulary
    )
    if (ids != fused_ids or not np.array_equal(targets, fused_targets)
            or not np.array_equal(participants, fused_participants)):
        raise SystemExit("paired predictions differ in order, target, or group identity")

    by_participant = {}
    for participant in sorted(set(participants.tolist())):
        mask = participants == participant
        by_participant[str(participant)] = {
            "examples": int(mask.sum()),
            "activities_present": sorted({vocabulary[int(x)] for x in targets[mask]}),
            "audio_only": basic_metrics(targets[mask], audio_probabilities[mask], vocabulary),
            "audio_sensor_gated": basic_metrics(
                targets[mask], fused_probabilities[mask], vocabulary
            ),
        }
    report = {
        "schema": "r2m_activity_paired_comparison_v1",
        "validation_ids_sha256": __import__("hashlib").sha256(
            "\n".join(ids).encode()
        ).hexdigest(),
        "validation_examples": len(ids),
        "held_out_participants": sorted(set(participants.tolist())),
        "macro_bootstrap": {
            "method": (
                "paired bootstrap of unique validation trials, stratified by "
                "participant and target activity"
            ),
            "repetitions": args.repetitions,
            "seed": 17,
            "interval": "percentile 95%",
            "deltas_fused_minus_audio": paired_bootstrap(
                targets, audio_probabilities, fused_probabilities, participants,
                vocabulary=vocabulary, repetitions=args.repetitions,
            ),
            "limitations": (
                "validation has only two participant clusters; intervals resample trials "
                "within these participants and do not estimate population-level participant "
                "uncertainty. Checkpoint selection used calibration, but validation remains "
                "development data and is not blind audit evidence."
            ),
        },
        "audio_only_validation": results["arms"][0]["selected_validation_metrics"],
        "audio_sensor_gated_validation": results["arms"][2]["selected_validation_metrics"],
        "per_participant": by_participant,
        "sealed_audit_loaded": False,
    }
    path = args.run_dir / "paired-audio-vs-fusion-analysis.json"
    if path.exists():
        raise SystemExit("refusing to overwrite an existing paired comparison")
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        "report": str(path),
        "paired_examples": len(ids),
        "participants": report["held_out_participants"],
        "bootstrap": report["macro_bootstrap"]["deltas_fused_minus_audio"],
        "sealed_audit_loaded": False,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
