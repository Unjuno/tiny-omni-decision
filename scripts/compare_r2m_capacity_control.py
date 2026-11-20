"""Compare a sensor-present fusion arm with its capacity-matched sensor-absent run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.analyze_r2m_activity_comparison import (
    basic_metrics,
    paired_bootstrap,
    read_jsonl,
    rows_to_arrays,
)

IGNORED_CONFIG_FIELDS = {
    "effective_config_sha256_excluding_self",
    "model_arms",
    "modality_condition",
    "source_commit",
    "source_provenance",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_run(run_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "effective-config.json").read_text(encoding="utf-8"))
    if results.get("sealed_audit_loaded") is not False:
        raise ValueError(f"sealed audit isolation is not established for {run_dir}")
    if config.get("sealed_audit_loaded") is not False:
        raise ValueError(f"sealed audit isolation is not established for {run_dir}")
    return results, config


def validate_run_pair(
    fusion_results: dict[str, object],
    fusion_config: dict[str, object],
    control_results: dict[str, object],
    control_config: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    fusion_arms = {str(row["mode"]): row for row in fusion_results["arms"]}
    control_arms = {str(row["mode"]): row for row in control_results["arms"]}
    if "audio_sensor_gated" not in fusion_arms or "audio_sensor_gated" not in control_arms:
        raise ValueError("both runs must contain the audio_sensor_gated arm")
    if len(control_arms) != 1:
        raise ValueError("capacity-control run must contain exactly one arm")
    fusion_arm = fusion_arms["audio_sensor_gated"]
    control_arm = control_arms["audio_sensor_gated"]
    if fusion_arm.get("modality_condition", "normal") != "normal":
        raise ValueError("reference fusion arm must use paired audio and sensor inputs")
    if control_arm.get("modality_condition") != "without_sensor":
        raise ValueError("capacity-control arm must have sensor absent during training")
    if fusion_arm["parameter_count"] != control_arm["parameter_count"]:
        raise ValueError("total model parameter counts differ")

    comparable_fusion = {
        key: value for key, value in fusion_config.items()
        if key not in IGNORED_CONFIG_FIELDS
    }
    comparable_control = {
        key: value for key, value in control_config.items()
        if key not in IGNORED_CONFIG_FIELDS
    }
    if comparable_fusion != comparable_control:
        differences = sorted(
            key for key in set(comparable_fusion) | set(comparable_control)
            if comparable_fusion.get(key) != comparable_control.get(key)
        )
        raise ValueError(f"run configurations differ outside the declared condition: {differences}")
    if fusion_config["choices"] != control_config["choices"]:
        raise ValueError("choice vocabularies differ")
    return fusion_arm, control_arm


def compare_predictions(
    baseline_path: Path,
    candidate_path: Path,
    vocabulary: tuple[str, ...],
    *,
    repetitions: int,
) -> dict[str, object]:
    baseline = read_jsonl(baseline_path)
    candidate = read_jsonl(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("paired validation sample IDs differ")
    ids, targets, baseline_probs, participants = rows_to_arrays(baseline, vocabulary)
    candidate_ids, candidate_targets, candidate_probs, candidate_participants = rows_to_arrays(
        candidate, vocabulary
    )
    if (ids != candidate_ids or (targets != candidate_targets).any()
            or (participants != candidate_participants).any()):
        raise ValueError("paired validation targets or participant IDs differ")
    bootstrap = paired_bootstrap(
        targets,
        baseline_probs,
        candidate_probs,
        participants,
        vocabulary=vocabulary,
        repetitions=repetitions,
        seed=17,
    )
    participant_metrics = {}
    for participant in sorted(set(int(value) for value in participants)):
        mask = participants == participant
        participant_metrics[str(participant)] = {
            "examples": int(mask.sum()),
            "baseline": basic_metrics(targets[mask], baseline_probs[mask], vocabulary),
            "candidate": basic_metrics(targets[mask], candidate_probs[mask], vocabulary),
        }
    return {
        "validation_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "examples": len(ids),
        "participants": sorted(set(int(value) for value in participants)),
        "baseline_metrics": basic_metrics(targets, baseline_probs, vocabulary),
        "candidate_metrics": basic_metrics(targets, candidate_probs, vocabulary),
        "per_participant": participant_metrics,
        "paired_bootstrap_candidate_minus_baseline": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-run-dir", type=Path, required=True)
    parser.add_argument("--capacity-control-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.repetitions < 100:
        raise SystemExit("use at least 100 paired bootstrap repetitions")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite comparison: {args.output}")

    fusion_results, fusion_config = read_run(args.fusion_run_dir)
    control_results, control_config = read_run(args.capacity_control_run_dir)
    fusion_arm, control_arm = validate_run_pair(
        fusion_results, fusion_config, control_results, control_config
    )
    vocabulary = tuple(str(value) for value in fusion_config["choices"])
    fusion_path = args.fusion_run_dir / "audio_sensor_gated-validation-predictions.jsonl"
    control_path = (
        args.capacity_control_run_dir / "audio_sensor_gated-validation-predictions.jsonl"
    )
    fusion_vs_control = compare_predictions(
        control_path, fusion_path, vocabulary, repetitions=args.repetitions
    )

    audio_path = args.fusion_run_dir / "audio_only-validation-predictions.jsonl"
    fusion_vs_audio = compare_predictions(
        audio_path, fusion_path, vocabulary, repetitions=args.repetitions
    )
    report = {
        "schema": "r2m_capacity_matched_sensor_control_comparison_v1",
        "fusion_run": {
            "path": str(args.fusion_run_dir),
            "config_sha256": sha256_file(args.fusion_run_dir / "effective-config.json"),
            "results_sha256": sha256_file(args.fusion_run_dir / "results.json"),
            "checkpoint_sha256": fusion_arm["best_checkpoint_sha256"],
            "selected_epoch": fusion_arm["selected_epoch"],
            "modality_condition": "paired audio + sensor",
        },
        "capacity_control_run": {
            "path": str(args.capacity_control_run_dir),
            "config_sha256": sha256_file(args.capacity_control_run_dir / "effective-config.json"),
            "results_sha256": sha256_file(args.capacity_control_run_dir / "results.json"),
            "checkpoint_sha256": control_arm["best_checkpoint_sha256"],
            "selected_epoch": control_arm["selected_epoch"],
            "modality_condition": "audio only; sensor masked during training and evaluation",
        },
        "equal_total_parameter_count": fusion_arm["parameter_count"],
        "same_effective_training_config_except_declared_sensor_condition": True,
        "fusion_minus_capacity_matched_audio": fusion_vs_control,
        "fusion_minus_legacy_audio_only": fusion_vs_audio,
        "bootstrap": {
            "method": "paired trial bootstrap stratified by participant × target activity",
            "repetitions": args.repetitions,
            "seed": 17,
            "interval": "percentile 95%",
            "limitations": (
                "Validation contains two previously observed participant clusters only. "
                "Intervals are conditional on these participants, not population-level "
                "subject uncertainty. Both checkpoints were selected on calibration. "
                "Validation is development data, not blind audit evidence."
            ),
        },
        "sealed_audit_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({
        "output": str(args.output),
        "validation_examples": fusion_vs_control["examples"],
        "participants": fusion_vs_control["participants"],
        "parameter_count": fusion_arm["parameter_count"],
        "fusion_macro_accuracy": fusion_vs_control["candidate_metrics"][
            "macro_accuracy_present_classes"
        ],
        "capacity_control_macro_accuracy": fusion_vs_control["baseline_metrics"][
            "macro_accuracy_present_classes"
        ],
        "sealed_audit_loaded": False,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
