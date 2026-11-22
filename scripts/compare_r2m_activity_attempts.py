"""Paired validation comparison of two fixed R2-M activity-readout attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from analyze_r2m_activity_comparison import (
    basic_metrics,
    paired_bootstrap,
    read_jsonl,
    rows_to_arrays,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_run(path: Path):
    results = json.loads((path / "results.json").read_text(encoding="utf-8"))
    config_path = path / "effective-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (results.get("sealed_audit_loaded") is not False
            or config.get("sealed_audit_loaded") is not False):
        raise ValueError(f"sealed-audit isolation not proven for {path}")
    return results, config, config_path


def declared_weight_decay(config: dict[str, object]) -> float:
    if "weight_decay" in config:
        return float(config["weight_decay"])
    optimizer = str(config.get("optimizer", ""))
    match = re.search(r"weight_decay\s*=\s*([0-9.eE+-]+)", optimizer)
    if not match:
        raise ValueError("run config does not record a recoverable weight decay")
    return float(match.group(1))


def comparable_config(config: dict[str, object]) -> dict[str, object]:
    normalized = dict(config)
    normalized.pop("effective_config_sha256_excluding_self", None)
    normalized.pop("source_provenance", None)
    normalized["optimizer"] = "AdamW"
    normalized.pop("weight_decay", None)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.repetitions < 100:
        raise SystemExit("use at least 100 paired bootstrap repetitions")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite comparison: {args.output}")

    base_results, base_config, base_config_path = read_run(args.baseline_run_dir)
    new_results, new_config, new_config_path = read_run(args.candidate_run_dir)
    if base_config["choices"] != new_config["choices"]:
        raise SystemExit("choice vocabulary differs between runs")
    if comparable_config(base_config) != comparable_config(new_config):
        raise SystemExit("configs differ beyond optimizer weight decay/provenance")
    base_decay = declared_weight_decay(base_config)
    new_decay = declared_weight_decay(new_config)
    if base_decay == new_decay:
        raise SystemExit("the declared weight-decay factor did not change")

    vocabulary = tuple(str(label) for label in base_config["choices"])
    modes = tuple(str(arm["mode"]) for arm in base_results["arms"])
    if modes != tuple(str(arm["mode"]) for arm in new_results["arms"]):
        raise SystemExit("model arm order differs between runs")
    baseline_by_mode = {str(arm["mode"]): arm for arm in base_results["arms"]}
    candidate_by_mode = {str(arm["mode"]): arm for arm in new_results["arms"]}
    results = {}
    shared_ids_hash = None
    shared_ids: list[str] | None = None
    for mode in modes:
        filename = f"{mode}-validation-predictions.jsonl"
        baseline_rows = read_jsonl(args.baseline_run_dir / filename)
        candidate_rows = read_jsonl(args.candidate_run_dir / filename)
        if set(baseline_rows) != set(candidate_rows):
            raise SystemExit(f"validation IDs differ for {mode}")
        ids, targets, baseline_probs, participants = rows_to_arrays(
            baseline_rows, vocabulary
        )
        candidate_ids, candidate_targets, candidate_probs, candidate_participants = rows_to_arrays(
            candidate_rows, vocabulary
        )
        if (ids != candidate_ids or (targets != candidate_targets).any()
                or (participants != candidate_participants).any()):
            raise SystemExit(f"validation targets/groups differ for {mode}")
        current_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if shared_ids_hash is not None and current_hash != shared_ids_hash:
            raise SystemExit("arms do not share identical validation ID ordering")
        shared_ids_hash = current_hash
        shared_ids = ids
        bootstrap = paired_bootstrap(
            targets, baseline_probs, candidate_probs, participants,
            vocabulary=vocabulary, repetitions=args.repetitions,
        )
        for metric in bootstrap.values():
            metric["mean_delta_candidate_minus_baseline"] = metric.pop(
                "mean_delta_fused_minus_audio"
            )
        results[mode] = {
            "baseline_selected_epoch": baseline_by_mode[mode]["selected_epoch"],
            "candidate_selected_epoch": candidate_by_mode[mode]["selected_epoch"],
            "baseline_trainable_parameters": baseline_by_mode[mode]["trainable_parameter_count"],
            "candidate_trainable_parameters": candidate_by_mode[mode]["trainable_parameter_count"],
            "baseline_validation": basic_metrics(targets, baseline_probs, vocabulary),
            "candidate_validation": basic_metrics(targets, candidate_probs, vocabulary),
            "paired_candidate_minus_baseline_bootstrap": bootstrap,
        }

    baseline_audio = read_jsonl(
        args.baseline_run_dir / "audio_only-validation-predictions.jsonl"
    )
    report = {
        "schema": "r2m_activity_attempt_paired_comparison_v1",
        "baseline_config_sha256": sha256_file(base_config_path),
        "candidate_config_sha256": sha256_file(new_config_path),
        "baseline_results_sha256": sha256_file(args.baseline_run_dir / "results.json"),
        "candidate_results_sha256": sha256_file(args.candidate_run_dir / "results.json"),
        "baseline_weight_decay": base_decay,
        "candidate_weight_decay": new_decay,
        "paired_validation_ids_sha256": shared_ids_hash,
        "validation_examples": len(shared_ids or []),
        "held_out_participants": sorted(set(
            int(baseline_audio[key]["participant_id"])
            for key in (shared_ids or [])
        )),
        "bootstrap": {
            "method": "paired trial bootstrap stratified by participant × target activity",
            "repetitions": args.repetitions,
            "seed": 17,
            "interval": "percentile 95%",
            "limitations": (
                "Only two held-out participant clusters are present; intervals are conditional "
                "on these participants and do not quantify population-level subject uncertainty. "
                "Both checkpoints were selected using calibration; validation is development "
                "data, not a blind audit."
            ),
        },
        "arms": results,
        "sealed_audit_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "weight_decay": [base_decay, new_decay],
        "validation_examples": report["validation_examples"],
        "arms": list(results),
        "sealed_audit_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
