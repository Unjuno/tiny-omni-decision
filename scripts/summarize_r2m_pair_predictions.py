"""Derive two-candidate matching metrics from saved validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from train_r2m_gated_decision_pilot import paired_choice_summary


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--model-predictions", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("refusing to overwrite prediction analysis")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifact = json.loads(args.pair_index.read_text(encoding="utf-8"))
    payload = {key: value for key, value in artifact.items()
               if key != "manifest_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != artifact.get("manifest_sha256"):
        raise SystemExit("pair-index hash verification failed")
    relation = {
        str(row["pair_id"]): str(row["negative_participant_relation"])
        for row in artifact["pairs"] if row["split"] == "validation"
    }
    expected_pairs = set(relation)
    result: dict[str, object] = {}
    hashes = {}
    for name, path in (("gated", args.model_predictions),
                       ("temporal_correlation", args.baseline_predictions)):
        rows = read_jsonl(path)
        seen: dict[str, set[int]] = {}
        for row in rows:
            pair_id = str(row["pair_id"])
            if pair_id not in relation:
                raise SystemExit("prediction references a non-validation pair")
            seen.setdefault(pair_id, set()).add(int(row["target"]))
            row["negative_relation"] = relation[pair_id]
        if (set(seen) != expected_pairs
                or any(targets != {0, 1} for targets in seen.values())):
            raise SystemExit("validation prediction rows are incomplete or duplicated")
        metrics, pairs = paired_choice_summary(rows)
        result[name] = metrics
        prediction_path = args.output_dir / f"{name}-paired-choices.jsonl"
        prediction_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in pairs),
            encoding="utf-8",
        )
        hashes[name] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    summary = {
        "schema": "r2m_pair_prediction_analysis_v1",
        "pair_index_sha256": artifact["manifest_sha256"],
        "model_predictions_sha256": hashlib.sha256(
            args.model_predictions.read_bytes()
        ).hexdigest(),
        "baseline_predictions_sha256": hashlib.sha256(
            args.baseline_predictions.read_bytes()
        ).hexdigest(),
        "paired_choice_metrics": result,
        "paired_choice_predictions_sha256": hashes,
        "sealed_audit_accessed": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["paired_choice_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
