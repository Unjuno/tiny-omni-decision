"""Audit train-target whitelist leakage in two saved Open Images diagnostics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.train_source_unimodal_cv import digest

TARGETS = Path("manifests/pretrained_reuse/openimages-v6-train-target-support-20261026.json")
TARGET_SHA = "a3a51bc54a1f14b3f584a7a9b44ba1a220cc7783b9480809ddfa3325f162cab3"
PILOT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/openimages-videoprism-readout-20261026-attempt-01/validation-predictions.jsonl"
)
PILOT_SHA = "42cbfd4b5e0d68be87f2f2657e13e14c78166906d06262896f68149d255b1400"
FRESH = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/openimages-freshval-20261026-attempt-01/tinyclip-zero-shot/predictions.jsonl"
)
FRESH_SHA = "6daf0a2c89f1a8abdef5df9dfc254f7f49b0ae1117f759627dbf0704c2c88908"


def audit(rows, whitelist):
    if len(whitelist) != 11 or not rows:
        raise ValueError("unexpected target scope or empty records")
    counts = Counter()
    prior, model, expected = [], [], []
    seen = set()
    for row in rows:
        options = row["options"]
        if row["sample_id"] in seen or len(options) != 4 or len(set(options)) != 4:
            raise ValueError("duplicate sample or invalid options")
        seen.add(row["sample_id"])
        available = [o for o in options if o in whitelist]
        if not available or row["target"] not in available or row["prediction"] not in options:
            raise ValueError("target-scope or prediction mismatch")
        counts[len(available)] += 1
        expected.append(1 / len(available))
        prior.append(int(available[0] == row["target"]))
        model.append(int(row["prediction"] == row["target"]))
    prior_array, model_array = np.asarray(prior), np.asarray(model)
    changes = model_array - prior_array
    draws = np.random.default_rng(17).integers(0, len(rows), size=(10_000, len(rows)))
    return {
        "n": len(rows),
        "target_whitelist_option_count": dict(sorted(counts.items())),
        "unique_whitelist_option_fraction": counts[1] / len(rows),
        "random_uniform_within_whitelist_expected_accuracy": float(np.mean(expected)),
        "first_whitelist_option_accuracy": float(prior_array.mean()),
        "saved_model_accuracy": float(model_array.mean()),
        "model_minus_prior_accuracy": float(changes.mean()),
        "paired_image_bootstrap_95": np.quantile(
            changes[draws].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "model_correct_prior_wrong": int(((model_array == 1) & (prior_array == 0)).sum()),
        "prior_correct_model_wrong": int(((model_array == 0) & (prior_array == 1)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path, expected in ((TARGETS, TARGET_SHA), (PILOT, PILOT_SHA), (FRESH, FRESH_SHA)):
        if digest(path) != expected:
            raise ValueError(f"pinned input mismatch: {path}")
    source = json.loads(TARGETS.read_bytes())
    whitelist = {x["display_name"] for x in source["selected_target_classes"]}
    results = {}
    for name, path in (("pilot", PILOT), ("fresh_task_ids", FRESH)):
        rows = list(map(json.loads, path.read_bytes().splitlines()))
        if len(rows) != 168:
            raise ValueError("unexpected example count")
        results[name] = audit(rows, whitelist)
    result = {
        "status": "PASS_POSTHOC_WHITELIST_SHORTCUT_AUDIT",
        "targets_sha256": TARGET_SHA,
        "pilot_predictions_sha256": PILOT_SHA,
        "fresh_predictions_sha256": FRESH_SHA,
        "script_sha256": digest(Path(__file__)),
        "bootstrap_seed": 17,
        "bootstrap_repetitions": 10000,
        "results": results,
        "scope": "posthoc on previously exposed development sets; no media, model or audit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
