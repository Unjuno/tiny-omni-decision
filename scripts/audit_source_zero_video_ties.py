"""Measure numerical ties in existing zero-video source-candidate predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SOURCE = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-source-candidate-validation-transfer-20261026-attempt-07/diagnostic-controls/predictions-audio_only_zero_video.jsonl"
)
SHA = "646dbbafd157c9b8bee66199d523677c77e6dc8e32c81c830a8ac97e92775dfd"


def summarize(rows):
    spreads, none_selected, choices = [], 0, Counter()
    if not rows:
        raise ValueError("empty predictions")
    for row in rows:
        ids = row["candidate_ids"]
        probabilities = row["candidate_probabilities"]
        if len(ids) != len(set(ids)) or set(ids) != set(probabilities):
            raise ValueError("candidate IDs and probabilities disagree")
        values = np.asarray([probabilities[key] for key in ids] + [row["none_visible_probability"]])
        if len(values) < 2 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("invalid probabilities")
        if not np.isclose(values.sum(), 1, atol=1e-6, rtol=0):
            raise ValueError("probabilities do not sum to one")
        spreads.append(float(np.max(values) - np.min(values)))
        none_selected += int(row["predicted"] == "none_visible")
        choices[len(values)] += 1
    return {
        "rows": len(rows),
        "max_probability_spread": max(spreads),
        "rows_with_spread_at_most_1e-6": sum(value <= 1e-6 for value in spreads),
        "none_visible_predictions": none_selected,
        "choice_count_distribution": dict(choices),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA:
        raise ValueError("prediction hash mismatch")
    rows = [json.loads(line) for line in data.splitlines()]
    result = {
        "source_sha256": SHA,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summarize(rows),
        "model_run": False,
        "training": False,
        "sealed_audit_opened": False,
        "scope": "post-hoc saved-prediction numerical diagnostic",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
