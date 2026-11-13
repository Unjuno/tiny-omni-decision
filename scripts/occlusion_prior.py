"""Reproduce the fixed train-frequency, no-media development baseline."""

import argparse
import collections
import json
from pathlib import Path

from scripts.audit_correspondence_validation import sha, write
from tiny_omni_decision.decision_math import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)


def evaluate(train, valid):
    counts = collections.defaultdict(collections.Counter)
    for row in train:
        for question in row["questions"]:
            counts[(tuple(row["options"]), question["condition"])][question["target"]] += 1
    result, correct = {}, {}
    for condition in ("initial", "final"):
        probabilities, targets, unseen = [], [], 0
        for row in valid:
            count = counts[(tuple(row["options"]), condition)]
            unseen += not bool(count)
            probabilities.append([(count[i] + 1) / (sum(count.values()) + 3) for i in range(3)])
            targets.append(
                next(q["target"] for q in row["questions"] if q["condition"] == condition)
            )
        correct[condition] = [
            max(range(3), key=p.__getitem__) == t
            for p, t in zip(probabilities, targets, strict=True)
        ]
        result[condition] = dict(
            accuracy=sum(correct[condition]) / len(targets),
            nll=negative_log_likelihood(probabilities, targets),
            brier=brier_score(probabilities, targets),
            ece=expected_calibration_error(probabilities, targets),
            unseen_option_sets=unseen,
        )
    result["both_correct"] = sum(
        a and b for a, b in zip(correct["initial"], correct["final"], strict=True)
    ) / len(valid)
    result["method"] = (
        "train-only per exact option-set and condition frequency; Laplace +1; "
        "unseen uniform; lowest-index ties"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-pairs-01")
    hashes = dict(
        train="8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7",
        valid="30846aaa87e2292e97ee3187c5f2f986ba3c3d57d46a249b361e5a14c0cc8ec2",
    )
    data = {}
    for split, digest in hashes.items():
        path = root / f"{split}.json"
        if sha(path) != digest:
            raise ValueError("input changed")
        data[split] = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    result = evaluate(data["train"], data["valid"])
    result["source_hashes"] = hashes
    reference = json.loads((root / "train-prior-baseline.json").read_text(encoding="utf-8"))
    if result != reference:
        raise ValueError("baseline reproduction mismatch")
    write(
        args.output,
        dict(status="PASS_EXACT_REPRODUCTION", result=result, script_sha256=sha(Path(__file__))),
    )


if __name__ == "__main__":
    main()
