"""Cross-validated option-only prior for fixed train image candidates."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.decision_math import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
CANDIDATES = ROOT / "openimages-in-scope-train-candidates-20261027-attempt-01/candidates.json"
MEDIA = ROOT / "openimages-in-scope-train-media-20261027-attempt-01/private-media-manifest.jsonl"
SCOPE = Path("manifests/pretrained_reuse/openimages-v6-train-target-support-20261026.json")
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_TRAIN_OPTION_PRIOR_PROTOCOL_20261027.md")
HASHES = {
    CANDIDATES: "67a9b9a468b868c82ec43cc73b33fe477fb8b97fb079f1035a2c67e051a07d5a",
    MEDIA: "7ff4b8bda635812ed404adf06b279e55a96838f18407e39ab80e9e659f59a9a7",
    SCOPE: "a3a51bc54a1f14b3f584a7a9b44ba1a220cc7783b9480809ddfa3325f162cab3",
}
REPO = Path(__file__).resolve().parents[1]


def hashed(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_tasks(candidates, saved_ids, scope):
    rows = [row for row in candidates if row["image_id"] in saved_ids]
    if len({row["image_id"] for row in rows}) != len(rows) or len(rows) != len(saved_ids):
        raise ValueError("candidate/media image identity mismatch")
    counts = Counter()
    tasks = []
    for row in sorted(rows, key=lambda x: hashed(f"option-v1|17|{x['image_id']}")):
        image_id, target = row["image_id"], row["target_label"]
        positive, negative = set(row["positive_labels"]), set(row["negative_labels"])
        if target not in positive or positive & negative or not (positive | negative) <= scope:
            raise ValueError("invalid in-scope positive or negative labels")
        if len(negative) < 3:
            raise ValueError("insufficient explicit negatives")
        chosen = sorted(
            negative, key=lambda x: (counts[x], hashed(f"negative-v1|17|{image_id}|{x}"))
        )[:3]
        counts.update(chosen)
        options = sorted([target, *chosen], key=lambda x: hashed(f"position-v1|17|{image_id}|{x}"))
        tasks.append({"image_id": image_id, "target": target, "options": options,
                      "fold": int(hashed(f"fold-v1|17|{image_id}"), 16) % 5})
    return tasks


def cross_validate(tasks):
    predictions = []
    for fold in range(5):
        train = [row for row in tasks if row["fold"] != fold]
        positive, negative = Counter(), Counter()
        for row in train:
            positive[row["target"]] += 1
            negative.update(x for x in row["options"] if x != row["target"])
        for row in tasks:
            if row["fold"] != fold:
                continue
            odds = [(positive[x] + 1) / (negative[x] + 1) for x in row["options"]]
            total = sum(odds)
            predictions.append(row | {"probabilities": [score / total for score in odds]})
    return predictions


def metrics(rows, uniform=False):
    probabilities = [[0.25] * 4 if uniform else row["probabilities"] for row in rows]
    targets = [row["options"].index(row["target"]) for row in rows]
    correct = sum(max(range(4), key=probs.__getitem__) == target
                  for probs, target in zip(probabilities, targets, strict=True))
    return {"count": len(rows), "accuracy": correct / len(rows),
            "nll": negative_log_likelihood(probabilities, targets),
            "brier": brier_score(probabilities, targets),
            "ece": expected_calibration_error(probabilities, targets, n_bins=15)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    if any(digest(path) != expected for path, expected in HASHES.items()):
        raise ValueError("pinned source mismatch")
    candidates = json.loads(CANDIDATES.read_bytes())["candidates"]
    media = [json.loads(line) for line in MEDIA.read_text(encoding="utf-8").splitlines()]
    saved_ids = {row["image_id"] for row in media if row["status"] == "saved"}
    names = {item["label_id"]: item["display_name"] for item in
             json.loads(SCOPE.read_bytes())["selected_target_classes"]}
    tasks = build_tasks(candidates, saved_ids, set(names))
    predictions = cross_validate(tasks)
    pos = Counter(row["target"] for row in tasks)
    neg = Counter(x for row in tasks for x in row["options"] if x != row["target"])
    result = {
        "status": "TRAIN_CANDIDATE_DIAGNOSTIC_ONLY", "rows": len(tasks),
        "fold_sizes": dict(Counter(str(row["fold"]) for row in tasks)),
        "class_roles": {names[key]: {"positive": pos[key], "negative": neg[key]} for key in names},
        "option_only": metrics(predictions), "uniform": metrics(predictions, uniform=True),
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
        "input_hashes": {str(path): value for path, value in HASHES.items()},
    }
    output.mkdir(parents=True)
    (output / "private-predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
