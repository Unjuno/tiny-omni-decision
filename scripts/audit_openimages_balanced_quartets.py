"""Explore disjoint four-target image quartets with identical option sets."""

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_openimages_train_option_prior import HASHES, hashed
from scripts.train_source_unimodal_cv import digest

PROTOCOL = Path("docs/pretrained_reuse") / (
    "R2_OPENIMAGES_BALANCED_QUARTET_FEASIBILITY_PROTOCOL_20261027.md"
)
REPO = Path(__file__).resolve().parents[1]


def build_quartets(candidates, saved, scope):
    eligible = [row for row in candidates if row["image_id"] in saved]
    if len(eligible) != len(saved) or {row["image_id"] for row in eligible} != saved:
        raise ValueError("candidate/media identity mismatch")
    possibilities = defaultdict(lambda: defaultdict(set))
    for row in eligible:
        positive, negative = set(row["positive_labels"]), set(row["negative_labels"])
        if positive & negative or not (positive | negative) <= scope:
            raise ValueError("invalid verified labels")
        for target in positive:
            for other in itertools.combinations(sorted(negative), 3):
                labels = tuple(sorted((target, *other)))
                possibilities[labels][target].add(row["image_id"])
    options = [key for key, value in possibilities.items() if set(value) == set(key)]
    options.sort(key=lambda key: hashed(f"quartet-set-v1|17|{'|'.join(key)}"))
    used, quartets = set(), []
    for key in options:
        key_string = "|".join(key)
        while True:
            chosen = []
            for label in key:
                available = possibilities[key][label] - used - set(chosen)
                if not available:
                    break
                chosen.append(min(available, key=lambda image: hashed(
                    f"quartet-image-v1|17|{key_string}|{label}|{image}")))
            if len(chosen) != 4:
                break
            ordered = sorted(key, key=lambda label: hashed(
                f"quartet-position-v1|17|{key_string}|{label}"))
            quartets.append({"options": ordered, "rows": [
                {"image_id": image, "target": label}
                for image, label in zip(chosen, key, strict=True)
            ]})
            used.update(chosen)
    return quartets, len(possibilities), len(options), len(eligible) - len(used)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    if any(digest(path) != expected for path, expected in HASHES.items()):
        raise ValueError("pinned source mismatch")
    candidates = json.loads(list(HASHES)[0].read_bytes())["candidates"]
    media = [json.loads(line) for line in list(HASHES)[1].read_text().splitlines()]
    saved = {row["image_id"] for row in media if row["status"] == "saved"}
    scope_rows = json.loads(list(HASHES)[2].read_bytes())["selected_target_classes"]
    names = {row["label_id"]: row["display_name"] for row in scope_rows}
    quartets, option_sets, complete_sets, unused = build_quartets(candidates, saved, set(names))
    target_counts = Counter(row["target"] for group in quartets for row in group["rows"])
    result = {
        "status": "TRAIN_CANDIDATE_FEASIBILITY_NOT_EVALUATION", "candidate_images": len(saved),
        "possible_option_sets": option_sets, "four_target_option_sets": complete_sets,
        "disjoint_quartets": len(quartets), "selected_images": 4 * len(quartets),
        "unused_images": unused, "target_counts": {names[k]: target_counts[k] for k in names},
        "option_only_expected_accuracy": 0.25 if quartets else None,
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
        "input_hashes": {str(path): value for path, value in HASHES.items()},
    }
    output.mkdir(parents=True)
    (output / "private-quartets.json").write_text(json.dumps(quartets, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
