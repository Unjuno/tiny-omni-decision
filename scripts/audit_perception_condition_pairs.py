"""Train-only feasibility of same-observation, same-options condition pairs."""

import argparse
import collections
import itertools
import json
import zipfile
from pathlib import Path

from scripts.audit_perception_train import sha


def summarize(data):
    counts = collections.Counter()
    reasoning = collections.Counter()
    examples = []
    templates = collections.Counter()
    pair_videos = set()
    for identity, row in data.items():
        if row["metadata"]["split"] != "train":
            raise ValueError("train only")
        groups = collections.defaultdict(list)
        for q in row["mc_question"]:
            options = [" ".join(x.lower().split()) for x in q["options"]]
            if len(options) != 3 or len(set(options)) != 3 or q["answer_id"] not in range(3):
                raise ValueError("invalid options/target")
            counts["questions"] += 1
            reasoning[q["reasoning"]] += 1
            groups[tuple(sorted(options))].append((q, options[q["answer_id"]]))
        for group in groups.values():
            for (a, target_a), (b, target_b) in itertools.combinations(group, 2):
                counts["same_video_same_options_pairs"] += 1
                if a["question"].strip().lower() == b["question"].strip().lower():
                    counts["same_question_pairs"] += 1
                    continue
                counts["different_question_pairs"] += 1
                templates[(a["question"], b["question"])] += 1
                pair_videos.add(identity)
                if target_a != target_b:
                    counts["different_question_different_answer_pairs"] += 1
                    if len(examples) < 12:
                        examples.append(dict(video_id=identity, first=a, second=b))
    counts["videos_with_pairs"] = len(pair_videos)
    return dict(
        counts=dict(counts),
        reasoning=dict(reasoning),
        examples=examples,
        question_templates=[
            dict(first=a, second=b, pairs=count) for (a, b), count in templates.most_common()
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    path = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01/train_annotations.zip"
    )
    digest = sha(path)
    if digest != "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78":
        raise ValueError("source changed")
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("all_train.json"))
    result = summarize(data)
    result.update(
        source_sha256=digest,
        script_sha256=sha(Path(__file__)),
        validation_opened=False,
        model_evaluation=False,
        limitation="lexical option equality is not semantic condition-pair validation",
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
