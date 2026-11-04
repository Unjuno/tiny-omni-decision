"""Inventory training-only CLEVRER grounding supervision; never open validation."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.range_generalization import sha, write


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    manifest = root / "clevrer-expanded-seed17-256train-200val/train.jsonl"
    raw = root / "clevrer-sample-seed17/train-questions.json"
    expected = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
    if sha(raw) != expected:
        raise ValueError("unrecognized training source")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if any(r["split"] != "train" for r in rows):
        raise ValueError("non-training record")
    scenes = {r["scene_index"] for r in rows}
    records = json.loads(raw.read_text())
    grouped = defaultdict(list)
    seen = set()
    for record in records:
        scene = record["scene_index"]
        if scene not in scenes:
            continue
        if scene in seen:
            raise ValueError("duplicate source scene")
        seen.add(scene)
        for q in record["questions"]:
            if q["question_type"] == "descriptive":
                grouped[tuple(q["program"])].append((scene, q["question_id"], q["answer"]))
    if seen != scenes:
        raise ValueError("missing cached scene")
    summaries = []
    for program, values in grouped.items():
        labels = Counter(str(v[2]) for v in values)
        summaries.append(
            dict(
                program=list(program),
                questions=len(values),
                scenes=len({v[0] for v in values}),
                labels=dict(labels),
                majority_training_fraction=max(labels.values()) / len(values),
            )
        )
    summaries.sort(key=lambda r: (-r["questions"], r["program"]))
    output = Path("docs/pretrained_reuse/R2_VIDEO_GROUNDING_COVERAGE_20261012.json")
    if output.exists():
        raise FileExistsError(output)
    write(
        output,
        dict(
            training_only=True,
            scenes=len(scenes),
            descriptive_questions=sum(r["questions"] for r in summaries),
            exact_programs=len(summaries),
            programs=summaries,
            hashes={str(p): sha(p) for p in (manifest, raw)},
        ),
    )
    print(
        json.dumps(
            dict(
                scenes=len(scenes),
                questions=sum(r["questions"] for r in summaries),
                programs=len(summaries),
                top=summaries[:6],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
