"""Read-only parent-question diagnostic of saved binary video predictions."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write


def summarize(rows):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate predictions")
    parents = defaultdict(list)
    for r in rows:
        p = np.asarray(r["probabilities"], dtype=float)
        if (
            p.shape != (2,)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or abs(p.sum() - 1) > 1e-5
            or r["target_index"] not in (0, 1)
            or int(p.argmax()) != r["prediction_index"]
        ):
            raise ValueError("invalid probabilities or prediction")
        parents[r["parent_question_id"]].append(r)
    for group in parents.values():
        if len({(r["scene_index"], r["question_type"]) for r in group}) != 1:
            raise ValueError("parent identity spans scenes or types")

    def aggregate(groups):
        flat = [r for group in groups for r in group]
        return dict(
            choices=len(flat),
            parents=len(groups),
            scenes=len({r["scene_index"] for r in flat}),
            choice_accuracy=float(
                np.mean([r["target_index"] == r["prediction_index"] for r in flat])
            ),
            parent_exact_accuracy=float(
                np.mean(
                    [all(r["target_index"] == r["prediction_index"] for r in g) for g in groups]
                )
            ),
            always_wrong_parent_accuracy=float(
                np.mean([all(r["target_index"] == 0 for r in g) for g in groups])
            ),
            always_correct_parent_accuracy=float(
                np.mean([all(r["target_index"] == 1 for r in g) for g in groups])
            ),
        )

    result = {"all": aggregate(list(parents.values()))}
    for kind in sorted({r["question_type"] for r in rows}):
        result[kind] = aggregate([g for g in parents.values() if g[0]["question_type"] == kind])
    return result


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/shared-text-video-01")
    output = Path("docs/pretrained_reuse/R2_VIDEO_PARENT_DIAGNOSTIC_20261012.json")
    if output.exists():
        raise FileExistsError(output)
    results, hashes, reference = {}, {}, None
    for mode in ("minilm_attention", "minilm_no_video", "shared_attention", "shared_no_video"):
        for suffix in ("", "_shuffled"):
            path = root / mode / f"predictions{suffix}.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            identity = [
                (
                    r["id"],
                    r["scene_index"],
                    r["parent_question_id"],
                    r["question_type"],
                    r["target_index"],
                )
                for r in rows
            ]
            if reference is None:
                reference = identity
            elif identity != reference:
                raise ValueError("conditions do not share identical evaluation records")
            results[mode + suffix] = summarize(rows)
            hashes[str(path)] = sha(path)
    write(
        output,
        dict(
            status="DIAGNOSTIC_ONLY",
            results=results,
            hashes=hashes,
            limitation=("Development validation; exact match only over recorded choices; "
                        "no joint probability independence assumption."),
        ),
    )
    for mode, result in results.items():
        print(mode, result["all"])


if __name__ == "__main__":
    main()
