"""No media/model: leave-video-out query-only baseline on previously read train."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from scripts.train_sensor_subjects import metrics
from tiny_omni_decision.query_prior import leave_group_out_prior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    source = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/perception-visibility-proposal-01/records.jsonl"
    )
    if sha(source) != "f2d19f6c148bd44c14d1f9b6949bf129f0e3fd2df300dedac0257a55cef7ac4a":
        raise ValueError("proposal changed")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    report = dict(
        source_sha256=sha(source),
        smoothing="Laplace alpha=1 fixed",
        scope="Train-only leave-video-out diagnostic; not participant-disjoint or blind",
        media_loaded=False,
        validation_loaded=False,
        results={},
    )
    for name, subset in (
        ("all", rows),
        ("candidate", [r for r in rows if r["status"] == "candidate"]),
    ):
        classes = [r["request"]["sound_class"] for r in subset]
        groups = [r["group_id"] for r in subset]
        targets = [int(r["supervision"]["target_option_id"] == "visible") for r in subset]
        probabilities = np.array(leave_group_out_prior(classes, groups, targets))
        y = np.array(targets)
        counts = collections.defaultdict(collections.Counter)
        for c, t in zip(classes, targets, strict=True):
            counts[c][str(t)] += 1
        m = metrics(probabilities, y)
        m["classes"] = dict(counts)
        m["unique_videos"] = len(set(groups))
        m["global_majority_accuracy"] = max(sum(targets), len(targets) - sum(targets)) / len(
            targets
        )
        report["results"][name] = m
        np.savez_compressed(
            args.output / f"{name}.npz",
            probabilities=probabilities,
            targets=y,
            ids=[r["record_id"] for r in subset],
            groups=groups,
        )
    write(args.output / "result.json", report)
    print(
        json.dumps(
            {
                k: {m: r[m] for m in ("accuracy", "nll", "brier", "ece")}
                for k, r in report["results"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
