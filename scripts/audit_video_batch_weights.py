"""Replay the completed token run's sampler and quantify objective-weight variability."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.range_generalization import sha, write
from scripts.train_frozen_clevrer_binary_reasoning import _sample_weights, _validate_inputs


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/video-token-readout-01")
    output = Path("C:/CodexArtifacts/pretrained-reuse-r2/video-batch-weight-audit-01")
    if output.exists():
        raise FileExistsError(output)
    path = Path("configs/pretrained_reuse/path_a_videoprism_clevrer_binary_reasoning.yaml")
    checked = _validate_inputs(path, yaml.safe_load(path.read_text()))
    rows = checked["train_rows"]
    recorded = json.loads((root / "config.json").read_text())
    if [r["id"] for r in rows] != recorded["ids"]["train"]:
        raise ValueError("training order changed")
    weights = _sample_weights(rows).numpy().astype(np.float64)
    generator = torch.Generator().manual_seed(17)
    digest, records = hashlib.sha256(), []
    for epoch in range(1, 25):
        order = torch.randperm(len(rows), generator=generator).tolist()
        digest.update(np.asarray(order, dtype="<i8").tobytes())
        for offset in range(0, len(order), 64):
            ids = order[offset : offset + 64]
            w = weights[ids]
            records.append(
                dict(
                    epoch=epoch,
                    batch=offset // 64,
                    count=len(ids),
                    loss_mass=float(w.sum() * len(rows) / len(ids)),
                    effective_sample_size=float(w.sum() ** 2 / (w * w).sum()),
                    maximum_weight_fraction=float(w.max() / w.sum()),
                    correct_weight_fraction=float(
                        sum(weights[i] for i in ids if rows[i]["target"] == "correct") / w.sum()
                    ),
                )
            )
    result = json.loads((root / "result.json").read_text())
    if any(r["order_sha256"] != digest.hexdigest() for r in result["conditions"].values()):
        raise ValueError("sampler replay differs from actual run")
    summary = {
        key: dict(
            minimum=float(min(r[key] for r in records)),
            median=float(np.median([r[key] for r in records])),
            maximum=float(max(r[key] for r in records)),
            std=float(np.std([r[key] for r in records])),
        )
        for key in [
            "loss_mass",
            "effective_sample_size",
            "maximum_weight_fraction",
            "correct_weight_fraction",
        ]
    }
    output.mkdir()
    write(
        output / "result.json",
        dict(
            status="PASS_SAMPLER_REPLAY_ONLY",
            runner_sha256=sha(Path(__file__)),
            source_result_sha256=sha(root / "result.json"),
            order_sha256=digest.hexdigest(),
            unique_train_examples=len(rows),
            batches=len(records),
            steps_per_epoch=len(records) // 24,
            global_weight_ess=float(weights.sum() ** 2 / (weights * weights).sum()),
            sample_weight_ratio=float(weights.max() / weights.min()),
            summary=summary,
            records=records,
            limitation="Loss-weight variability is not a measured gradient or causal explanation.",
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
