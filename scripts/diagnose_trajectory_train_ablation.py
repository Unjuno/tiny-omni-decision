"""Train-only replay of existing best/final checkpoints; no fitting."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_correspondence_validation import sha, write
from scripts.train_trajectory_readout import load_inputs, predict, summarize
from tiny_omni_decision.trajectory_readout import TrajectoryReadout

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def main():
    output = ROOT / "perception-occlusion-trajectory-train-ablation-01.json"
    if output.exists():
        raise ValueError("output already exists")
    bank, data, hashes = load_inputs(
        ROOT / "perception-occlusion-trajectory-features-01",
        ROOT / "perception-occlusion-text-features-01",
    )
    train = data["train"]
    x = train["text"].numpy().reshape(-1, 2, 3, 1536)
    delta = np.linalg.norm(x[:, 0] - x[:, 1], axis=-1)
    scale = np.linalg.norm(x[:, 0], axis=-1)
    result = dict(
        scope="train only; best/final fixed checkpoints, no selection",
        input_hashes=hashes,
        condition_relative_l2=dict(
            minimum=float((delta / scale).min()),
            median=float(np.median(delta / scale)),
            maximum=float((delta / scale).max()),
        ),
        exactly_equal_condition_vectors=int((delta == 0).sum()),
        checkpoints={},
    )
    torch.set_num_threads(4)
    run = ROOT / "perception-occlusion-trajectory-readout-01"
    ledger = json.loads((run / "result.json").read_text())
    for mode in ("attention",):
        for checkpoint in ("best", "final"):
            path = run / mode / f"{checkpoint}.pt"
            if sha(path) != ledger[mode]["artifacts"][path.name]["sha256"]:
                raise ValueError("checkpoint changed")
            model = TrajectoryReadout()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            for control in ("normal", "zero_geometry", "zero_appearance", "misaligned"):
                controlled = {}
                for key, value in bank.items():
                    altered = value.clone()
                    if control == "zero_geometry":
                        altered[:, :128] = 0
                    elif control == "zero_appearance":
                        altered[:, 128:] = 0
                    elif control == "misaligned":
                        altered[:, :128] = value[:, :128].roll(1, 0)
                    controlled[key] = altered
                probabilities = predict(model, controlled, train)
                labels = probabilities.argmax(-1).reshape(-1, 2)
                result["checkpoints"][f"{mode}/{checkpoint}/{control}"] = dict(
                    sha256=sha(path),
                    metrics=summarize(probabilities, train["targets"].numpy()),
                    condition_switches=int((labels[:, 0] != labels[:, 1]).sum()),
                )
    write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
