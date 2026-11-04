"""Training-only timestamp perturbation; fixed endpoint, no optimization."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_correspondence_validation import sha, write
from scripts.train_slot_readout import load_train, summary
from scripts.train_trajectory_readout import predict
from tiny_omni_decision.trajectory_readout import TrajectoryReadout


def perturb(bank, mode):
    result = {key: value.clone() for key, value in bank.items()}
    if mode not in ("normal", "reverse_time", "zero_time"):
        raise ValueError("unknown perturbation")
    for value in result.values():
        if mode == "reverse_time":
            value[:, -1] = 1 - value[:, -1]
        elif mode == "zero_time":
            value[:, -1] = 0
    return result


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    output = root / "perception-occlusion-slot-time-diagnostic-01"
    output.mkdir(exist_ok=False)
    write(output / "protocol.json", dict(
        scope="all99 train videos only; no validation or learning",
        modes=["normal", "reverse_time", "zero_time"],
        interpretation="input sensitivity, not proof of temporal reasoning",
    ))
    bank, data, hashes = load_train(
        Path("C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-slots-01")
    )
    checkpoint = root / "perception-occlusion-slot-train-readout-01/final.pt"
    expected = "3d12ea26148724ba0113e251b29c1e69e9aca9ff7c3bd3f95aed6d685c97b765"
    assert sha(checkpoint) == expected
    torch.set_num_threads(4)
    model = TrajectoryReadout(input_dim=129)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.requires_grad_(False)
    results, probabilities = {}, {}
    for mode in ("normal", "reverse_time", "zero_time"):
        p = predict(model, perturb(bank, mode), data)
        probabilities[mode] = p
        results[mode] = dict(
            metrics=summary(p, data["targets"].numpy()),
            max_probability_delta=float(np.abs(p - probabilities["normal"]).max()),
            changed_predictions=int((p.argmax(-1) != probabilities["normal"].argmax(-1)).sum()),
        )
    assert sha(checkpoint) == expected
    write(output / "predictions.json", {k: v.tolist() for k, v in probabilities.items()})
    write(output / "result.json", dict(results=results, input_hashes=hashes,
          checkpoint_sha256=expected, script_sha256=sha(Path(__file__))))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
