"""Replay consumed positive queries; no evaluation selection or additional fitting."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.fetch_starss23_frozen_media import write_json
from scripts.fsd_acquisition_gate import complete_acquisition
from scripts.fsd_positive_queries import positive_queries
from scripts.train_starss23_readouts import candidate_inputs
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    output = root / "fsd-positive-training-fit-01"
    acquisition = complete_acquisition(root / "fsd50k-training-128-02")
    queries = positive_queries(acquisition["selection"])
    feature_root = root / "fsd50k-features-128-01"
    report = json.loads((feature_root / "result.json").read_text())
    if sha256_file(feature_root / "features.npy") != report["feature_sha256"]:
        raise ValueError("features changed")
    if [r["id"] for r in report["rows"]] != [q["clip_id"] for q in queries]:
        raise ValueError("feature order mismatch")
    raw = np.load(feature_root / "features.npy", allow_pickle=False)
    baseline = root / "starss-pretrained-head-comparison-01/pooled"
    candidate = root / "starss-fsd-positive-128-run-01"
    normalization = np.load(candidate / "normalization.npz", allow_pickle=False)
    reference = np.load(baseline / "normalization.npz", allow_pickle=False)
    for key in ("mean", "std"):
        if not np.array_equal(normalization[key], reference[key]):
            raise ValueError("normalization differs between conditions")
    values = np.zeros((128, 1152), dtype=np.float32)
    values[:, :384] = (raw - normalization["mean"]) / normalization["std"]
    indices = torch.tensor([q["feature_index"] * 12 + q["event_index"] for q in queries])
    inputs = candidate_inputs(torch.from_numpy(values), indices)
    events = np.array([r["event"] for r in acquisition["selection"]])
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    result = {"selection": "consumed training positives only", "calibration_applied": False,
              "training": False, "feature_sha256": report["feature_sha256"], "conditions": {}}
    for name, directory, kind in (("baseline_best", baseline, "best"),
                                  ("augmented_best", candidate, "best"),
                                  ("augmented_final", candidate, "final")):
        metadata = json.loads((directory / "result.json").read_text())
        path = directory / f"{kind}.pt"
        if sha256_file(path) != metadata[f"{kind}_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        model = FrozenFeatureCandidateScorer(1166, 64).eval()
        model.load_state_dict(torch.load(path, weights_only=True))
        with torch.inference_mode():
            probabilities = model(inputs).softmax(-1)[:, 1].numpy()
        np.save(output / f"{name}.npy", probabilities)
        per_event = {}
        for event in sorted(set(events)):
            p = probabilities[events == event]
            per_event[event] = {
                "positives": len(p), "correct_positive_decisions": int((p > 0.5).sum()),
                "positive_only_nll": float(-np.log(np.clip(p, 1e-12, 1)).mean()),
                "mean_positive_probability": float(p.mean()),
            }
        result["conditions"][name] = {"checkpoint_sha256": sha256_file(path),
                                       "per_event": per_event}
    write_json(output / "result.json", result)
    print(json.dumps(result["conditions"]))


if __name__ == "__main__":
    main()
