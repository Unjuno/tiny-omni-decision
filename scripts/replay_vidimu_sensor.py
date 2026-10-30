"""Measured CPU reload/inference replay, without training or recalibration."""

import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from scripts.range_generalization import sha, write
from scripts.train_vidimu_sensor_arm import load
from tiny_omni_decision.classification_json import classification_json


def main():
    torch.set_num_threads(4)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = root / "vidimu-sensor-arm-01"
    expected = "b7c76c036fa3e66b4a71876c08772396d75f1421f31a01e15eac69747a8715d2"
    if sha(source / "best.pt") != expected:
        raise ValueError("best checkpoint changed")
    started = time.perf_counter()
    x, placement, _, rows = load("validation")
    with np.load(source / "normalization.npz", allow_pickle=False) as normalization:
        mean, std = normalization["mean"], normalization["std"]
    x = torch.from_numpy(np.concatenate(((x - mean) / std,
                                        np.eye(2, dtype=np.float32)[placement]), axis=1))
    model = torch.nn.Linear(362, 13).eval()
    model.load_state_dict(torch.load(source / "best.pt", weights_only=True))
    load_seconds = time.perf_counter() - started
    with torch.inference_mode():
        start = time.perf_counter()
        probabilities = model(x).softmax(-1).numpy()
        batch_seconds = time.perf_counter() - start
        times = []
        for row in x:
            start = time.perf_counter()
            model(row[None]).softmax(-1)
            times.append(time.perf_counter() - start)
    with np.load(source / "epoch-100.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(probabilities, saved["validation"])
    classes = [f"A{i:02}" for i in range(1, 14)]
    options = [{"option_id": c, "class_id": c} for c in classes]
    outputs = [classification_json(p, classes, options) for p in probabilities]
    for p, output in zip(probabilities, outputs, strict=True):
        reverse = classification_json(p, classes, options[::-1])
        if (reverse["selected_option_id"] != output["selected_option_id"]
                or reverse["options"][::-1] != output["options"]):
            raise ValueError("option permutation mismatch")
    memory = psutil.Process().memory_info()
    report = {"count": len(rows), "exact_batch_reload": True,
              "load_and_prepare_seconds": load_seconds, "batch_seconds": batch_seconds,
              "single_row_seconds_median": float(np.median(times)),
              "single_row_seconds_p95": float(np.quantile(times, .95)),
              "process_rss_bytes": memory.rss,
              "process_peak_working_set_bytes": getattr(memory, "peak_wset", None),
              "weights_bytes": (source / "best.pt").stat().st_size,
              "parameters": sum(p.numel() for p in model.parameters()),
              "scope": "cached sensor features; excludes STO decoding and feature computation",
              "training_rerun": False, "sealed_audit_read": False}
    out = root / "vidimu-sensor-runtime-replay-01"
    out.mkdir(exist_ok=False)
    write(out / "decisions.json", [{"record": r, "decision": d}
                                    for r, d in zip(rows, outputs, strict=True)])
    write(out / "result.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
