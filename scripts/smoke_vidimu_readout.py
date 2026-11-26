"""Training-only CPU fit/save/reload smoke for the fixed episode readout."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.range_generalization import sha, write


def main():
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    source = Path("C:/CodexArtifacts/pretrained-reuse-r2/vidimu-training-sensor-features-01")
    pins = {"features.npz": "84bead6d313839b7fa0949bebc4d66af2574c53f393ce717459359eb61496925",
            "records.json": "72d6934344b746af1d76c6eeb3dc380104dc41967a025e175004a1a617b0bdf6"}
    for name, expected in pins.items():
        if sha(source / name) != expected:
            raise ValueError("cache changed")
    rows = json.loads((source / "records.json").read_text(encoding="utf-8"))
    with np.load(source / "features.npz", allow_pickle=False) as archive:
        x = archive["values"].reshape(len(rows), -1)
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 1e-6, std, 1)
    y = torch.tensor([int(r["activity"][1:]) - 1 for r in rows])
    layouts = [r["sensors"][0] for r in rows]
    if not set(layouts) <= {"pelvis_imu", "torso_imu"}:
        raise ValueError("unknown sensor placement")
    placement = torch.nn.functional.one_hot(
        torch.tensor([int(name == "torso_imu") for name in layouts]), 2
    ).float()
    x = torch.cat((torch.from_numpy((x - mean) / std), placement), dim=1)
    model = torch.nn.Linear(x.shape[1], 13)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all()
                                           for p in model.parameters()):
            raise ValueError("nonfinite smoke")
        losses.append(float(loss.detach()))
        optimizer.step()
    out = Path("C:/CodexArtifacts/pretrained-reuse-r2/vidimu-readout-smoke-01")
    out.mkdir(exist_ok=False)
    torch.save(model.state_dict(), out / "smoke.pt")
    restored = torch.nn.Linear(x.shape[1], 13)
    restored.load_state_dict(torch.load(out / "smoke.pt", weights_only=True))
    with torch.inference_mode():
        probabilities = model(x).softmax(-1)
        torch.testing.assert_close(probabilities, restored(x).softmax(-1), rtol=0, atol=0)
    np.savez_compressed(out / "normalization.npz", mean=mean, std=std)
    np.savez_compressed(out / "predictions.npz", probabilities=probabilities.numpy())
    report = {"training_only": True, "optimizer_steps": 2, "losses": losses,
              "validation_read": False, "exact_reload": True, "seed": 17,
              "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "wall_seconds": time.perf_counter() - start,
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "torch_version": torch.__version__, "device": "cpu", "dtype": "float32",
              "weights_sha256": sha(out / "smoke.pt"), "input_hashes": pins,
              "not_a_quality_experiment": True}
    write(out / "result.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
