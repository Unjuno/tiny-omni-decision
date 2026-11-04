"""Execute the frozen training-only native sensor smoke; never reads held-out data."""

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.range_generalization import sha, write
from tiny_omni_decision.native_sensor import ACTIVITY_CODES, NativeSensorDecision, decide_sensor
from tiny_omni_decision.observations import ObservationBatch

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def main():
    source = ROOT / "harth-native-window-smoke-02"
    output = ROOT / "native-sensor-decision-smoke-01"
    expected = {
        "observation-features.npz": (
            "52ee95851ba8dc5128518ba12ec07e04ed61ac83c5a9b5a9c940fb0aa8ccd702"
        ),
        "supervision.npz": "7ed4020b79cacd8eda4512441b54892843be2eb69cde1fe50e2720f7b915af7f",
        "first-window-request.json": (
            "14821ee1e331ab26084983231c494e67cf8fb67e37c7b0fbb9a408b61eccb8dd"
        ),
    }
    for name, digest in expected.items():
        if sha(source / name) != digest:
            raise ValueError(f"source hash mismatch: {name}")
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    with np.load(source / "observation-features.npz", allow_pickle=False) as data:
        values = torch.tensor(data["values"][:8])
        times = torch.tensor(data["timestamps_ms"][:8])
    with np.load(source / "supervision.npz", allow_pickle=False) as data:
        labels = data["label_sequence"][:8, 64].tolist()
    targets = torch.tensor([ACTIVITY_CODES.index(label) for label in labels])
    model = NativeSensorDecision()
    torch.save(model.state_dict(), output / "initial.pt")
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    history = []
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(values, times), targets)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.stack([p.grad.norm() ** 2 for p in model.parameters()]).sum().sqrt()
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        history.append({"step": step, "loss": loss.item(), "gradient_norm": norm.item()})
    assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
    torch.save(model.state_dict(), output / "final.pt")
    request = ObservationBatch.model_validate_json(
        (source / "first-window-request.json").read_text()
    )
    model.eval()
    prediction = decide_sensor(model, request)
    json.dumps(prediction, allow_nan=False)
    assert abs(sum(o["probability"] for o in prediction["options"]) - 1) < 1e-12
    restored = NativeSensorDecision().eval()
    restored.load_state_dict(torch.load(output / "final.pt", weights_only=True))
    assert decide_sensor(restored, request) == prediction
    reversed_request = request.model_copy(update={"options": tuple(reversed(request.options))})
    reversed_result = decide_sensor(restored, reversed_request)
    assert reversed_result["selected_option_id"] == prediction["selected_option_id"]
    assert reversed_result["options"] == list(reversed(prediction["options"]))
    result = {
        "status": "functional_smoke_pass",
        "quality_measured": False,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": expected,
        "seed": 17,
        "updates": 2,
        "unique_training_windows": 8,
        "training_encounters": 16,
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01},
        "backend": "CPU float32",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "history": history,
        "prediction": prediction,
        "exact_reload": True,
        "exact_option_permutation": True,
        "seconds": time.perf_counter() - started,
        "artifacts": {
            name: {"sha256": sha(output / name), "bytes": (output / name).stat().st_size}
            for name in ("initial.pt", "final.pt")
        },
    }
    write(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
