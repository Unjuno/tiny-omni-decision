"""Independent-process fixed training-set gap and saved validation replay."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.range_generalization import sha, write
from scripts.train_sensor_subjects import metrics

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
run = root / "groundcua-context-comparison-01"
config = json.loads((run / "config.json").read_text())
for name, digest in config["inputs"].items():
    assert sha(Path(name)) == digest
rows = json.loads((root / "groundcua-classifier-01/records.json").read_text())
screens = json.loads((root / "groundcua-fullscreen-features-01/records.json").read_text())
lookup = {r["screen"]: i for i, r in enumerate(screens)}
local = np.load(root / "groundcua-classifier-01/features.npy")
full = np.load(root / "groundcua-fullscreen-features-01/features.npy")
context = full[[lookup[r["screen"]] for r in rows]]
train = np.array([r["split"] == "train" for r in rows])
target = np.array([r["target"] for r in rows])
torch.set_num_threads(2)
result = {"input_hashes_verified": True, "arms": {}}
for arm in ["zero", "context"]:
    path = run / arm
    saved = json.loads((path / "result.json").read_text())
    for name, digest in saved["hashes"].items():
        assert sha(path / name) == digest
    norm = np.load(path / "normalization.npz")
    values = np.concatenate([local, context if arm == "context" else np.zeros_like(context)], 1)
    x = torch.tensor((values - norm["mean"]) / norm["std"])
    model = torch.nn.Linear(1536, 8).eval().requires_grad_(False)
    result["arms"][arm] = {}
    for checkpoint in ["best", "final"]:
        model.load_state_dict(torch.load(path / f"{checkpoint}.pt", weights_only=True), strict=True)
        with torch.inference_mode():
            p = model(x[train]).double().softmax(-1).numpy()
            q = model(x[~train]).double().softmax(-1).numpy()
        epoch = saved["best_epoch"] if checkpoint == "best" else 30
        assert np.array_equal(q, np.load(path / f"epoch-{epoch}.npy"))
        result["arms"][arm][checkpoint] = {
            "epoch": epoch,
            "train": metrics(p, target[train]),
            "validation": metrics(q, target[~train]),
            "independent_process_validation_exact": True,
            "checkpoint_sha256": sha(path / f"{checkpoint}.pt"),
        }
write(Path("docs/pretrained_reuse/R2_SCREEN_CONTEXT_GAP_20261013.json"), result)
print(
    {
        arm: {
            k: {
                s: (round(v[s]["accuracy"], 4), round(v[s]["nll"], 4))
                for s in ["train", "validation"]
            }
            for k, v in checkpoints.items()
        }
        for arm, checkpoints in result["arms"].items()
    }
)
