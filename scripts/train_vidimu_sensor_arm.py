"""Execute the preregistered sensor arm; no video-dependent settings or audit."""

import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.range_generalization import sha, write
from scripts.train_sensor_subjects import metrics

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PINS = {
    "training": ("84bead6d313839b7fa0949bebc4d66af2574c53f393ce717459359eb61496925",
                 "72d6934344b746af1d76c6eeb3dc380104dc41967a025e175004a1a617b0bdf6"),
    "validation": ("1329f39d60b179446bcef095f7685e58bba478925b56196ebc877f3f0237486e",
                   "fbaf822aff1ce4844dd6481470b4e84c2fa9130d3f56a21ad29972a2b23a6792"),
}


def load(name):
    folder = ROOT / f"vidimu-{name}-sensor-features-01"
    for filename, expected in zip(("features.npz", "records.json"), PINS[name], strict=True):
        if sha(folder / filename) != expected:
            raise ValueError("input changed")
    rows = json.loads((folder / "records.json").read_text(encoding="utf-8"))
    with np.load(folder / "features.npz", allow_pickle=False) as archive:
        features = archive["values"].reshape(len(rows), -1)
    layouts = [r["sensors"][0] for r in rows]
    if not set(layouts) <= {"pelvis_imu", "torso_imu"}:
        raise ValueError("unknown placement")
    placement = np.asarray([int(s == "torso_imu") for s in layouts])
    targets = np.asarray([int(r["activity"][1:]) - 1 for r in rows])
    return features, placement, targets, rows


def main():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    out = ROOT / "vidimu-sensor-arm-01"
    out.mkdir(exist_ok=False)
    train, tp, ty, tr = load("training")
    val, vp, vy, vr = load("validation")
    mean, std = train.mean(0), train.std(0)
    std = np.where(std > 1e-6, std, 1)
    tensors = [torch.from_numpy(np.concatenate(
        ((x - mean) / std, np.eye(2, dtype=np.float32)[p]), axis=1
    )) for x, p in ((train, tp), (val, vp))]
    write(out / "config.json", {"seed": 17, "epochs": 100, "lr": .001,
                               "weight_decay": .01, "optimizer": "AdamW", "batch": "full",
                               "selector": "earliest minimum validation NLL including epoch0",
                               "input_hashes": PINS, "device": "cpu", "dtype": "float32",
                               "torch": torch.__version__, "source_commit": subprocess.check_output(
                                   ["git", "rev-parse", "HEAD"], text=True).strip()})
    write(out / "records.json", {"train": tr, "validation": vr})
    np.savez_compressed(out / "normalization.npz", mean=mean, std=std)
    model = torch.nn.Linear(tensors[0].shape[1], 13)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    started = time.perf_counter()
    best_nll, best_epoch, best_state = float("inf"), None, None
    history = []
    for epoch in range(101):
        if epoch:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(tensors[0]), torch.from_numpy(ty))
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all()
                                               for p in model.parameters()):
                raise ValueError("nonfinite fit")
            optimizer.step()
        with torch.inference_mode():
            probabilities = [model(x).softmax(-1).numpy() for x in tensors]
        results = [metrics(p, y) for p, y in zip(probabilities, (ty, vy), strict=True)]
        history.append({"epoch": epoch, "train": results[0], "validation": results[1]})
        np.savez_compressed(out / f"epoch-{epoch:03}.npz", train=probabilities[0],
                            validation=probabilities[1], train_targets=ty, validation_targets=vy)
        if results[1]["nll"] < best_nll:
            best_nll, best_epoch = results[1]["nll"], epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out / "best.pt")
        write(out / "history.json", history)
    torch.save(model.state_dict(), out / "final.pt")
    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    with torch.inference_mode():
        p = model(tensors[1]).softmax(-1).numpy()
    with np.load(out / f"epoch-{best_epoch:03}.npz") as saved:
        np.testing.assert_array_equal(p, saved["validation"])
    priors = np.stack([np.bincount(ty[tp == group], minlength=13) + 1 for group in range(2)])
    priors = priors / priors.sum(1, keepdims=True)
    result = {"best_epoch": best_epoch, "selected": metrics(p, vy),
              "placement_prior": metrics(priors[vp], vy),
              "by_placement": {str(g): metrics(p[vp == g], vy[vp == g]) for g in range(2)},
              "by_subject": {s: metrics(p[np.array([r['subject'] == s for r in vr])],
                                        vy[np.array([r['subject'] == s for r in vr])])
                             for s in sorted({r['subject'] for r in vr})},
              "parameters": sum(p.numel() for p in model.parameters()),
              "wall_seconds": time.perf_counter() - started, "exact_reload": True,
              "best_sha256": sha(out / "best.pt"), "final_sha256": sha(out / "final.pt"),
              "sealed_audit_read": False}
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("best_epoch", "wall_seconds", "exact_reload")}))


if __name__ == "__main__":
    main()
