"""Fixed gated fusion experiment; no search or legacy artifact changes."""

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.range_generalization import sha, write
from scripts.train_sensor_subjects import metrics
from scripts.train_vidimu_sensor_arm import PINS
from scripts.train_vidimu_sensor_arm import load as load_sensor
from tiny_omni_decision.fusion_decision import FusionDecisionScorer

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


def load(name, arm):
    sensor, placement, targets, rows = load_sensor(name)
    split = "train" if name == "training" else "validation"
    folder = ROOT / "vidimu-video-features-01"
    report = json.loads((folder / "result.json").read_text())
    if sha(folder / f"{split}.npz") != report["feature_hashes"][split]:
        raise ValueError("video cache hash mismatch")
    video_rows = json.loads((folder / f"{split}-records.json").read_text())
    def identity(row):
        return row["subject"], row["activity"], row["trial"]

    if list(map(identity, rows)) != list(map(identity, video_rows)):
        raise ValueError("video/sensor row identities or order differ")
    with np.load(folder / f"{split}.npz", allow_pickle=False) as saved:
        video = saved["values"]
    if video.shape != (len(rows), 8, 768) or not np.isfinite(video).all():
        raise ValueError("invalid video cache")
    features = video.reshape(len(rows), -1)
    if arm == "joint":
        features = np.concatenate((features, sensor), axis=1)
    return features, placement, targets, rows


class VidimuFusion(torch.nn.Module):
    """Fixed activity options, placement context; no mandatory text path."""

    def __init__(self):
        super().__init__()
        self.scorer = FusionDecisionScorer({"video": 6144, "sensor": 360}, 13, 2, width=64)
        self.register_buffer("options", torch.eye(13))

    def forward(self, x, missing=None, option_order=None):
        if missing not in (None, "video", "sensor"):
            raise ValueError("unknown missing modality")
        if x.ndim != 2 or x.shape[1] != 6506:
            raise ValueError("6506 feature columns required")
        batch = len(x)
        features = {"video": x[:, :6144], "sensor": x[:, 6144:6504]}
        masks = {name: torch.full((batch,), name != missing, dtype=torch.bool, device=x.device)
                 for name in features}
        options = self.options if option_order is None else self.options[option_order]
        return self.scorer(features, masks, options.expand(batch, -1, -1),
                           torch.ones(batch, 13, dtype=torch.bool, device=x.device),
                           x[:, 6504:], torch.ones(batch, dtype=torch.bool, device=x.device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("joint",), default="joint")
    arm = parser.parse_args().arm
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    out = ROOT / "vidimu-gated-fusion-arm-01"
    out.mkdir(exist_ok=False)
    train, tp, ty, tr = load("training", arm)
    val, vp, vy, vr = load("validation", arm)
    mean, std = train.mean(0), train.std(0)
    std = np.where(std > 1e-6, std, 1)
    std[:6144] /= np.float32(np.sqrt(360 / 6144))
    tensors = [torch.from_numpy(np.concatenate(
        ((x - mean) / std, np.eye(2, dtype=np.float32)[p]), axis=1
    )) for x, p in ((train, tp), (val, vp))]
    write(out / "config.json", {"seed": 17, "epochs": 100, "lr": .001,
                               "weight_decay": .01, "optimizer": "AdamW", "batch": "full",
                               "selector": "earliest minimum validation NLL including epoch0",
                               "input_hashes": PINS, "arm": arm,
                               "architecture": "masked fusion width64, shared option MLP",
                               "video_block_scale": float(np.sqrt(360 / 6144)),
                               "module_hashes": {str(p): sha(p) for p in map(Path, (
                                   "src/tiny_omni_decision/fusion_decision.py",
                                   "src/tiny_omni_decision/masked_fusion.py",
                                   "src/tiny_omni_decision/decision.py"))},
                               "video_report_sha256": sha(
                                   ROOT / "vidimu-video-features-01/result.json"),
                               "script_sha256": sha(Path(__file__)),
                               "device": "cpu", "dtype": "float32",
                               "torch": torch.__version__, "source_commit": subprocess.check_output(
                                   ["git", "rev-parse", "HEAD"], text=True).strip()})
    write(out / "records.json", {"train": tr, "validation": vr})
    np.savez_compressed(out / "normalization.npz", mean=mean, std=std)
    model = VidimuFusion()
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
            gradient_norm = float(torch.sqrt(sum(
                p.grad.square().sum() for p in model.parameters())))
            optimizer.step()
        with torch.inference_mode():
            probabilities = [model(x).softmax(-1).numpy() for x in tensors]
        results = [metrics(p, y) for p, y in zip(probabilities, (ty, vy), strict=True)]
        history.append({"epoch": epoch, "train": results[0], "validation": results[1],
                        "gradient_norm": gradient_norm if epoch else None})
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
              "resident_parameters": 114365184 + sum(p.numel() for p in model.parameters()),
              "wall_seconds": time.perf_counter() - started, "exact_reload": True,
              "best_sha256": sha(out / "best.pt"), "final_sha256": sha(out / "final.pt"),
              "sealed_audit_read": False}
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("best_epoch", "wall_seconds", "exact_reload")}))


if __name__ == "__main__":
    main()

