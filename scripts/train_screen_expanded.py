"""Fixed frozen crop classifier with expanded training apps; same validation."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts.range_generalization import sha, write
from scripts.train_dinov2_image_probe import WEIGHTS
from scripts.train_screen_pilot import CATEGORIES
from scripts.train_sensor_subjects import metrics


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    old = root / "groundcua-classifier-01"
    extra = root / "groundcua-train-expansion-features-01"
    report = json.loads((extra / "result.json").read_text())
    assert sha(extra / "features.npy") == report["features_sha256"]
    assert sha(extra / "records.json") == report["records_sha256"]
    output = root / "groundcua-expanded-classifier-01"
    output.mkdir(exist_ok=False)
    records = json.loads((old / "records.json").read_text())
    added = json.loads((extra / "records.json").read_text())
    records += [dict(r, split="train", target=CATEGORIES.index(r["category"])) for r in added]
    values = np.concatenate([np.load(old / "features.npy"), np.load(extra / "features.npy")])
    write(
        output / "inputs.json",
        {
            str(p): sha(p)
            for p in [
                old / "records.json",
                old / "features.npy",
                extra / "records.json",
                extra / "features.npy",
            ]
        },
    )
    torch.set_num_threads(2)
    started = time.perf_counter()
    extraction_seconds = 0.0
    train = np.array([r["split"] == "train" for r in records])
    targets = np.array([r["target"] for r in records])
    mean, std = values[train].mean(0), np.maximum(values[train].std(0), 1e-6)
    np.savez(output / "normalization.npz", mean=mean, std=std)
    x = torch.tensor((values - mean) / std)
    y = torch.tensor(targets)
    tx, ty, vx, vy = x[train], y[train], x[~train], y[~train]
    # Independent disposable training-only smoke before main initialization.
    torch.manual_seed(17)
    smoke = torch.nn.Linear(768, 8)
    loss = torch.nn.functional.cross_entropy(smoke(tx[:8]), ty[:8])
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in smoke.parameters()):
        raise ValueError("smoke gradient failure")
    del smoke
    torch.manual_seed(17)
    model = torch.nn.Linear(768, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    generator = torch.Generator().manual_seed(17)
    history, best, best_epoch = [], float("inf"), 0
    write(
        output / "config.json",
        {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "manifest_sha256": sha(extra / "result.json"),
            "model_sha256": WEIGHTS,
            "seed": 17,
            "epochs": 30,
            "lr": 0.001,
            "weight_decay": 0.01,
            "batch": 64,
            "categories": CATEGORIES,
            "train": len(tx),
            "validation": len(vx),
            "torch": torch.__version__,
            "feature_sha256": sha(extra / "features.npy"),
        },
    )
    for epoch in range(31):
        total = 0.0
        if epoch:
            for indices in torch.randperm(len(tx), generator=generator).split(64):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(tx[indices]), ty[indices])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(indices)
        with torch.inference_mode():
            p = model(vx).double().softmax(-1).numpy()
        score = metrics(p, vy.numpy())
        history.append(
            {"epoch": epoch, "train_ce": total / len(tx) if epoch else None, "validation": score}
        )
        np.save(output / f"epoch-{epoch}.npy", p, allow_pickle=False)
        if score["nll"] < best:
            best, best_epoch = score["nll"], epoch
            torch.save(model.state_dict(), output / "best.pt")
    torch.save(model.state_dict(), output / "final.pt")
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True))
    with torch.inference_mode():
        p = model(vx).double().softmax(-1).numpy()
    assert np.array_equal(p, np.load(output / f"epoch-{best_epoch}.npy"))
    prior = (np.bincount(ty.numpy(), minlength=8) + 1) / (len(ty) + 8)
    apps = np.array([r["app"] for r in records])[~train]
    result = {
        "best_epoch": best_epoch,
        "selected": metrics(p, vy.numpy()),
        "prior": metrics(np.tile(prior, (len(vy), 1)), vy.numpy()),
        "apps": {a: metrics(p[apps == a], vy.numpy()[apps == a]) for a in sorted(set(apps))},
        "exact_reload": True,
        "head_parameters": sum(p.numel() for p in model.parameters()),
        "encoder_parameters": 22056576,
        "extraction_seconds": extraction_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_allocated_vram": 0,
        "best_sha256": sha(output / "best.pt"),
        "final_sha256": sha(output / "final.pt"),
    }
    write(output / "history.json", history)
    write(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
