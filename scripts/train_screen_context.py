"""Fixed CPU screen-context comparison on cached frozen features."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.range_generalization import sha, write
from scripts.train_sensor_subjects import metrics

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
old = root / "groundcua-classifier-01"
context = root / "groundcua-fullscreen-features-01"
out = root / "groundcua-context-comparison-01"
out.mkdir(exist_ok=False)
r = json.loads((context / "result.json").read_text())
assert sha(context / "features.npy") == r["features_sha256"]
assert sha(context / "records.json") == r["records_sha256"]
rows = json.loads((old / "records.json").read_text())
screens = json.loads((context / "records.json").read_text())
local = np.load(old / "features.npy")
full = np.load(context / "features.npy")
lookup = {x["screen"]: i for i, x in enumerate(screens)}
assert all(x["split"] == screens[lookup[x["screen"]]]["split"] for x in rows)
train = np.array([x["split"] == "train" for x in rows])
y = torch.tensor([x["target"] for x in rows])
assert len(rows) == 1740 and train.sum() == 1017
screen_features = full[[lookup[x["screen"]] for x in rows]]
write(
    out / "config.json",
    {
        "seed": 17,
        "epochs": 30,
        "lr": 0.001,
        "weight_decay": 0.01,
        "batch": 64,
        "selector": "earliest minimum validation NLL including epoch0",
        "script_sha256": sha(Path(__file__)),
        "inputs": {
            str(p): sha(p)
            for p in [
                old / "records.json",
                old / "features.npy",
                context / "features.npy",
                context / "records.json",
            ]
        },
    },
)
torch.set_num_threads(2)
started = time.perf_counter()
results = {}
initial = None
for arm in ["zero", "context"]:
    dest = out / arm
    dest.mkdir()
    values = np.concatenate(
        [local, screen_features if arm == "context" else np.zeros_like(screen_features)], 1
    )
    mean = values[train].mean(0)
    std = values[train].std(0)
    std = np.where(std < 1e-6, 1, std)
    np.savez(dest / "normalization.npz", mean=mean, std=std)
    x = torch.tensor((values - mean) / std)
    tx, ty, vx, vy = x[train], y[train], x[~train], y[~train]
    torch.manual_seed(17)
    model = torch.nn.Linear(1536, 8)
    if initial is None:
        initial = {k: v.clone() for k, v in model.state_dict().items()}
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    torch.save(model.state_dict(), dest / "initial.pt")
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    generator = torch.Generator().manual_seed(17)
    history = []
    best = float("inf")
    best_epoch = 0
    for epoch in range(31):
        total = 0.0
        if epoch:
            for indices in torch.randperm(len(tx), generator=generator).split(64):
                optim.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(tx[indices]), ty[indices])
                assert torch.isfinite(loss)
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters())
                optim.step()
                total += loss.item() * len(indices)
        with torch.inference_mode():
            p = model(vx).double().softmax(-1).numpy()
        score = metrics(p, vy.numpy())
        history.append(
            {"epoch": epoch, "train_ce": total / len(tx) if epoch else None, "validation": score}
        )
        np.save(dest / f"epoch-{epoch}.npy", p)
        if score["nll"] < best:
            best = score["nll"]
            best_epoch = epoch
            torch.save(model.state_dict(), dest / "best.pt")
    torch.save(model.state_dict(), dest / "final.pt")
    model.load_state_dict(torch.load(dest / "best.pt", weights_only=True))
    with torch.inference_mode():
        p = model(vx).double().softmax(-1).numpy()
    assert np.array_equal(p, np.load(dest / f"epoch-{best_epoch}.npy"))
    apps = np.array([x["app"] for x in rows])[~train]
    result = {
        "best_epoch": best_epoch,
        "selected": metrics(p, vy.numpy()),
        "exact_reload": True,
        "head_parameters": sum(p.numel() for p in model.parameters()),
        "apps": {a: metrics(p[apps == a], vy.numpy()[apps == a]) for a in sorted(set(apps))},
    }
    if arm == "context":
        ids = sorted({x["screen"] for x in rows if x["split"] == "validation"})
        shift = dict(zip(ids, ids[1:] + ids[:1], strict=True))
        vrows = [x for x in rows if x["split"] == "validation"]
        wrong = np.concatenate(
            [local[~train], full[[lookup[shift[x["screen"]]] for x in vrows]]], 1
        )
        with torch.inference_mode():
            q = model(torch.tensor((wrong - mean) / std)).double().softmax(-1).numpy()
        np.save(dest / "shifted-context.npy", q)
        result["shifted_context"] = metrics(q, vy.numpy())
        result["screen_shift"] = shift
    write(dest / "history.json", history)
    result["hashes"] = {p.name: sha(p) for p in dest.iterdir() if p.is_file()}
    write(dest / "result.json", result)
    results[arm] = result
write(
    out / "result.json",
    {
        "results": results,
        "seconds": time.perf_counter() - started,
        "initial_tensors_equal": True,
        "encoder_parameters": 22056576,
    },
)
print(
    json.dumps(
        {
            k: {
                "epoch": v["best_epoch"],
                "accuracy": v["selected"]["accuracy"],
                "nll": v["selected"]["nll"],
            }
            for k, v in results.items()
        }
    )
)
