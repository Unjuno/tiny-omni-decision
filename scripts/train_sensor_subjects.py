"""Fixed subject-disjoint native sensor experiment and static control."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.range_generalization import sha, write
from tiny_omni_decision.native_sensor import ACTIVITY_CODES, NativeSensorDecision

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")


class StaticSensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(24, 64), nn.GELU(), nn.Linear(64, 12))

    def forward(self, x, times):
        features = torch.cat((x.mean(1), x.std(1), x.amin(1), x.amax(1)), -1)
        return self.head(features)


def metrics(p, y):
    correct = p.argmax(1) == y
    losses = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-30))
    confidence = p.max(1)
    ece = 0.0
    for i in range(10):
        mask = (confidence >= i / 10) & (confidence < (i + 1) / 10)
        if i == 9:
            mask |= confidence == 1
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    classes = {}
    for c in range(p.shape[1]):
        mask = y == c
        classes[str(c)] = {
            "support": int(mask.sum()),
            "recall": float(correct[mask].mean()) if mask.any() else None,
            "nll": float(losses[mask].mean()) if mask.any() else None,
        }
    return {
        "count": len(y),
        "accuracy": float(correct.mean()),
        "nll": float(losses.mean()),
        "brier": float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()),
        "ece": float(ece),
        "classes": classes,
        "balanced_accuracy": float(
            np.mean([v["recall"] for v in classes.values() if v["support"]])
        ),
    }


def load(split, manifest, smoke):
    arrays = {k: [] for k in ("values", "timestamps_ms", "target", "mixed", "ids", "subjects")}
    for row in manifest["subjects"]:
        if row["split"] != split:
            continue
        path = Path(row["path"])
        if sha(path) != row["sha256"]:
            raise ValueError("prepared subject hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            n = min(16, len(data["target"])) if smoke else len(data["target"])
            for k in ("values", "timestamps_ms", "target", "mixed"):
                arrays[k].append(data[k][:n])
            arrays["ids"].append(np.array([f"{row['subject']}:{i}" for i in range(n)]))
            arrays["subjects"].append(np.repeat(row["subject"], n))
    result = {k: np.concatenate(v) for k, v in arrays.items()}
    result["target"] = np.array([ACTIVITY_CODES.index(int(y)) for y in result["target"]])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = ROOT / ("sensor-subject-smoke-01" if args.smoke else "sensor-subject-training-01")
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    device = "cpu" if args.smoke else "cuda"
    manifest_path = ROOT / "harth-subject-windows-01/result.json"
    manifest = json.loads(manifest_path.read_text())
    # The smoke is training-only; the full run alone reads validation arrays.
    train = load("train", manifest, args.smoke)
    val = train if args.smoke else load("validation", manifest, False)
    write(
        output / "config.json",
        {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "manifest_sha256": sha(manifest_path),
            "seed": 17,
            "epochs": 1 if args.smoke else 12,
            "batch": 128,
            "lr": 0.001,
            "weight_decay": 0.01,
            "device": device,
            "smoke": args.smoke,
            "class_index_to_activity_code": ACTIVITY_CODES,
            "torch": torch.__version__,
            "train_windows": len(train["target"]),
            "evaluation_windows": len(val["target"]),
        },
    )
    tensors = {
        s: (
            torch.tensor(d["values"], device=device),
            torch.tensor(d["timestamps_ms"], device=device),
            torch.tensor(d["target"], device=device),
        )
        for s, d in (("train", train), ("val", val))
    }
    summaries = {}
    for mode, constructor in (("temporal", NativeSensorDecision), ("static", StaticSensor)):
        torch.manual_seed(17)
        model = constructor().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        folder = output / mode
        folder.mkdir()
        started = time.perf_counter()
        history = []
        best = float("inf")
        generator = torch.Generator().manual_seed(17)

        def predict(shuffled=False, model=model):
            model.eval()
            x, t, _ = tensors["val"]
            permutation = torch.randperm(128, generator=torch.Generator().manual_seed(17))
            results = []
            with torch.inference_mode():
                for a in range(0, len(x), 128):
                    batch = x[a : a + 128]
                    if shuffled:
                        batch = batch[:, permutation.to(device)]
                    results.append(model(batch, t[a : a + 128]).double().softmax(-1).cpu().numpy())
            return np.concatenate(results)

        for epoch in range((1 if args.smoke else 12) + 1):
            loss_sum = 0.0
            if epoch:
                model.train()
                x, t, y = tensors["train"]
                order = torch.randperm(len(x), generator=generator)
                for indices in order.split(128):
                    indices = indices.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(x[indices], t[indices]), y[indices])
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite loss")
                    loss.backward()
                    if any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError("nonfinite gradient")
                    optimizer.step()
                    loss_sum += loss.item() * len(indices)
            p = predict()
            score = metrics(p, val["target"])
            record = {
                "epoch": epoch,
                "train_ce": loss_sum / len(train["target"]) if epoch else None,
                "validation": score,
                "seconds": time.perf_counter() - started,
            }
            history.append(record)
            np.savez_compressed(
                folder / f"epoch-{epoch}.npz",
                probabilities=p,
                targets=val["target"],
                ids=val["ids"],
                subjects=val["subjects"],
                mixed=val["mixed"],
            )
            if score["nll"] < best:
                best, best_epoch = score["nll"], epoch
                torch.save(model.state_dict(), folder / "best.pt")
            write(folder / "history.json", history)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "epoch": epoch,
                        "nll": score["nll"],
                        "accuracy": score["accuracy"],
                        "best_epoch": best_epoch,
                    }
                ),
                flush=True,
            )
        torch.save(model.state_dict(), folder / "final.pt")
        model.load_state_dict(
            torch.load(folder / "best.pt", weights_only=True, map_location=device)
        )
        p = predict()
        with np.load(folder / f"epoch-{best_epoch}.npz") as saved:
            if not np.array_equal(p, saved["probabilities"]):
                raise ValueError("best reload mismatch")
        shuffled = predict(True)
        np.savez_compressed(
            folder / "selected-time-shuffled.npz",
            probabilities=shuffled,
            ids=val["ids"],
            targets=val["target"],
        )
        summaries[mode] = {
            "best_epoch": best_epoch,
            "selected": metrics(p, val["target"]),
            "shuffled": metrics(shuffled, val["target"]),
            "exact_reload": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": time.perf_counter() - started,
            "best_sha256": sha(folder / "best.pt"),
            "final_sha256": sha(folder / "final.pt"),
            "peak_vram": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        }
        write(output / "result.json", summaries)


if __name__ == "__main__":
    main()
