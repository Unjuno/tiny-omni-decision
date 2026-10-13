"""Fixed question-only control for the admitted development audio-question task."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.starss23_readout_common import binary_metrics
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def inputs(features):
    # Reserved audio dimensions remain zero in this question-only condition.
    shared = F.pad(features, (384, 0))
    return torch.cat((shared[:, None].expand(-1, 2, -1),
                      torch.eye(2).expand(len(features), -1, -1)), dim=-1)


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    feature_dir = root / "clotho-aqa-question-features-01"
    manifests = root / "clotho-aqa-development-split-01"
    metadata = json.loads((feature_dir / "result.json").read_text())
    output = root / "clotho-aqa-question-control-01"
    output.mkdir(exist_ok=False)
    protocol = dict(seed=17, epochs=24, batch_size=128, learning_rate=0.001,
                    weight_decay=0.001, selector="earliest minimum raw validation NLL",
                    calibration="17 geometric temperatures .25..4, calibration split only",
                    condition="question only, audio dimensions zero", features=metadata,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    x, y, rows = {}, {}, {}
    for split, info in metadata["splits"].items():
        data = (manifests / (split + ".json")).read_bytes()
        path = feature_dir / (split + ".npy")
        if hashlib.sha256(data).hexdigest() != info["input_sha256"]:
            raise ValueError("input identity changed")
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["feature_sha256"]:
            raise ValueError("features changed")
        rows[split] = json.loads(data)
        x[split] = np.load(path, allow_pickle=False)
        y[split] = torch.tensor([r["target"] for r in rows[split]])
        if x[split].shape != (len(rows[split]), 384) or not np.isfinite(x[split]).all():
            raise ValueError("invalid features")
    mean, std = x["train"].mean(0), x["train"].std(0)
    std = np.where(std > 1e-6, std, 1)
    np.savez(output / "normalization.npz", mean=mean, std=std)
    x = {s: inputs(torch.from_numpy((a - mean) / std)) for s, a in x.items()}
    torch.set_num_threads(4)
    torch.manual_seed(17)
    model = FrozenFeatureCandidateScorer(770, 64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    generator = torch.Generator().manual_seed(17)
    best, history = float("inf"), []
    started = time.perf_counter()
    for epoch in range(1, 25):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(y["train"]), generator=generator).split(128):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x["train"][batch]), y["train"][batch])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            p = model(x["validation"]).softmax(-1)[:, 1].numpy()
        metrics = binary_metrics(y["validation"].numpy(), p)
        history.append(dict(epoch=epoch, train_ce=total / len(y["train"]), **metrics))
        if metrics["nll"] < best:
            best, best_epoch = metrics["nll"], epoch
            torch.save(model.state_dict(), output / "best.pt")
    torch.save(model.state_dict(), output / "final.pt")
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True))
    with torch.inference_mode():
        cal = model(x["calibration"])
        grid = np.geomspace(0.25, 4, 17)
        temperature = float(grid[np.argmin([
            float(F.cross_entropy(cal / float(t), y["calibration"])) for t in grid])])
        logits = model(x["validation"])
        reversed_logits = model(x["validation"].flip(1)).flip(1)
        if not torch.equal(logits, reversed_logits):
            raise ValueError("option permutation changed logits")
        probs = (logits / temperature).softmax(-1)[:, 1].numpy()
    np.save(output / "validation_probabilities.npy", probs)
    (output / "validation_ids.json").write_text(json.dumps([
        [r["source_id"], r["question"]] for r in rows["validation"]]))
    prior = float(y["train"].float().mean())
    result = dict(best_epoch=best_epoch, temperature=temperature,
                  metrics=binary_metrics(y["validation"].numpy(), probs),
                  train_prior=binary_metrics(y["validation"].numpy(), np.full(len(probs), prior)),
                  parameters=sum(p.numel() for p in model.parameters()),
                  seconds=time.perf_counter() - started, history=history,
                  best_sha256=hashlib.sha256((output / "best.pt").read_bytes()).hexdigest(),
                  unique_train_questions=len(y["train"]),
                  consumed_train_questions=24 * len(y["train"]))
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}))


if __name__ == "__main__":
    main()
