"""One fixed train-only optimization diagnostic; never reads validation data."""

import hashlib
import json
import time
from pathlib import Path

import torch

from scripts.train_joint_video_readout import load_features


def main():
    out = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-train-convergence-01"
    )
    out.mkdir(exist_ok=False)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    pair_path = root / "perception-occlusion-pairs-01/train.json"
    assert (
        hashlib.sha256(pair_path.read_bytes()).hexdigest()
        == "8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7"
    )
    features_dir = Path("C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-train-features-01")
    assert (
        hashlib.sha256((features_dir / "predictions.jsonl").read_bytes()).hexdigest()
        == "8d061543600446f0dfd659feeb1e88c6c039c7f8cfab3104141c92a5fe9be3ef"
    )
    pairs = json.loads(pair_path.read_text())["pairs"]
    features, targets, _ = load_features(features_dir, pairs)
    config = dict(
        scope="train-only convex optimization diagnostic, no validation selection or evaluation",
        dtype="float64",
        threads=4,
        initialization="zero",
        optimizer="LBFGS",
        max_iter=200,
        max_eval=250,
        line_search_fn="strong_wolfe",
        l2=0.0001,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )
    (out / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(4)
    results = {}
    for mode, x in features.items():
        start = time.perf_counter()
        x = x.double()
        w = torch.nn.Parameter(torch.zeros(192, dtype=torch.float64))
        history = []
        optimizer = torch.optim.LBFGS(
            [w],
            max_iter=200,
            max_eval=250,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure(optimizer=optimizer, x=x, w=w, history=history):
            optimizer.zero_grad()
            ce = torch.nn.functional.cross_entropy(x @ w, targets)
            objective = ce + 0.0001 * w.square().sum() / 2
            objective.backward()
            assert torch.isfinite(objective) and torch.isfinite(w.grad).all()
            history.append(
                dict(
                    ce=float(ce.detach()),
                    objective=float(objective.detach()),
                    gradient_max=float(w.grad.abs().max()),
                )
            )
            return objective

        optimizer.step(closure)
        closure()
        with torch.no_grad():
            pred = (x @ w).argmax(-1)
            correct = (pred == targets).reshape(-1, 2)
            changed = targets.reshape(-1, 2)[:, 0] != targets.reshape(-1, 2)[:, 1]
        result = dict(
            accuracy=float(correct.double().mean()),
            both_correct=float(correct.all(-1).double().mean()),
            changed_both_correct=float(correct[changed].all(-1).double().mean()),
            ce=history[-1]["ce"],
            objective=history[-1]["objective"],
            gradient_max=history[-1]["gradient_max"],
            iterations=optimizer.state[w]["n_iter"],
            closure_calls=len(history),
            seconds=time.perf_counter() - start,
            weight_l2=float(w.detach().norm()),
        )
        torch.save(w.detach(), out / f"{mode}.pt")
        (out / f"{mode}-history.json").write_text(json.dumps(history, indent=2))
        results[mode] = result
    (out / "result.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
