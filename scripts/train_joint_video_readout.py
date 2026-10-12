"""Fixed 192-parameter readout comparison over immutable joint features."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.report_joint_video_diagnostic import align
from scripts.train_trajectory_readout import PINS, ROOT, summarize

CHECKPOINT = "d79415e85a6ade9d14e932bead53fadd7c61248fcd72808e2c6d86d46e2e9ca7"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_features(directory, pairs):
    rows = [json.loads(x) for x in (directory / "predictions.jsonl").read_text().splitlines()]
    completion = json.loads((directory / "completion.json").read_text())
    if completion["rows"] != len(rows) or len(rows) != len(pairs) * 4:
        raise ValueError("complete feature extraction required")
    protocol = json.loads((directory / "protocol.json").read_text())
    if (
        protocol["seed"] != 17
        or protocol["frames"] != 3
        or protocol["modes"] != ["normal", "black_pixels"]
        or protocol["training"] is not False
    ):
        raise ValueError("feature protocol mismatch")
    frames = {}
    for row in rows:
        ids = row["frame_ids"]
        if len(ids) != 3 or ids != sorted(set(ids)) or ids[0] < 0:
            raise ValueError("invalid frame IDs")
        prior = frames.setdefault(row["video_id"], ids)
        if prior != ids:
            raise ValueError("condition/control frame mismatch")
    probabilities, targets = align(rows, pairs)
    indexed = {(r["video_id"], r["condition"], r["mode"]): r["features"] for r in rows}
    features = {}
    for mode in probabilities:
        array = np.asarray(
            [indexed[(p["video_id"], q["condition"], mode)] for p in pairs for q in p["questions"]],
            dtype=np.float32,
        )
        if array.shape != (len(pairs) * 2, 3, 192) or not np.isfinite(array).all():
            raise ValueError("invalid joint features")
        features[mode] = torch.from_numpy(array)
    return features, torch.from_numpy(targets.ravel()), probabilities


def fit(train, valid, targets, valid_targets, initial, output):
    output.mkdir(exist_ok=False)
    torch.manual_seed(17)
    weight = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([weight], lr=0.001, weight_decay=0.01)
    history = []
    best_nll = float("inf")
    best_epoch = None
    start = time.perf_counter()
    for epoch in range(51):
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(train @ weight, targets)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if not torch.isfinite(weight.grad).all():
                raise ValueError("nonfinite gradient")
            optimizer.step()
        with torch.no_grad():
            tp = (train @ weight).softmax(-1).numpy()
            vp = (valid @ weight).softmax(-1).numpy()
        row = dict(
            epoch=epoch,
            train=summarize(tp, targets.numpy()),
            valid=summarize(vp, valid_targets.numpy()),
        )
        history.append(row)
        if row["valid"]["nll"] < best_nll:
            best_nll, best_epoch = row["valid"]["nll"], epoch
            torch.save(weight.detach(), output / "best.pt")
            best_predictions = vp.copy()
    torch.save(weight.detach(), output / "final.pt")
    reloaded = torch.load(output / "best.pt", weights_only=True)
    with torch.no_grad():
        actual = (valid @ reloaded).softmax(-1).numpy()
    if not np.array_equal(actual, best_predictions):
        raise ValueError("best reload mismatch")
    write(output / "history.json", history)
    write(
        output / "predictions.json",
        dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            script_sha256=sha(Path(__file__)),
            environment=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                numpy=np.__version__,
                device="cpu",
                dtype="float32",
                threads=4,
            ),
            targets=valid_targets.tolist(),
            probabilities=actual.tolist(),
            final_probabilities=vp.tolist(),
        ),
    )
    return dict(
        best_epoch=best_epoch,
        final_epoch=50,
        trainable_parameters=weight.numel(),
        seconds=time.perf_counter() - start,
        exact_reload=True,
        best=history[best_epoch],
        final=history[-1],
        artifacts={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in output.iterdir()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    if sha(args.checkpoint) != CHECKPOINT:
        raise ValueError("checkpoint changed")
    pairs = {}
    for split, pin in PINS.items():
        path = ROOT / f"perception-occlusion-pairs-01/{split}.json"
        if sha(path) != pin:
            raise ValueError("pair corpus changed")
        pairs[split] = json.loads(path.read_text())["pairs"]
    if {p["video_id"] for p in pairs["train"]} & {p["video_id"] for p in pairs["valid"]}:
        raise ValueError("video overlap")
    train, targets, train_probs = load_features(args.train, pairs["train"])
    valid, valid_targets, valid_probs = load_features(args.valid, pairs["valid"])

    class InertMarker:
        pass

    with torch.serialization.safe_globals(
        [(InertMarker, "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint")]
    ):
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    initial = state["itm_score.fc.weight"][1] - state["itm_score.fc.weight"][0]
    del state
    if initial.shape != (192,):
        raise ValueError("ITM initial shape mismatch")
    for features, expected in ((train, train_probs), (valid, valid_probs)):
        for mode, tensor in features.items():
            actual = (tensor @ initial).softmax(-1).numpy()
            if not np.allclose(actual, expected[mode].reshape(-1, 3), atol=1e-6, rtol=0):
                raise ValueError("initial ITM probability parity failed")
    args.output.mkdir()
    write(
        args.output / "config.json",
        dict(
            seed=17,
            epochs=50,
            lr=0.001,
            weight_decay=0.01,
            selector="minimum validation NLL including epoch0; earliest ties",
            feature_hashes={
                name: sha(path / "predictions.jsonl")
                for name, path in (("train", args.train), ("valid", args.valid))
            },
            feature_protocol_hashes={
                name: sha(path / "protocol.json")
                for name, path in (("train", args.train), ("valid", args.valid))
            },
            pair_hashes=PINS,
            checkpoint_sha256=CHECKPOINT,
        ),
    )
    torch.set_num_threads(4)
    results = {
        mode: fit(train[mode], valid[mode], targets, valid_targets, initial, args.output / mode)
        for mode in train
    }
    write(args.output / "result.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
