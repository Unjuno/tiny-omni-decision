"""Fixed train-only slot readout; never opens validation features or selects best."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_correspondence_validation import sha, write
from scripts.slot_feature_sampling import sample_indices
from scripts.train_trajectory_readout import batch, predict, summarize
from tiny_omni_decision.trajectory_readout import TrajectoryReadout

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PAIR_SHA = "8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7"
MEDIA_SHA = "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3"


def validate_feature(row, value, media):
    if row["split"] != "train" or row["media_sha256"] != media["sha256"]:
        raise ValueError("feature media binding mismatch")
    if value.shape != (224, 129) or not np.isfinite(value).all():
        raise ValueError("invalid feature shape/value")
    expected = sample_indices(row["frame_count"])
    if row["frame_indices"] != expected:
        raise ValueError("frame sampling changed")
    times = np.repeat(np.asarray(expected, dtype=np.float32) / (row["frame_count"] - 1), 7)
    if not np.array_equal(value[:, -1], times):
        raise ValueError("time features changed")


def load_train(folder):
    ledger = json.loads((folder / "result.json").read_text())
    if ledger["status"] != "PASS_TRAIN_ONLY_SLOT_EXTRACTION" or not ledger["weights_unchanged"]:
        raise ValueError("complete frozen extraction required")
    protocol_path = folder / "protocol.json"
    if sha(protocol_path) != ledger["protocol_sha256"]:
        raise ValueError("protocol changed")
    protocol = json.loads(protocol_path.read_text())
    if protocol["manifest_sha256"] != MEDIA_SHA or protocol["pair_sha256"] != PAIR_SHA:
        raise ValueError("source corpus changed")
    pair_path = ROOT / "perception-occlusion-pairs-01/train.json"
    manifest_path = ROOT / "perception-occlusion-train-media-01/manifest.json"
    if sha(pair_path) != PAIR_SHA or sha(manifest_path) != MEDIA_SHA:
        raise ValueError("local corpus changed")
    pairs = json.loads(pair_path.read_text())["pairs"]
    media = {r["video_id"]: r for r in json.loads(manifest_path.read_text())["clips"]}
    if len(pairs) != 99 or len(media) != 99 or len({r["sha256"] for r in media.values()}) != 99:
        raise ValueError("incomplete/duplicate corpus")
    if any([q["condition"] for q in p["questions"]] != ["initial", "final"] for p in pairs):
        raise ValueError("condition order changed")
    plan = {r["video_id"]: r for r in protocol["plan"]}
    if set(plan) != set(media):
        raise ValueError("feature plan incomplete")
    bank = {}
    hashes = {
        str(folder / "result.json"): sha(folder / "result.json"),
        str(protocol_path): sha(protocol_path),
    }
    for row in ledger["features"]:
        path = Path(row["path"])
        key = row["video_id"]
        if key in bank or key not in media or sha(path) != row["sha256"]:
            raise ValueError("duplicate/missing/changed feature")
        value = np.load(path, allow_pickle=False)
        if row["frame_count"] != plan[key]["frame_count"]:
            raise ValueError("frame count differs from frozen plan")
        validate_feature(row, value, media[key])
        bank[key] = torch.from_numpy(value)
        hashes[str(path)] = sha(path)
    if set(bank) != {p["video_id"] for p in pairs}:
        raise ValueError("all 99 clips required")
    text_path = ROOT / "perception-occlusion-text-features-01/result.json"
    text = json.loads(text_path.read_text())
    if text["status"] != "PASS_FROZEN_TEXT_FEATURES" or text["pair_hashes"]["train"] != PAIR_SHA:
        raise ValueError("text provenance changed")
    row = text["features"]["train"]
    path = Path(row["path"])
    if sha(path) != row["sha256"]:
        raise ValueError("text features changed")
    expected = [
        dict(video_id=p["video_id"], condition=q["condition"], target=q["target"])
        for p in pairs
        for q in p["questions"]
    ]
    x = np.load(path, allow_pickle=False)
    if row["rows"] != expected or x.shape != (198, 3, 1536) or not np.isfinite(x).all():
        raise ValueError("text routing/shape/value mismatch")
    hashes[str(text_path)] = sha(text_path)
    hashes[str(path)] = sha(path)
    return (
        bank,
        dict(
            pairs=pairs,
            text=torch.from_numpy(x),
            targets=torch.tensor([r["target"] for r in expected]),
        ),
        hashes,
    )


def summary(probabilities, targets):
    out = summarize(probabilities, targets)
    choices = probabilities.argmax(-1).reshape(-1, 2)
    out["condition_switches"] = int((choices[:, 0] != choices[:, 1]).sum())
    return out


def fit(bank, data, output, hashes):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    generator = torch.Generator().manual_seed(17)
    model = TrajectoryReadout(input_dim=129)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    write(
        output / "config.json",
        dict(
            seed=17,
            epochs=50,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_pairs=8,
            input_dim=129,
            hidden=64,
            selector="none; fixed final train diagnostic",
            device="cpu",
            parameters=sum(p.numel() for p in model.parameters()),
            input_hashes=hashes,
            script_sha256=sha(Path(__file__)),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            python=platform.python_version(),
            torch=torch.__version__,
        ),
    )
    targets = data["targets"].numpy()
    history = []
    updates = 0
    started = time.perf_counter()
    write(
        output / "examples.json",
        [
            dict(
                video_id=p["video_id"],
                condition=condition,
                target=int(targets[2 * i + j]),
                options=p.get("options"),
            )
            for i, p in enumerate(data["pairs"])
            for j, condition in enumerate(("initial", "final"))
        ],
    )
    initial = predict(model, bank, data)
    torch.save(model.state_dict(), output / "untrained.pt")
    write(output / "untrained-predictions.json", initial.tolist())
    history.append(dict(epoch=0, updates=0, metrics=summary(initial, targets)))
    for epoch in range(1, 51):
        model.train()
        order = torch.randperm(len(data["pairs"]), generator=generator).tolist()
        for offset in range(0, len(order), 8):
            v, text, mask, y = batch(bank, data, order[offset : offset + 8])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(v, text, mask), y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("missing/nonfinite gradient")
            optimizer.step()
            updates += 1
        probabilities = predict(model, bank, data)
        history.append(dict(epoch=epoch, updates=updates, metrics=summary(probabilities, targets)))
        write(output / "history.json", history)
    torch.save(model.state_dict(), output / "final.pt")
    controls = {mode: predict(model, bank, data, mode) for mode in ["normal", "mismatch", "zero"]}
    write(output / "final-predictions.json", {k: v.tolist() for k, v in controls.items()})
    reloaded = TrajectoryReadout(input_dim=129)
    reloaded.load_state_dict(
        torch.load(output / "final.pt", map_location="cpu", weights_only=True), strict=True
    )
    if not np.array_equal(predict(reloaded, bank, data), controls["normal"]):
        raise ValueError("reload mismatch")
    result = dict(
        status="COMPLETE_TRAIN_ONLY_NOT_GENERALIZATION",
        seconds=time.perf_counter() - started,
        updates=updates,
        unique_questions=len(targets),
        unique_videos=len(data["pairs"]),
        example_presentations=len(targets) * 50,
        reload_exact=True,
        controls={k: summary(v, targets) for k, v in controls.items()},
        artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "result.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank, data, hashes = load_train(args.features)
    fit(bank, data, args.output, hashes)


if __name__ == "__main__":
    main()
