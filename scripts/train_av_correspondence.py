"""Fixed development comparison; frozen features, no test/audit inputs."""

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
from scripts.smoke_av_correspondence import CorrespondenceProjection, measure

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
ROUTES = {
    "train": (
        "perception-correspondence-plan-01/train-routing.json",
        256,
        "560f3bc7235c842746038e6e5a886471f4dc08cc515d9eba4d12968e608f393b",
    ),
    "valid": (
        "perception-correspondence-valid-audit-01/valid-routing.json",
        200,
        "03d389b0eda847b4d16e0205a7c285ad78ebecfb6405fde1c85c0ddfd569d395",
    ),
}


def read_checked(path, hashes, expected=None):
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"changed input: {path}")
    hashes[str(path)] = digest
    return json.loads(path.read_text(encoding="utf-8"))


def load_features(folder, dimension, window_hash, ids, hashes):
    report = read_checked(folder / "result.json", hashes)
    if report["status"] != "PASS" or report["windows_sha256"] != window_hash:
        raise ValueError("feature provenance mismatch")
    features = {}
    for index, row in enumerate(report["features"]):
        path = folder / f"feature-{index:03}.npy"
        digest = sha(path)
        hashes[str(path)] = digest
        if digest != row["sha256"] or row["id"] in features:
            raise ValueError("changed/duplicate feature")
        value = np.load(path, allow_pickle=False)
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError("invalid feature")
        features[row["id"]] = value
    if set(features) != ids:
        raise ValueError("incomplete feature IDs")
    return features


def scalar_metrics(value):
    return {k: value[k] for k in ("accuracy", "nll", "brier", "ece")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hashes = {}
    windows = read_checked(args.windows, hashes)
    if windows["status"] != "PASS" or len(windows["windows"]) != 456:
        raise ValueError("complete development windows required")
    rows = {}
    for split, (path, count, digest) in ROUTES.items():
        rows[split] = read_checked(ROOT / path, hashes, digest)
        if len(rows[split]) != count or any(r["split"] != split for r in rows[split]):
            raise ValueError("routing count/split mismatch")
    ids = {r["audio_video_id"] for split in rows.values() for r in split}
    if len(ids) != 456 or {r["id"] for r in windows["windows"]} != ids:
        raise ValueError("window/routing identity mismatch")
    for split, records in rows.items():
        if {r["audio_video_id"] for r in records} != {
            r["id"] for r in windows["windows"] if r["split"] == split
        }:
            raise ValueError("window split mismatch")
    audio = load_features(args.audio, 384, hashes[str(args.windows)], ids, hashes)
    video = load_features(args.video, 768, hashes[str(args.windows)], ids, hashes)
    tensors = {}
    for split, records in rows.items():
        tensors[split] = (
            torch.tensor(np.stack([audio[r["audio_video_id"]] for r in records])),
            torch.tensor(np.stack([[video[i] for i in r["candidate_video_ids"]] for r in records])),
            torch.tensor([r["target_index"] for r in records]),
        )
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    model = CorrespondenceProjection()
    write(
        args.output / "config.json",
        dict(
            seed=17,
            epochs=100,
            learning_rate=0.001,
            weight_decay=0.01,
            batch_size=4,
            projection_dimension=64,
            temperature=0.1,
            selector="earliest minimum validation NLL",
            backend="cpu",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            hashes=hashes,
            script_sha256=sha(Path(__file__)),
            projection_source_sha256=sha(Path("scripts/smoke_av_correspondence.py")),
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
        ),
    )
    before = measure(model, *tensors["valid"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best_nll, best_epoch = float("inf"), None
    history = []
    start = time.perf_counter()
    for epoch in range(1, 101):
        a, v, y = tensors["train"]
        for group in torch.randperm(64).tolist():
            index = slice(group * 4, group * 4 + 4)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(a[index], v[index]), y[index])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite gradient")
            optimizer.step()
        metrics = {s: measure(model, *t) for s, t in tensors.items()}
        if metrics["valid"]["nll"] < best_nll:
            best_nll, best_epoch = metrics["valid"]["nll"], epoch
            torch.save(model.state_dict(), args.output / "best.pt")
        history.append(
            dict(
                epoch=epoch,
                updates=epoch * 64,
                presentations=epoch * 256,
                best_epoch=best_epoch,
                seconds=time.perf_counter() - start,
                **{s: scalar_metrics(m) for s, m in metrics.items()},
            )
        )
        write(args.output / "history.json", history)
    seconds = time.perf_counter() - start
    torch.save(model.state_dict(), args.output / "final.pt")
    results = {"untrained": before}
    for name in ("final", "best"):
        restored = CorrespondenceProjection()
        restored.load_state_dict(torch.load(args.output / f"{name}.pt", weights_only=True))
        measured = measure(restored, *tensors["valid"])
        expected = history[-1 if name == "final" else best_epoch - 1]["valid"]
        if scalar_metrics(measured) != expected:
            raise ValueError("reload metrics changed")
        results[name] = measured
    a, v, y = tensors["valid"]
    results.update(
        audio_zero=measure(restored, torch.zeros_like(a), v, y),
        video_zero=measure(restored, a, torch.zeros_like(v), y),
        shifted_audio=measure(restored, a.reshape(50, 4, 384).roll(1, 1).reshape(200, 384), v, y),
        reversed_options=measure(restored, a, v.flip(1), 3 - y),
    )
    with torch.no_grad():
        torch.testing.assert_close(
            restored(a, v.flip(1)).flip(1), restored(a, v), atol=1e-6, rtol=1e-6
        )
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("input changed during fitting")
    write(args.output / "predictions.json", dict(records=rows["valid"], conditions=results))
    write(
        args.output / "result.json",
        dict(
            status="PASS_DEVELOPMENT_COMPARISON",
            best_epoch=best_epoch,
            final_epoch=100,
            unique_train=256,
            unique_valid=200,
            presentations=25600,
            selected_presentations=best_epoch * 256,
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            seconds=seconds,
            encoder_weights_loaded=False,
            general_quality_claim=False,
            checkpoints={
                n: dict(
                    sha256=sha(args.output / f"{n}.pt"),
                    bytes=(args.output / f"{n}.pt").stat().st_size,
                )
                for n in ("best", "final")
            },
            conditions={n: scalar_metrics(m) for n, m in results.items()},
        ),
    )


if __name__ == "__main__":
    main()
