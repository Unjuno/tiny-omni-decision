"""Train-only mechanics smoke of the fixed 64D correspondence projection."""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_correspondence_validation import sha, write
from tiny_omni_decision.decision_math import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)


class CorrespondenceProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = nn.Linear(384, 64, bias=False)
        self.video = nn.Linear(768, 64, bias=False)

    def forward(self, audio, video):
        a = F.normalize(self.audio(audio), dim=-1)
        v = F.normalize(self.video(video), dim=-1)
        return (a[:, None, :] * v).sum(-1) / 0.1


def load_features(folder, dimension, hashes):
    report_path = folder / "result.json"
    hashes[str(report_path)] = sha(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "PASS" or len(report["features"]) != 8:
        raise ValueError("eight verified features required")
    result = {}
    for index, row in enumerate(report["features"]):
        path = folder / f"feature-{index:03}.npy"
        hashes[str(path)] = sha(path)
        if hashes[str(path)] != row["sha256"] or row["id"] in result:
            raise ValueError("feature changed or duplicated")
        vector = np.load(path, allow_pickle=False)
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError("invalid feature")
        result[row["id"]] = vector
    return result, report["windows_sha256"]


def measure(model, audio, video, targets):
    with torch.no_grad():
        probabilities = model(audio, video).softmax(-1).tolist()
    y = targets.tolist()
    prediction = [int(np.argmax(p)) for p in probabilities]
    return dict(
        accuracy=sum(a == b for a, b in zip(prediction, y, strict=True)) / len(y),
        nll=negative_log_likelihood(probabilities, y),
        brier=brier_score(probabilities, y),
        ece=expected_calibration_error(probabilities, y),
        predictions=prediction,
        targets=y,
        probabilities=probabilities,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    hashes = {}
    route_path = root / "perception-correspondence-plan-01/train-routing.json"
    hashes[str(route_path)] = sha(route_path)
    if (
        hashes[str(route_path)]
        != "560f3bc7235c842746038e6e5a886471f4dc08cc515d9eba4d12968e608f393b"
    ):
        raise ValueError("routing changed")
    rows = json.loads(route_path.read_text(encoding="utf-8"))[:8]
    audio, aw = load_features(root / "perception-correspondence-audio-01", 384, hashes)
    video, vw = load_features(root / "perception-correspondence-video-01", 768, hashes)
    ids = [r["audio_video_id"] for r in rows]
    if set(ids) != set(audio) or set(ids) != set(video) or aw != vw:
        raise ValueError("feature routing or window mismatch")
    if any(r["split"] != "train" for r in rows):
        raise ValueError("smoke is train only")
    a = torch.tensor(np.stack([audio[i] for i in ids]))
    v = torch.tensor(np.stack([[video[i] for i in r["candidate_video_ids"]] for r in rows]))
    y = torch.tensor([r["target_index"] for r in rows])
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    model = CorrespondenceProjection()
    config = dict(
        seed=17,
        epochs=100,
        learning_rate=0.001,
        weight_decay=0.01,
        temperature=0.1,
        projection_dimension=64,
        batch_size=4,
        selector=None,
        scope="TRAIN_ONLY_MECHANICS_SMOKE_NOT_QUALITY",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=sha(Path(__file__)),
        hashes=hashes,
        python=platform.python_version(),
        torch=torch.__version__,
        backend="cpu",
    )
    write(args.output / "config.json", config)
    before = measure(model, a, v, y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    history = []
    start = time.perf_counter()
    for epoch in range(100):
        for group in torch.randperm(2).tolist():
            index = slice(group * 4, group * 4 + 4)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(a[index], v[index]), y[index])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite gradient")
            optimizer.step()
        metrics = measure(model, a, v, y)
        history.append(
            dict(epoch=epoch + 1, **{k: metrics[k] for k in ("accuracy", "nll", "brier", "ece")})
        )
    seconds = time.perf_counter() - start
    checkpoint = args.output / "final.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = CorrespondenceProjection()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        logits = model(a, v)
        torch.testing.assert_close(restored(a, v), logits, rtol=0, atol=0)
        torch.testing.assert_close(model(a, v.flip(1)).flip(1), logits, rtol=1e-6, atol=1e-6)
    controls = dict(
        untrained=before,
        trained=measure(restored, a, v, y),
        audio_zero=measure(restored, torch.zeros_like(a), v, y),
        video_zero=measure(restored, a, torch.zeros_like(v), y),
        shifted_audio=measure(restored, a.reshape(2, 4, 384).roll(1, 1).reshape(8, 384), v, y),
        reversed_options=measure(restored, a, v.flip(1), 3 - y),
    )
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("input artifact changed")
    write(args.output / "history.json", history)
    write(args.output / "predictions.json", dict(records=rows, controls=controls))
    result = dict(
        status="PASS_MECHANICS_ONLY",
        unique_train_recordings=8,
        quartets=2,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        updates=200,
        repeated_presentations=800 - 8,
        seconds=seconds,
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=sha(checkpoint),
        reload_max_delta=0,
        controls={
            name: {k: result[k] for k in ("accuracy", "nll", "brier", "ece")}
            for name, result in controls.items()
        },
        validation_examples=0,
        quality_claim=False,
        encoder_weights_loaded=False,
    )
    write(args.output / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
