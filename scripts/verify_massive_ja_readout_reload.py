"""Verify saved MASSIVE 60-intent readout probabilities from frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_reload(
    features_path: Path, checkpoint_path: Path, predictions_path: Path, hidden_size: int
) -> dict[str, Any]:
    torch.set_num_threads(4)
    features = torch.load(features_path, map_location="cpu", weights_only=True)
    cached = features["validation_features"].float()
    targets = features["validation_targets"].long()
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    if cached.ndim != 2 or cached.shape[1] != hidden_size:
        raise ValueError("cached validation features have unexpected dimensions")
    if len(rows) != len(targets) or not rows:
        raise ValueError("prediction rows do not match cached validation targets")
    if [int(row["target_index"]) for row in rows] != targets.tolist():
        raise ValueError("prediction target order differs from cached validation order")
    head = nn.Sequential(
        nn.Linear(hidden_size, 128),
        nn.Tanh(),
        nn.Dropout(0.1),
        nn.Linear(128, 60),
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    head.load_state_dict(state)
    head.eval()
    with torch.inference_mode():
        actual = head(cached).float().softmax(-1)
    expected = torch.tensor([row["probabilities"] for row in rows], dtype=torch.float32)
    if expected.shape != actual.shape or not torch.isfinite(expected).all():
        raise ValueError("saved predictions have invalid shape or non-finite values")
    delta = (actual - expected).abs()
    matches = actual.argmax(-1).eq(expected.argmax(-1))
    max_delta = float(delta.max())
    if not torch.allclose(actual, expected, atol=1e-7, rtol=0):
        raise ValueError(f"checkpoint reload probability delta too large: {max_delta}")
    return {
        "status": "pass",
        "rows": len(rows),
        "matching_argmax_rows": int(matches.sum()),
        "max_probability_abs_delta": max_delta,
        "tolerance_atol": 1e-7,
        "tolerance_rtol": 0,
        "feature_cache_sha256": sha256_file(features_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "predictions_sha256": sha256_file(predictions_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    receipt = verify_reload(
        args.features, args.checkpoint, args.predictions, args.hidden_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
