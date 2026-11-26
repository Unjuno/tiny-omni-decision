"""Run one train-only, two-stream sub-SEVIR input through pinned VideoPrism."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

MODEL_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
FRAME_SIZE = 288
FRAME_COUNT = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_only_percentiles(group: Any, train_event_ids: set[str]) -> list[list[float]]:
    ids = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in group["id"][:]
    ]
    selected = np.fromiter((event_id in train_event_ids for event_id in ids), dtype=bool)
    if not selected.any():
        raise ValueError("no source rows matched frozen training event IDs")
    channel_samples: list[list[np.ndarray]] = [[], [], [], []]
    source = group["features"]
    for start in range(0, len(ids), source.chunks[0]):
        stop = min(start + source.chunks[0], len(ids))
        local_mask = selected[start:stop]
        if not local_mask.any():
            continue
        block = np.asarray(source[start:stop], dtype=np.float32).reshape(stop - start, 48 * 48, 4)
        sample = block[local_mask, ::97, :]
        if not np.isfinite(sample).all():
            raise ValueError("non-finite training feature values")
        for channel in range(4):
            channel_samples[channel].append(sample[..., channel].reshape(-1))
    stats = []
    for values in channel_samples:
        merged = np.concatenate(values)
        low, high = np.percentile(merged, [1.0, 99.0]).astype(float).tolist()
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("invalid train-only percentile range")
        stats.append([low, high])
    return stats


def _load_model(checkpoint: Path, source_root: Path) -> tuple[Any, Any]:
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != MODEL_SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {commit}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint hash mismatch")
    sys.path.insert(0, str(source_root))
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    return model, variables


def _normalized_rgb(
    frames: np.ndarray,
    indices: tuple[int, int, int],
    stats: list[list[float]],
) -> np.ndarray:
    channels = []
    for channel in indices:
        low, high = stats[channel]
        value = np.clip((frames[..., channel] - low) / (high - low), 0.0, 1.0)
        channels.append(value)
    return np.stack(channels, axis=-1).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_groups = set(manifest["split_group_ids"]["train"])
    validation_groups = set(manifest["split_group_ids"]["validation"])
    if train_groups & validation_groups:
        raise ValueError("frozen manifest contains event overlap")
    windows = [window for window in manifest["windows"] if window["split"] == "train"]
    selected = []
    seen = set()
    for window in windows:
        if window["event_id"] not in seen:
            selected.append(window)
            seen.add(window["event_id"])
        if len(selected) == args.examples:
            break
    if len(selected) != args.examples:
        raise ValueError("insufficient unique train events for requested smoke size")

    group = zarr.open_group(str(args.train_zarr), mode="r")
    ids = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in group["id"][:]
    ]
    stats = train_only_percentiles(group, train_groups)
    if group["n_channel"][:].tolist() != ["ir069", "ir107", "vil", "vis"]:
        raise ValueError("unexpected source channel order")
    referenced_ids = [
        ids[index] for window in selected for index in window["input_indices"]
    ]
    if any(event_id not in train_groups for event_id in referenced_ids):
        raise ValueError("smoke sample refers outside frozen train events")

    raw = np.stack([
        np.asarray(group["features"][list(window["input_indices"])], dtype=np.float32)
        for window in selected
    ])
    satellite = _normalized_rgb(raw, (0, 1, 3), stats)
    radar = _normalized_rgb(raw, (2, 2, 2), stats)

    import jax
    import psutil

    if jax.default_backend() != "cpu":
        raise ValueError(f"expected the audited CPU backend, got {jax.default_backend()}")
    model, variables = _load_model(args.checkpoint, args.source_root)
    encode = jax.jit(lambda value: model.apply(variables, value, train=False))
    process = psutil.Process()
    rss_start = process.memory_info().rss
    outputs: dict[str, dict[str, object]] = {}
    timings = {}
    for name, stream in (("satellite", satellite), ("radar", radar)):
        resized = jax.image.resize(
            jax.numpy.asarray(stream),
            (len(selected), FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3),
            method="linear",
        )
        started = time.perf_counter()
        result = encode(resized)
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready()
                if hasattr(value, "block_until_ready")
                else value
            ),
            result,
        )
        timings[name] = time.perf_counter() - started
        tokens = result[0] if isinstance(result, tuple) else result
        token_array = np.asarray(tokens)
        if token_array.shape != (len(selected), 2048, 768) or not np.isfinite(token_array).all():
            raise ValueError(f"invalid {name} VideoPrism output: {token_array.shape}")
        per_frame = token_array.reshape(len(selected), FRAME_COUNT, 16 * 16, 768).mean(axis=2)
        outputs[name] = {
            "token_shape": list(token_array.shape),
            "per_frame_shape": list(per_frame.shape),
            "feature_sha256": hashlib.sha256(per_frame.astype("<f4").tobytes()).hexdigest(),
            "finite": bool(np.isfinite(per_frame).all()),
        }

    result = {
        "schema": "tiny-omni-decision/subsevir-videoprism-smoke-v1",
        "status": "pass",
        "scope": "train-only one-example input/encoder smoke; no fitting or validation access",
        "manifest_sha256": sha256_file(args.manifest),
        "train_zmetadata_sha256": sha256_file(args.train_zarr / ".zmetadata"),
        "model_revision": MODEL_REVISION,
        "source_commit": MODEL_SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "backend": jax.default_backend(),
        "examples": len(selected),
        "sample_ids": [window["sample_id"] for window in selected],
        "normalization": {
            "method": (
                "train-event-only deterministic every-97th-pixel sample; "
                "per-channel p01/p99 clip to [0,1]"
            ),
            "channels": ["ir069", "ir107", "vil", "vis"],
            "percentile_ranges": stats,
            "satellite_rgb_channels": ["ir069", "ir107", "vis"],
            "radar_rgb_channels": ["vil", "vil", "vil"],
        },
        "output": outputs,
        "seconds": timings,
        "rss_start_bytes": rss_start,
        "rss_end_bytes": process.memory_info().rss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-zarr", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=1)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
