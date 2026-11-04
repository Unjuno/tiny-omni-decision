"""Pinned CPU inference probe: change later frames, inspect earlier tokens."""

import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    _preprocess,
    sha256_file,
)
from tiny_omni_decision.video_frame_timestamps import decode_endpoint_frames


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    source = root / "videoprism-base-r1/source"
    checkpoint = root / "videoprism-base-f16r288.npz"
    r2 = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    media = r2 / "vidimu-video-decode-smoke-01/input.mp4"
    expected_media = "b6d8fd31bc9459e3f822c4df20a844232db491d260cd82d5e0585af3df3de0d9"
    if sha256_file(checkpoint) != CHECKPOINT_SHA256 or sha256_file(media) != expected_media:
        raise ValueError("pinned checkpoint/media hash mismatch")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError("pinned source mismatch")
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("CPU-only probe")
    out = r2 / "videoprism-future-context-probe-01"
    out.mkdir(exist_ok=False)
    config = dict(model="videoprism_public_v1_base", checkpoint_sha256=CHECKPOINT_SHA256,
                  upstream_commit=revision, media_sha256=expected_media,
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        text=True).strip(),
                  script_sha256=sha256_file(Path(__file__)),
                  intervention="invert normalized frames4..7; preserve frames0..3 exactly",
                  inference_order=["original", "repeat_original", "changed_future"],
                  scope="one known training clip, no training or quality benchmark")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    model = vp.get_model(config["model"])
    variables = vp.load_pretrained_weights(config["model"], checkpoint_path=str(checkpoint))
    frames, timestamps = decode_endpoint_frames(media)
    original = _preprocess(frames, tf)
    altered = original.copy()
    altered[4:] = 1 - altered[4:]
    np.testing.assert_array_equal(original[:4], altered[:4])
    encode = jax.jit(lambda x: model.apply(variables, x, train=False))
    features, seconds = {}, {}
    for name, x in (("original", original), ("repeat_original", original),
                    ("changed_future", altered)):
        start = time.perf_counter()
        result = encode(jnp.asarray(x[None]))
        tokens = np.asarray(result[0] if isinstance(result, tuple) else result)
        if tokens.shape != (1, 2048, 768) or not np.isfinite(tokens).all():
            raise ValueError("unexpected/nonfinite output")
        features[name] = tokens.reshape(8, 256, 768)
        seconds[name] = time.perf_counter() - start
    delta = features["changed_future"] - features["original"]
    pooled = {name: value.mean(1) for name, value in features.items()}
    np.savez_compressed(out / "pooled-features.npz", **pooled)
    report = dict(timestamps=timestamps, forward_seconds=seconds,
                  repeat_max_abs=float(np.abs(features["original"]
                                              - features["repeat_original"]).max()),
                  per_time_token_max_abs=np.abs(delta).max(axis=(1, 2)).tolist(),
                  per_time_token_rms=np.sqrt(np.mean(delta ** 2, axis=(1, 2))).tolist(),
                  per_time_pooled_max_abs=np.abs(pooled["changed_future"]
                                                - pooled["original"]).max(1).tolist(),
                  unchanged_prefix_frames=4,
                  parameters=sum(x.size for x in jax.tree_util.tree_leaves(variables)),
                  devices=[str(x) for x in jax.devices()],
                  peak_working_set_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
                  pooled_sha256=sha256_file(out / "pooled-features.npz"),
                  versions=dict(jax=jax.__version__, numpy=np.__version__,
                                tensorflow=tf.__version__),
                  sealed_audit_read=False)
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
