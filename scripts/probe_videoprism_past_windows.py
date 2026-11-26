"""Offline fixed past-window extraction on one pinned training video."""

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
from tiny_omni_decision.past_video_windows import past_window_indices


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    source = root / "videoprism-base-r1/source"
    checkpoint = root / "videoprism-base-f16r288.npz"
    r2 = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    media = r2 / "vidimu-video-decode-smoke-01/input.mp4"
    digest = "b6d8fd31bc9459e3f822c4df20a844232db491d260cd82d5e0585af3df3de0d9"
    if sha256_file(media) != digest or sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("pinned input mismatch")
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                               text=True).strip() != SOURCE_COMMIT:
        raise ValueError("upstream changed")
    sys.path.insert(0, str(source))
    import av
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    if any(d.platform != "cpu" for d in jax.devices()):
        raise ValueError("CPU-only probe")
    out = r2 / "videoprism-past-window-probe-01"
    out.mkdir(exist_ok=False)
    config = dict(media_sha256=digest, checkpoint_sha256=CHECKPOINT_SHA256,
                  upstream_commit=SOURCE_COMMIT, window_ms=4000, stride_ms=4000,
                  ends_ms=list(range(4000, 36001, 4000)), frames=8,
                  script_sha256=sha256_file(Path(__file__)),
                  sampler_sha256=sha256_file(Path("src/tiny_omni_decision/past_video_windows.py")),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        text=True).strip(),
                  scope="offline one training clip; no quality or online latency claim")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    decode_start = time.perf_counter()
    with av.open(str(media)) as container:
        times = [float(frame.time) * 1000 for frame in container.decode(container.streams.video[0])]
    windows = [past_window_indices(times, start_ms=end - 4000, end_ms=end)
               for end in config["ends_ms"]]
    needed = set(i for indices in windows for i in indices)
    decoded = {}
    with av.open(str(media)) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index in needed:
                if frame.time is None or float(frame.time) * 1000 != times[index]:
                    raise ValueError("timestamps changed between decode passes")
                decoded[index] = frame.to_ndarray(format="rgb24")
    if set(decoded) != needed:
        raise ValueError("incomplete frame extraction")
    decode_seconds = time.perf_counter() - decode_start
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint))
    encode = jax.jit(lambda x: model.apply(variables, x, train=False))
    values, details = [], []
    for end, indices in zip(config["ends_ms"], windows, strict=True):
        # Prefix-only planning must choose the same input without later timestamps.
        assert indices == past_window_indices([t for t in times if t <= end],
                                               start_ms=end - 4000, end_ms=end)
        start = time.perf_counter()
        x = _preprocess(np.stack([decoded[i] for i in indices]), tf)
        result = encode(jnp.asarray(x[None]))
        tokens = np.asarray(result[0] if isinstance(result, tuple) else result)
        if tokens.shape != (1, 2048, 768) or not np.isfinite(tokens).all():
            raise ValueError("unexpected/nonfinite features")
        values.append(tokens.reshape(8, 256, 768).mean(1))
        details.append(dict(start_ms=end - 4000, end_ms=end, indices=indices,
                            selected_timestamps_ms=[times[i] for i in indices],
                            preprocess_forward_seconds=time.perf_counter() - start))
    np.savez_compressed(out / "features.npz", values=np.stack(values))
    report = dict(windows=details, decoded_frame_count=len(times),
                  two_pass_decode_seconds=decode_seconds,
                  features_sha256=sha256_file(out / "features.npz"),
                  no_selected_future_frames=True, exact_prefix_selection=True,
                  peak_working_set_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
                  devices=[str(d) for d in jax.devices()], sealed_audit_read=False,
                  availability="not per-frame; after window evidence and actual encode completion")
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
