"""CPU VideoPrism worker for eight fixed selector images; no learning."""

import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.extract_frozen_videoprism_clevr4_features import (
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    _preprocess_rgb_image,
    sha256_file,
)
from tiny_omni_decision.videoprism_spatial_features import spatial_cells


def main():
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    r2 = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = r1 / "videoprism-base-r1/source"
    weights = r1 / "videoprism-base-f16r288.npz"
    cache = r2 / "videoprism-spatial-image-cache-01"
    media = r2 / "clevr4-sample-seed17-1024-512/val/images"
    if sha256_file(weights) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint changed")
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                               text=True).strip() != SOURCE_COMMIT:
        raise ValueError("upstream changed")
    if sha256_file(cache / "records.json") != (
            "32944fc822c45a7a93ee06904c9a97acaa5bf6364c33c40525f535c59da55b41"):
        raise ValueError("record order changed")
    rows = json.loads((cache / "records.json").read_text())["val"][:8]
    out = r2 / "shared-global-raw-image-replay-01"
    out.mkdir(exist_ok=False)
    config = dict(rows=rows, checkpoint_sha256=CHECKPOINT_SHA256, source=SOURCE_COMMIT,
                  script_sha256=sha256_file(Path(__file__)), batch=8, frames=1,
                  atol=1e-5, rtol=1e-5, profile="CPU JAX, original fixed 288 center crop")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    assert all(d.platform == "cpu" for d in jax.devices())
    start = time.perf_counter()
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(weights))
    load_seconds = time.perf_counter() - start
    encode = jax.jit(lambda x: model.apply(variables, x, train=False,
        return_intermediate={"spatial_features"})[1]["spatial_features"])
    frames = []
    start = time.perf_counter()
    for row in rows:
        path = media / (row["id"] + ".png")
        assert sha256_file(path) == row["sha256"]
        image = tf.io.decode_png(tf.io.read_file(str(path)), channels=3)
        frames.append(_preprocess_rgb_image(image, tf)[:1])
    frames = jnp.asarray(np.stack(frames))
    preprocessing_seconds = time.perf_counter() - start
    timings, outputs = [], []
    for _ in range(2):
        start = time.perf_counter()
        outputs.append(spatial_cells(np.asarray(encode(frames))))
        timings.append(time.perf_counter() - start)
    reference = np.load(cache / "val-cells.npy", allow_pickle=False)[:8]
    np.save(out / "cells.npy", outputs[-1])
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outputs[-1], reference, rtol=1e-5, atol=1e-5)
    result = dict(status="PASS_RAW_FEATURES", load_seconds=load_seconds,
        preprocessing_seconds=preprocessing_seconds, first_and_warm_batch_seconds=timings,
        max_cache_delta=float(np.abs(outputs[-1]-reference).max()),
        features_sha256=sha256_file(out / "cells.npy"),
        peak_working_set_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
        parameters=sum(x.size for x in jax.tree_util.tree_leaves(variables)))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
