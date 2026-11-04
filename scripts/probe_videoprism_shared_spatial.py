"""Pinned training-image probe of shared VideoPrism spatial intermediate features."""

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


def main():
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    r2 = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = r1 / "videoprism-base-r1/source"
    weights = r1 / "videoprism-base-f16r288.npz"
    corpus = r2 / "clevr4-full-coverage-seed17"
    manifest = corpus / "train-image-manifest.jsonl"
    expected = "2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010"
    if sha256_file(manifest) != expected or sha256_file(weights) != CHECKPOINT_SHA256:
        raise ValueError("input hash mismatch")
    with manifest.open() as stream:
        row = json.loads(next(stream))
    if row["split"] != "train":
        raise ValueError("training image only")
    image = corpus / "train/images" / (row["id"] + ".png")
    if sha256_file(image) != row["sha256"]:
        raise ValueError("image changed")
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                               text=True).strip() != SOURCE_COMMIT:
        raise ValueError("upstream source mismatch")
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    if any(d.platform != "cpu" for d in jax.devices()):
        raise ValueError("CPU only")
    out = r2 / "videoprism-shared-spatial-smoke-02"
    out.mkdir(exist_ok=False)
    config = dict(image=row, manifest_sha256=expected, checkpoint_sha256=CHECKPOINT_SHA256,
                  upstream_commit=SOURCE_COMMIT, script_sha256=sha256_file(Path(__file__)),
                  numerical_tolerance=dict(rtol=1e-5, atol=1e-5),
                  scope="one training image; single versus eight repeated frames; no fit")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(weights))
    spatial = jax.jit(lambda x: model.apply(
        variables, x, train=False, return_intermediate={"spatial_features"})[1]["spatial_features"])
    frames = _preprocess_rgb_image(tf.io.decode_png(tf.io.read_file(str(image)), channels=3), tf)
    timings, features = {}, {}
    for count in (1, 8):
        for repetition in (0, 1):
            start = time.perf_counter()
            value = np.asarray(spatial(jnp.asarray(frames[:count][None])))
            timings[f"frames{count}_repeat{repetition}"] = time.perf_counter() - start
            if value.shape != (1, count * 256, 768) or not np.isfinite(value).all():
                raise ValueError("invalid spatial output")
            features[f"frames{count}_repeat{repetition}"] = value.reshape(count, 256, 768)
    delta = features["frames8_repeat1"] - features["frames1_repeat1"]
    close = np.isclose(features["frames8_repeat1"],
                       np.repeat(features["frames1_repeat1"], 8, axis=0),
                       **config["numerical_tolerance"])
    np.savez_compressed(out / "features.npz", **features)
    report = dict(timings=timings, maximum_absolute_difference=float(np.abs(delta).max()),
                  declared_tolerance_passed=bool(close.all()),
                  elements_outside_tolerance=int((~close).sum()),
                  comparison_elements=int(close.size),
                  repeat_max_abs={str(n): float(np.abs(features[f"frames{n}_repeat0"]
                                                       - features[f"frames{n}_repeat1"]).max())
                                  for n in (1, 8)},
                  parameters_resident=sum(x.size for x in jax.tree_util.tree_leaves(variables)),
                  parameter_groups={k: sum(x.size for x in jax.tree_util.tree_leaves(v))
                                    for k, v in variables["params"].items()},
                  feature_sha256=sha256_file(out / "features.npz"),
                  peak_working_set_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
                  devices=[str(d) for d in jax.devices()], sealed_audit_read=False,
                  quality_evaluated=False, weights_modified=False)
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
