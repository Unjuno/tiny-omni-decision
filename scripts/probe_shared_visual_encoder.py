"""One loaded parameter tree, alternating raw image and past-video profiles."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.extract_frozen_videoprism_clevr4_features import _preprocess_rgb_image
from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    SOURCE_COMMIT,
    _preprocess,
    sha256_file,
)
from tiny_omni_decision.shared_visual_encoder import SharedVisualEncoder


def main():
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    r2 = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = r1 / "videoprism-base-r1/source"
    checkpoint = r1 / "videoprism-base-f16r288.npz"
    media = r2 / "vidimu-video-decode-smoke-01/input.mp4"
    image_cache = r2 / "videoprism-spatial-image-cache-01"
    video_cache = r2 / "videoprism-past-window-probe-01/features.npz"
    pins = {checkpoint: CHECKPOINT_SHA256,
        media: "b6d8fd31bc9459e3f822c4df20a844232db491d260cd82d5e0585af3df3de0d9",
        image_cache / "records.json":
            "32944fc822c45a7a93ee06904c9a97acaa5bf6364c33c40525f535c59da55b41",
        image_cache / "val-cells.npy":
            "1be3736b12cea9265233ebf005234c8e81cfa3c96645704e0cc2d00944b85fec",
        video_cache: "4c3238ee977e092a7fa28aaac244755665f900d193b60755b53777fa3532cea5"}
    for path, digest in pins.items():
        assert sha256_file(path) == digest
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                   text=True).strip() == SOURCE_COMMIT
    out = r2 / "shared-visual-encoder-probe-02"
    out.mkdir(exist_ok=False)
    config = dict(pins={str(p): h for p, h in pins.items()}, source=SOURCE_COMMIT,
        script_sha256=sha256_file(Path(__file__)),
        helper_sha256=sha256_file(Path("src/tiny_omni_decision/shared_visual_encoder.py")),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        feature_tolerance=dict(atol=1e-4, rtol=1e-5),
        order=["image", "video", "image", "video"],
        video_indices=[0, 17, 34, 51, 69, 86, 103, 120],
        video_window_ms=[0, 4000], sealed_audit=False, fitting=False)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    sys.path.insert(0, str(source))
    import av
    import jax
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    assert all(d.platform == "cpu" for d in jax.devices())
    rows = json.loads((image_cache / "records.json").read_text())["val"][:8]
    frames = []
    for row in rows:
        path = r2 / "clevr4-sample-seed17-1024-512/val/images" / (row["id"] + ".png")
        assert sha256_file(path) == row["sha256"]
        frames.append(_preprocess_rgb_image(
            tf.io.decode_png(tf.io.read_file(str(path)), channels=3), tf)[:1])
    image_frames = np.stack(frames)
    video_frames, timestamps = [], []
    with av.open(str(media)) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index in config["video_indices"]:
                video_frames.append(frame.to_ndarray(format="rgb24"))
                timestamps.append(float(frame.time) * 1000)
            if index >= 120:
                break
    assert len(video_frames) == 8 and min(timestamps) == 0 and max(timestamps) == 4000
    video_frames = _preprocess(np.stack(video_frames), tf)[None]
    model = vp.get_model("videoprism_public_v1_base")
    variables = jax.device_put(vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)))
    encoder = SharedVisualEncoder(model, variables, compile_fn=jax.jit)
    leaves = jax.tree_util.tree_leaves(variables)

    def tree_hash():
        digest = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(encoder.variables):
            value = np.asarray(leaf)
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    before = tree_hash()
    results, values = [], {}
    references = dict(image=np.load(image_cache / "val-cells.npy", allow_pickle=False)[:8],
                      video=np.load(video_cache, allow_pickle=False)["values"][:1])
    for modality in config["order"]:
        started = time.perf_counter()
        value = encoder.encode(image_frames if modality == "image" else video_frames,
                               modality=modality)
        duration = time.perf_counter() - started
        delta = float(np.abs(value-references[modality]).max())
        np.testing.assert_allclose(value, references[modality], **config["feature_tolerance"])
        if modality in values:
            np.testing.assert_array_equal(value, values[modality])
        values[modality] = value
        results.append(dict(modality=modality, seconds=duration, maximum_cache_delta=delta,
                            working_set_bytes=psutil.Process().memory_info().rss))
    assert tree_hash() == before
    assert encoder.variables is variables
    np.savez_compressed(out / "features.npz", **values)
    result = dict(status="PASS_SHARED_PROFILES", requests=results,
        parameters=sum(x.size for x in leaves), parameter_tree_sha256=before,
        same_parameter_tree=True, parameter_tree_unchanged=True, exact_repeat_outputs=True,
        features_sha256=sha256_file(out / "features.npz"), image_ids=[r["id"] for r in rows],
        selected_video_timestamps_ms=timestamps,
        peak_working_set_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
        backend="CPU JAX explicit parameter arguments", versions=dict(jax=jax.__version__),
        physical_backend_buffer_deduplication_proven=False)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
