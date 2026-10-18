"""Fixed frozen single-image spatial features; excludes reporting/audit images."""

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
from tiny_omni_decision.dinov2_features import require_image_splits
from tiny_omni_decision.videoprism_spatial_features import spatial_cells


def main():
    r2 = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    paths = {"train": r2 / "clevr4-full-coverage-seed17",
             "val": r2 / "clevr4-sample-seed17-1024-512"}
    pins = {"train": "2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010",
            "val": "77faf433844be6b350cd0205fd3834cd8b028a6491ddd9eae0f6e9b7610cd38a"}
    rows = {}
    for split, directory in paths.items():
        path = directory / f"{split}-image-manifest.jsonl"
        if sha256_file(path) != pins[split]:
            raise ValueError("manifest changed")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        if any(r["split"] != split for r in rows[split]):
            raise ValueError("split mismatch")
    if (len(rows["train"]), len(rows["val"])) != (8424, 512):
        raise ValueError("fixed counts changed")
    require_image_splits(rows["train"], rows["val"])
    source = r1 / "videoprism-base-r1/source"
    checkpoint = r1 / "videoprism-base-f16r288.npz"
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint changed")
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                               text=True).strip() != SOURCE_COMMIT:
        raise ValueError("source changed")
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    if any(d.platform != "cpu" for d in jax.devices()):
        raise ValueError("CPU extraction only")
    out = r2 / "videoprism-spatial-image-cache-01"
    out.mkdir(exist_ok=False)
    config = dict(batch=8, single_frame=True, manifests=pins, checkpoint_sha256=CHECKPOINT_SHA256,
                  upstream_commit=SOURCE_COMMIT, script_sha256=sha256_file(Path(__file__)),
                  helper_sha256=sha256_file(
                      Path("src/tiny_omni_decision/videoprism_spatial_features.py")),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        text=True).strip(),
                  features="global spatial mean + sixteen 4x4 mean cells, float32,17x768")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "records.json").write_text(json.dumps(rows))
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint))
    encode = jax.jit(lambda x: model.apply(
        variables, x, train=False, return_intermediate={"spatial_features"})[1]["spatial_features"])
    started = time.perf_counter()
    with (out / "progress.jsonl").open("w") as log:
        for split, data in rows.items():
            values = np.lib.format.open_memmap(out / f"{split}-cells.npy", mode="w+",
                                              dtype=np.float32, shape=(len(data), 17, 768))
            for offset in range(0, len(data), 8):
                images = []
                for row in data[offset:offset + 8]:
                    path = paths[split] / split / "images" / (row["id"] + ".png")
                    if sha256_file(path) != row["sha256"]:
                        raise ValueError("media changed")
                    image = tf.io.decode_png(tf.io.read_file(str(path)), channels=3)
                    images.append(_preprocess_rgb_image(image, tf)[:1])
                tokens = np.asarray(encode(jnp.asarray(np.stack(images))))
                values[offset:offset + len(images)] = spatial_cells(tokens)
                values.flush()
                progress = dict(split=split, completed=offset + len(images), total=len(data),
                                elapsed_seconds=time.perf_counter() - started)
                log.write(json.dumps(progress) + "\n")
                log.flush()
                if offset % 256 == 0:
                    print(json.dumps(progress), flush=True)
            del values
    report = dict(status="PASS", feature_hashes={s: sha256_file(out / f"{s}-cells.npy")
                                                 for s in rows},
                  config_sha256=sha256_file(out / "config.json"),
                  records_sha256=sha256_file(out / "records.json"),
                  seconds=time.perf_counter() - started,
                  parameters=sum(x.size for x in jax.tree_util.tree_leaves(variables)),
                  peak_working_set_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
                  sealed_audit_read=False, reporting_subset_read=False)
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
