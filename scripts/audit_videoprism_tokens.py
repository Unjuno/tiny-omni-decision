"""Two training-video unpooled-token feasibility audit; no training or quality evaluation."""

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.extract_frozen_videoprism_features import (
    CHECKPOINT_SHA256,
    MODEL_REVISION,
    SOURCE_COMMIT,
    _decode_video,
    _preprocess,
    sha256_file,
)
from tiny_omni_decision.video_token_grid import token_grid

MANIFEST = "5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d"
CACHE = "40eaff323e3654bf34653950d5fea4cd532f1a0e096438ec9cb73703dc589ea1"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def environment_versions(tensorflow_version):
    # Windows may install tensorflow-cpu rather than a distribution named tensorflow.
    return {p: importlib.metadata.version(p) for p in ["jax", "jaxlib", "numpy", "av"]} | {
        "tensorflow": tensorflow_version
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["sample-dir", "cache", "checkpoint", "source-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint changed")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.source_root, text=True
    ).strip()
    if source_commit != SOURCE_COMMIT:
        raise ValueError("source changed")
    manifest = args.sample_dir / "train.jsonl"
    if sha256_file(manifest) != MANIFEST or sha256_file(args.cache) != CACHE:
        raise ValueError("pinned data changed")
    selected, seen = [], set()
    for row in map(json.loads, manifest.read_text().splitlines()):
        if row["split"] != "train":
            raise ValueError("not training data")
        if row["media_sha256"] not in seen:
            seen.add(row["media_sha256"])
            selected.append(row)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError("need two distinct training assets")
    for row in selected:
        if sha256_file(args.sample_dir / row["media_path"]) != row["media_sha256"]:
            raise ValueError("media changed")
    if shutil.disk_usage(args.output.parent).free < 1024**3 + 2 * 2048 * 768 * 4:
        raise ValueError("insufficient disk reserve")
    with np.load(args.cache, allow_pickle=False) as cache:
        prior = dict(zip(cache["media_sha256"].astype(str), cache["features"], strict=True))
    args.output.mkdir(parents=True)
    config = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runner_sha256=sha256_file(Path(__file__)),
        argv={k: str(v) for k, v in vars(args).items()},
        model_revision=MODEL_REVISION,
        upstream_source_commit=SOURCE_COMMIT,
        checkpoint_sha256=CHECKPOINT_SHA256,
        train_manifest_sha256=MANIFEST,
        pooled_cache_sha256=CACHE,
        selected=selected,
        started_utc=datetime.now(UTC).isoformat(),
        python=sys.version,
        quality_evaluation=False,
        training=False,
    )
    write(args.output / "config.json", config)
    sys.path.insert(0, str(args.source_root))
    import jax
    import jax.numpy as jnp
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    process = psutil.Process()
    started = time.perf_counter()
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(args.checkpoint)
    )
    parameters = sum(int(x.size) for x in jax.tree_util.tree_leaves(variables))
    if parameters != 114365184:
        raise ValueError("unexpected resident parameters")
    encode = jax.jit(lambda video: model.apply(variables, video, train=False))
    loaded_seconds = time.perf_counter() - started
    records = []
    first_video, first_tokens = None, None
    for index, row in enumerate(selected):
        tick = time.perf_counter()
        frames = _decode_video(args.sample_dir / row["media_path"])
        video = jnp.asarray(_preprocess(frames, tf)[None])
        preprocess_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        result = encode(video)
        jax.block_until_ready(result)
        forward_seconds = time.perf_counter() - tick
        tokens = np.asarray(result[0] if isinstance(result, tuple) else result)
        grid = token_grid(tokens)
        delta = float(np.abs(tokens.mean(axis=1)[0] - prior[row["media_sha256"]]).max())
        if delta > 1e-5:
            raise ValueError(f"pooled reference mismatch: {delta}")
        path = args.output / f"tokens-{index}.npy"
        np.save(path, grid)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), grid)
        records.append(
            dict(
                scene=row["scene_index"],
                media_sha256=row["media_sha256"],
                shape=list(grid.shape),
                dtype=str(grid.dtype),
                array_bytes=grid.nbytes,
                file_bytes=path.stat().st_size,
                file_sha256=sha256_file(path),
                pooled_max_abs=delta,
                preprocess_seconds=preprocess_seconds,
                forward_seconds=forward_seconds,
                memory=process.memory_info()._asdict(),
            )
        )
        write(args.output / "progress.json", records)
        print(index, grid.shape, forward_seconds, delta, flush=True)
        if index == 0:
            first_video, first_tokens = video, tokens.copy()
    tick = time.perf_counter()
    repeated = encode(first_video)
    jax.block_until_ready(repeated)
    repeat_seconds = time.perf_counter() - tick
    repeat_tokens = np.asarray(repeated[0] if isinstance(repeated, tuple) else repeated)
    np.testing.assert_array_equal(first_tokens, repeat_tokens)
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint changed during audit")
    write(
        args.output / "result.json",
        dict(
            status="PASS_FEASIBILITY_ONLY",
            records=records,
            parameters=parameters,
            repeat_exact=True,
            repeat_forward_seconds=repeat_seconds,
            load_seconds=loaded_seconds,
            total_seconds=time.perf_counter() - started,
            memory=process.memory_info()._asdict(),
            devices=[str(d) for d in jax.devices()],
            backend=jax.default_backend(),
            versions=environment_versions(tf.__version__),
            projected456_scene_array_bytes=456 * records[0]["array_bytes"],
            projected456_scene_npy_bytes=456 * records[0]["file_bytes"],
            free_disk_bytes=shutil.disk_usage(args.output).free,
            finished_utc=datetime.now(UTC).isoformat(),
            cost_usd=0,
            limitation="Two training scenes; no full cache, training, or video quality evaluation.",
        ),
    )


if __name__ == "__main__":
    main()
