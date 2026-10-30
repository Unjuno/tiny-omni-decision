"""Persistent pinned native CPU VideoPrism worker, isolated from Torch/NumPy."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    protocol = sys.stdout
    tick = time.perf_counter()
    report = dict(status="RUNNING", calls=[], training=False, gpu_used=False)
    with contextlib.redirect_stdout(sys.stderr):
        import av
        import jax
        import jax.numpy as jnp
        import numpy as np
        import psutil
        import tensorflow as tf
        from videoprism import models as vp

        source = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
        checkpoint = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-f16r288.npz")
        revision = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
        if (
            subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            != revision
            or subprocess.check_output(
                ["git", "-C", str(source), "diff", "--name-only"], text=True
            ).strip()
            or sha(checkpoint) != "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
        ):
            raise ValueError("native source/checkpoint changed")
        pins = {str(p): sha(p) for p in source.rglob("*.py")}
        pins[str(checkpoint)] = sha(checkpoint)
        pins[str(Path(__file__).resolve())] = sha(__file__)
        for module in [jax, np, tf, av, psutil, vp]:
            for p in Path(module.__file__).resolve().parent.rglob("*"):
                if p.is_file() and p.suffix in {".py", ".pyd", ".dll"}:
                    pins[str(p)] = sha(p)
        dependency_root = Path(
            "C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Lib/site-packages"
        )
        for p in dependency_root.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".pyd", ".dll"}:
                pins[str(p)] = sha(p)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        if any(d.platform != "cpu" for d in jax.devices()):
            raise ValueError("CPU required")
        model = vp.get_model("videoprism_public_v1_base")
        variables = vp.load_pretrained_weights(
            "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
        )
        params = sum(v.size for v in jax.tree_util.tree_leaves(variables))
        if params != 114365184:
            raise ValueError("native parameter count differs")

        def state_sha():
            h = hashlib.sha256()
            for v in jax.tree_util.tree_leaves(variables):
                a = np.asarray(v)
                h.update(str(a.shape).encode())
                h.update(a.tobytes())
            return h.hexdigest()

        before = state_sha()
        encode = jax.jit(lambda x: model.apply(variables, x, train=False))
        report.update(
            parameters=params,
            source_revision=revision,
            validated_hashes=pins,
            parameter_sha_before=before,
            python=sys.version,
            jax=jax.__version__,
            numpy=np.__version__,
            tensorflow=tf.__version__,
            av=av.__version__,
            load_seconds=time.perf_counter() - tick,
        )
    protocol.write(
        json.dumps(
            dict(
                ready=True,
                load_seconds=report["load_seconds"],
                profile_sha256=hashlib.sha256(
                    json.dumps(pins, sort_keys=True).encode()
                ).hexdigest(),
            )
        )
        + "\n"
    )
    protocol.flush()
    try:
        for line in sys.stdin:
            with contextlib.redirect_stdout(sys.stderr):
                if time.perf_counter() - tick > 600 or shutil.disk_usage(args.output).free < 2**30:
                    raise RuntimeError("native worker time/disk guard")
                request = json.loads(line)
                path = Path(request["path"])
                digest = sha(path)
                if digest != request["sha256"]:
                    raise ValueError("raw media changed")
                pins[str(path.resolve())] = digest
                start = time.perf_counter()
                frames = []
                pts = []
                with av.open(io.BytesIO(path.read_bytes())) as container:
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 1
                    for frame in container.decode(stream):
                        pts.append(float(frame.pts * frame.time_base))
                        frames.append(frame.to_ndarray(format="rgb24"))
                if not frames or not np.isfinite(pts).all() or not (np.diff(pts) > 0).all():
                    raise ValueError("complete ordered native frames required")
                indices = [
                    int(np.abs(np.asarray(pts) - t).argmin())
                    for t in np.linspace(pts[0], pts[-1], 16)
                ]
                chosen = np.stack([frames[i] for i in indices])
                h, w = chosen.shape[1:3]
                factor = max(288 / h, 288 / w)
                resized = tf.image.resize(
                    chosen, (round(h * factor), round(w * factor)), method="bilinear"
                )
                inp = (
                    np.asarray(tf.image.resize_with_crop_or_pad(resized, 288, 288), np.float32)[
                        None
                    ]
                    / 255.0
                )
                decode = time.perf_counter() - start
                start = time.perf_counter()
                value = encode(jnp.asarray(inp))
                jax.tree_util.tree_map(lambda v: v.block_until_ready(), value)
                tokens = np.asarray(value[0] if isinstance(value, tuple) else value)
                if tokens.shape != (1, 4096, 768) or not np.isfinite(tokens).all():
                    raise ValueError("native tokens differ")
                feature = tokens.reshape(1, 16, 256, 768).mean(2)[0]
                item = dict(
                    media_sha256=digest,
                    frames=len(frames),
                    indices=indices,
                    pts=[pts[i] for i in indices],
                    preprocessed_sha256=hashlib.sha256(inp.tobytes()).hexdigest(),
                    decode_preprocess_seconds=decode,
                    forward_seconds=time.perf_counter() - start,
                )
                report["calls"].append(item)
                peak = getattr(
                    psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss
                )
                if peak > 6 * 2**30:
                    raise RuntimeError("native worker RAM guard")
                np.save(args.output / f"call-{len(report['calls']):02}.npy", feature)
            protocol.write(json.dumps(dict(feature=feature.tolist(), metadata=item)) + "\n")
            protocol.flush()
        after = state_sha()
        if after != before or any(sha(p) != h for p, h in pins.items()):
            raise ValueError("native terminal hash differs")
        report.update(
            status="PASS_NATIVE_VIDEO_WORKER",
            parameter_sha_after=after,
            parameters_unchanged=True,
            artifact_hashes={p.name: sha(p) for p in args.output.glob("*.npy")},
        )
    except Exception as error:
        report.update(status="FAIL_INCOMPLETE", error=repr(error))
        raise
    finally:
        report.update(
            seconds=time.perf_counter() - tick,
            peak_cpu_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
        )
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
