"""Replay two pinned saved tensors through the JAX Python; synthetic RAM capacity check."""

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tiny_omni_decision.feature_cache import BoundedFeatureCache
from tiny_omni_decision.token_transport import read_tokens, write_tokens

SHAPE = (8, 16, 16, 768)
HASHES = [
    "e1e2d5307da6a7af6447ae14d9b81924ac4e1a5434ebae4f9009954168bcca17",
    "53cd5745bb1706668796893eb3256507816915a8cb43418245866d4c5ec39ddf",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_tree_rss(parent):
    """Include Windows venv launcher descendants, not only the launcher PID."""
    import psutil

    total = parent.memory_info().rss
    for child in parent.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass  # A child may finish between enumeration and measurement.
    return total


def live_worker(audit, stream, requested=None):
    """Re-extract the same two training assets, keeping the binary pipe clean."""
    from scripts.audit_videoprism_tokens import MANIFEST, environment_versions
    from scripts.extract_frozen_videoprism_features import (
        CHECKPOINT_SHA256,
        SOURCE_COMMIT,
        _decode_video,
        _preprocess,
    )
    from tiny_omni_decision.video_token_grid import token_grid

    config = json.loads((audit / "config.json").read_text())
    source = Path(config["argv"]["source_root"])
    checkpoint = Path(config["argv"]["checkpoint"])
    sample_dir = Path(config["argv"]["sample_dir"])
    if sha(checkpoint) != CHECKPOINT_SHA256 or sha(sample_dir / "train.jsonl") != MANIFEST:
        raise ValueError("pinned input changed")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != SOURCE_COMMIT
    ):
        raise ValueError("upstream source changed")
    if subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=source):
        raise ValueError("upstream tracked source modified")
    manifest = [json.loads(line) for line in (sample_dir / "train.jsonl").read_text().splitlines()]
    selected, seen = [], set()
    for row in manifest:
        if row["media_sha256"] not in seen:
            selected.append(row)
            seen.add(row["media_sha256"])
        if len(selected) == 2:
            break
    if selected != config["selected"]:
        raise ValueError("audited selection changed")
    if requested is not None:
        manifests = manifest + [
            json.loads(line) for line in (sample_dir / "validation.jsonl").read_text().splitlines()
        ]
        known = {row["media_sha256"]: row for row in manifests}
        if len(requested) != len(set(requested)) or not set(requested) <= known.keys():
            raise ValueError("unknown or repeated requested media")
        selected = [known[key] for key in requested]
    with contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, str(source))
        import jax
        import jax.numpy as jnp
        import psutil
        import tensorflow as tf
        from videoprism import models as vp

        model = vp.get_model("videoprism_public_v1_base")
        variables = vp.load_pretrained_weights(
            "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
        )
        parameters = sum(int(v.size) for v in jax.tree_util.tree_leaves(variables))
        if parameters != 114365184:
            raise ValueError("parameter count changed")
        encode = jax.jit(lambda x: model.apply(variables, x, train=False))
        for index, row in enumerate(selected):
            media = sample_dir / row["media_path"]
            allowed = {"train"} if requested is None else {"train", "validation"}
            if row["split"] not in allowed or sha(media) != row["media_sha256"]:
                raise ValueError("training media mismatch")
            started = time.perf_counter()
            video = jnp.asarray(_preprocess(_decode_video(media), tf)[None])
            result = encode(video)
            jax.block_until_ready(result)
            tokens = token_grid(np.asarray(result[0] if isinstance(result, tuple) else result))
            write_tokens(stream, str(index), tokens)
            print(
                json.dumps(
                    dict(
                        index=index,
                        media_sha256=row["media_sha256"],
                        encode_and_send_seconds=time.perf_counter() - started,
                    )
                ),
                file=sys.stderr,
                flush=True,
            )
        if sha(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("checkpoint changed during extraction")
        print(
            json.dumps(
                dict(
                    parameters=parameters,
                    versions=environment_versions(tf.__version__),
                    backend=jax.default_backend(),
                    process_memory=psutil.Process().memory_info()._asdict(),
                )
            ),
            file=sys.stderr,
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    paths = [args.audit / f"tokens-{i}.npy" for i in range(2)]
    if [sha(p) for p in paths] != HASHES:
        raise ValueError("pinned saved tensors changed")
    if args.worker:
        if args.live:
            requested = json.loads(args.request.read_text()) if args.request else None
            live_worker(args.audit, sys.stdout.buffer, requested)
            return
        for i, path in enumerate(paths):
            write_tokens(sys.stdout.buffer, str(i), np.load(path, allow_pickle=False))
        return
    import psutil

    if args.output is None or args.output.exists() or args.worker_python is None:
        raise ValueError("new output and existing worker Python required")
    available = psutil.virtual_memory().available
    required = 456 * int(np.prod(SHAPE)) * 4
    if available < required + 8 * 1024**3:
        raise ValueError("insufficient RAM with 8 GiB reserve")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    config = dict(
        argv=sys.argv,
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=sha(Path(__file__)),
        input_file_sha256=HASHES,
        worker_python=str(args.worker_python),
        available_ram_bytes=available,
        training=False,
        quality_evaluation=False,
        live_encoder_extraction=args.live,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    bank = np.empty((456, *SHAPE), dtype=np.float32)
    bank.fill(0)  # Touch every page: capacity measurement, NOT 456 extracted scenes.
    cache = BoundedFeatureCache(456)
    command = [
        str(args.worker_python),
        "-m",
        "scripts.audit_token_ram_transport",
        "--audit",
        str(args.audit),
        "--worker",
    ]
    if args.live:
        command.append("--live")
    records = []
    sampled_peak = 0
    sample_count = 0
    stop = threading.Event()
    with (args.output / "worker-stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)

        def sample_memory():
            nonlocal sampled_peak, sample_count
            parent = psutil.Process()
            while not stop.is_set():
                try:
                    sampled_peak = max(sampled_peak, process_tree_rss(parent))
                    sample_count += 1
                except psutil.NoSuchProcess:
                    break
                stop.wait(0.05)

        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
        try:
            for index in range(2):
                record = read_tokens(child.stdout, SHAPE)
                if record is None or record[0] != str(index):
                    raise ValueError("missing/out-of-order asset")
                bank[index] = record[1]
                original = np.load(paths[index], allow_pickle=False)
                np.testing.assert_array_equal(bank[index], original)
                feature = bank[index].view()
                feature.flags.writeable = False
                digest = hashlib.sha256(feature.tobytes()).hexdigest()
                cache.put(record[0], feature, digest)
                first, _ = cache.get(record[0])
                second, _ = cache.get(record[0])
                if first is not second or not np.shares_memory(first, bank):
                    raise ValueError("per-question copy detected")
                records.append(
                    dict(
                        key=record[0],
                        exact=True,
                        payload_sha256=digest,
                        repeated_lookup_same_object=True,
                    )
                )
            if read_tokens(child.stdout, SHAPE) is not None or child.wait(timeout=30) != 0:
                raise ValueError("unexpected worker output/exit")
        finally:
            if child.poll() is None:
                child.kill()  # This script owns this replay worker only.
                child.wait()
            child.stdout.close()
            stop.set()
            monitor.join()
    result = dict(
        status="PASS_TRANSPORT_CAPACITY_ONLY",
        records=records,
        allocated_slots=456,
        real_saved_assets_loaded=2,
        live_assets_encoded=2 if args.live else 0,
        synthetic_zero_slots=454,
        bank_bytes=bank.nbytes,
        process_memory=psutil.Process().memory_info()._asdict(),
        sampled_process_tree_peak_rss=sampled_peak,
        memory_sample_interval_seconds=0.05,
        memory_sample_count=sample_count,
        available_ram_after_bytes=psutil.virtual_memory().available,
        seconds=time.perf_counter() - started,
        worker_command=command,
        finished_utc=datetime.now(UTC).isoformat(),
        cost_usd=0,
        limitation=(
            "Two live training scenes only; no full extraction, training, or quality result."
            if args.live
            else "Saved-token replay only; no full extraction, training, or quality result."
        ),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
