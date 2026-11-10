"""Extract pinned frozen VideoPrism features from the fixed train/valid media inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
MODEL_PARAMETERS = 114_365_184
FRAME_COUNT = 8
FRAME_SIZE = 288


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def uniform_indices(total: int, requested: int = FRAME_COUNT) -> list[int]:
    if total < requested or requested < 1:
        raise ValueError("video must contain at least the requested number of frames")
    if requested == 1:
        return [0]
    stride = (total - 1) / (requested - 1)
    result = [round(i * stride) for i in range(requested)]
    if len(set(result)) != requested:
        raise ValueError("uniform frame selection produced duplicate indices")
    return result


def _decode_frames(path: Path, requested: int = FRAME_COUNT) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"video stream missing: {path}")
        stream = streams[0]
        total = int(stream.frames or 0)
        if total >= requested:
            wanted = set(uniform_indices(total, requested))
            selected: dict[int, np.ndarray] = {}
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    selected[index] = frame.to_ndarray(format="rgb24")
                if index >= max(wanted):
                    break
            if set(selected) != wanted:
                raise ValueError(f"decoder returned {len(selected)}/{requested} selected frames")
            return np.stack([selected[index] for index in sorted(wanted)])

        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if len(frames) < requested:
        raise ValueError(f"video decoded only {len(frames)} frames; need {requested}")
    return np.stack([frames[index] for index in uniform_indices(len(frames), requested)])


def _preprocess(frames: np.ndarray, tf: Any) -> np.ndarray:
    height, width = frames.shape[1:3]
    scale = max(FRAME_SIZE / height, FRAME_SIZE / width)
    resized = tf.image.resize(
        frames,
        (round(height * scale), round(width * scale)),
        method="bilinear",
    )
    cropped = tf.image.resize_with_crop_or_pad(resized, FRAME_SIZE, FRAME_SIZE)
    return np.asarray(cropped, dtype=np.float32) / 255.0


def _load_model(checkpoint: Path, source_root: Path):
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError("VideoPrism source checkout differs from pinned pristine revision")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import jax
    import jax.numpy as jnp
    import tensorflow as tf

    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("VideoPrism extraction is pinned to CPU")
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise ValueError("TensorFlow GPU was initialized before CPU isolation") from error
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    sys.path.insert(0, str(source_root))
    from videoprism import models as vp

    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    parameters = sum(int(value.size) for value in jax.tree_util.tree_leaves(variables))
    if parameters != MODEL_PARAMETERS:
        raise ValueError(f"unexpected VideoPrism parameter count: {parameters}")
    encode = jax.jit(lambda batch: model.apply(variables, batch, train=False))
    return jax, jnp, tf, encode, parameters, commit


def _load_fixed_inputs(preflight_path: Path, media_manifest_path: Path, split: str):
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    media_manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))
    preflight_hash = sha256_file(preflight_path)
    if preflight.get("status") != "PASS_METADATA_ONLY_MEDIA_PREFLIGHT":
        raise ValueError("media preflight did not pass")
    if preflight.get("official_test_archive_opened") is not False:
        raise ValueError("media preflight does not attest test-split isolation")
    if media_manifest.get("media_preflight_sha256") != preflight_hash:
        raise ValueError("media acquisition belongs to a different preflight")
    if media_manifest.get("official_test_archive_opened") is not False:
        raise ValueError("media acquisition does not attest test-split isolation")
    expected = preflight["splits"][split]["members"]
    actual = [row for row in media_manifest["media_files"] if row["split"] == split]
    expected_by_id = {row["video_id"]: row for row in expected}
    actual_by_id = {row["video_id"]: row for row in actual}
    if set(expected_by_id) != set(actual_by_id):
        raise ValueError(f"{split} media IDs differ from frozen preflight")
    for video_id, row in expected_by_id.items():
        receipt = actual_by_id[video_id]
        for field in ("member", "expanded_bytes", "compressed_bytes", "crc32"):
            if receipt.get(field) != row.get(field):
                raise ValueError(f"{split} media receipt differs from preflight: {video_id}")
    ordered = sorted(actual, key=lambda row: row["video_id"])
    return preflight_hash, sha256_file(media_manifest_path), ordered


def extract_split(
    *,
    preflight_path: Path,
    media_manifest_path: Path,
    media_root: Path,
    checkpoint: Path,
    source_root: Path,
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    if split not in {"train", "valid"}:
        raise ValueError("only train and public validation splits are supported")
    for path in (preflight_path, media_manifest_path, checkpoint, source_root, media_root):
        path.resolve(strict=True)
    if output_dir.drive.upper() != "D:":
        raise ValueError("feature outputs must stay on D:")
    preflight_hash, media_hash, media_rows = _load_fixed_inputs(
        preflight_path, media_manifest_path, split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run.json"
    identity = {
        "schema_version": 1,
        "status": "RUNNING",
        "split": split,
        "preflight_sha256": preflight_hash,
        "media_manifest_sha256": media_hash,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "frame_count": FRAME_COUNT,
        "frame_size": FRAME_SIZE,
        "pooling": "mean of all spatial-temporal VideoPrism tokens, per unique video",
        "labels_loaded": False,
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in identity.items() if key != "status"):
            raise ValueError("existing feature run metadata differs; refusing resume")
    else:
        metadata_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    receipt_path = output_dir / "feature-progress.jsonl"
    receipts: dict[str, dict[str, Any]] = {}
    if receipt_path.exists():
        for line in receipt_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["video_id"] in receipts:
                raise ValueError("duplicate feature progress receipt")
            receipts[row["video_id"]] = row
    expected_ids = {row["video_id"] for row in media_rows}
    media_by_id = {row["video_id"]: row for row in media_rows}
    if not set(receipts).issubset(expected_ids):
        raise ValueError("feature receipts contain an unexpected video")
    for video_id, receipt in receipts.items():
        if (
            receipt.get("split") != split
            or receipt.get("media_sha256") != media_by_id[video_id]["sha256"]
        ):
            raise ValueError(f"feature receipt media identity mismatch: {video_id}")
        path = output_dir / f"{video_id}.npy"
        if not path.is_file() or sha256_file(path) != receipt["feature_sha256"]:
            raise ValueError(f"feature shard missing or hash mismatch: {video_id}")

    jax, jnp, tf, encode, parameters, commit = _load_model(checkpoint, source_root)
    started = time.perf_counter()
    import psutil

    process = psutil.Process()
    pending = [row for row in media_rows if row["video_id"] not in receipts]
    with receipt_path.open("a", encoding="utf-8", newline="\n") as progress:
        for item in pending:
            video_id = item["video_id"]
            relative = Path(item["local_relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe path in acquisition receipt")
            video_path = media_root / relative
            if (
                not video_path.is_file()
                or video_path.stat().st_size != item["expanded_bytes"]
                or sha256_file(video_path) != item["sha256"]
            ):
                raise ValueError(f"media file differs from pinned receipt: {video_id}")
            frames = _decode_frames(video_path)
            batch = jnp.asarray(_preprocess(frames, tf)[None, ...])
            forward_started = time.perf_counter()
            result = encode(batch)
            jax.tree_util.tree_map(
                lambda value: value.block_until_ready()
                if hasattr(value, "block_until_ready")
                else value,
                result,
            )
            tokens = result[0] if isinstance(result, tuple) else result
            feature = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
            if feature.shape != (768,) or not np.isfinite(feature).all():
                raise ValueError(f"invalid VideoPrism feature for {video_id}")
            destination = output_dir / f"{video_id}.npy"
            temporary = output_dir / f"{video_id}.npy.tmp"
            if destination.exists() or temporary.exists():
                raise FileExistsError(f"refusing to overwrite feature shard: {video_id}")
            with temporary.open("xb") as stream:
                np.save(stream, feature, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
            receipt = {
                "video_id": video_id,
                "split": split,
                "media_sha256": item["sha256"],
                "feature_sha256": sha256_file(destination),
                "feature_bytes": destination.stat().st_size,
                "forward_seconds": time.perf_counter() - forward_started,
            }
            progress.write(json.dumps(receipt, sort_keys=True) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
            receipts[video_id] = receipt

    if set(receipts) != expected_ids:
        raise RuntimeError("feature extraction ended before every split video was processed")
    report = {
        **identity,
        "status": "COMPLETE",
        "video_count": len(receipts),
        "model_parameters": parameters,
        "devices": [str(device) for device in jax.devices()],
        "gpu_used": False,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_working_set_bytes": getattr(process.memory_info(), "peak_wset", None),
        "feature_receipts": [receipts[key] for key in sorted(receipts)],
    }
    report_path = output_dir / "feature-manifest.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite feature manifest: {report_path}")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["feature_manifest_sha256"] = sha256_file(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    args = parser.parse_args()
    report = extract_split(
        preflight_path=args.preflight,
        media_manifest_path=args.media_manifest,
        media_root=args.media_root,
        checkpoint=args.checkpoint,
        source_root=args.source_root,
        output_dir=args.output_dir,
        split=args.split,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "split": report["split"],
                "video_count": report["video_count"],
                "feature_manifest_sha256": report["feature_manifest_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
