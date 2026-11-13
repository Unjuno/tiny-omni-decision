"""Extract pinned frozen VideoPrism tokens for the grouped CATER temporal task."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_omni_decision.temporal_tokens import reshape_videoprism_temporal_tokens  # noqa: E402


def select_interval_frame_indices(start: int, end: int, *, count: int = 13) -> tuple[int, ...]:
    """Sample a fixed-length clip wholly inside one labelled 75-frame interval."""
    if start < 0 or end <= start or count < 2:
        raise ValueError("expected a non-empty interval and at least two frame samples")
    indices = tuple(round(start + (end - start) * index / (count - 1)) for index in range(count))
    if len(set(indices)) != count:
        raise ValueError("interval is too short for the requested number of distinct frames")
    return indices

SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
PARAMETER_COUNT = 114_365_184
FRAME_SIZE = 288


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def attach_media_records(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    media_records: list[dict[str, Any]],
    media_root: Path,
) -> tuple[list[tuple[dict[str, Any], Path, str]], list[tuple[dict[str, Any], Path, str]]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in media_records:
        key = (str(record["split"]), str(record["video_id"]))
        if key in by_key:
            raise ValueError(f"duplicate media manifest entry: {key}")
        by_key[key] = record

    attached = []
    verified_media: dict[tuple[str, str], tuple[Path, str]] = {}
    split_videos: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split, rows in (("train", train), ("validation", validation)):
        for row in rows:
            if row.get("split") != split:
                raise ValueError(f"mislabelled {split} sample: {row.get('id')}")
            video_id = str(row["video_id"])
            group_id = str(row["group_id"])
            if group_id != f"cater-video-{Path(video_id).stem.removeprefix('CATER_new_')}":
                raise ValueError(f"video/group identity mismatch: {row.get('id')}")
            record = by_key.get((split, video_id))
            if record is None:
                raise ValueError(f"missing downloaded media record: {split}/{video_id}")
            media_key = (split, video_id)
            verified = verified_media.get(media_key)
            if verified is None:
                media = media_root / str(record["relative_path"])
                expected_hash = str(record["media_sha256"])
                if not media.is_file() or sha256_file(media) != expected_hash:
                    raise ValueError(f"missing or hash-mismatched CATER video: {media}")
                verified = (media, expected_hash)
                verified_media[media_key] = verified
            media, expected_hash = verified
            split_videos[split].add(video_id)
            attached.append((row, media, expected_hash))

    if split_videos["train"] & split_videos["validation"]:
        raise ValueError("CATER train/validation video overlap")
    expected_keys = {
        (split, video_id)
        for split, videos in split_videos.items()
        for video_id in videos
    }
    if set(by_key) != expected_keys:
        raise ValueError("downloaded media manifest does not match the frozen task manifests")
    train_attached = [item for item in attached if item[0]["split"] == "train"]
    validation_attached = [item for item in attached if item[0]["split"] == "validation"]
    return train_attached, validation_attached


def decode_cater_frames(path: Path, av: Any, frame_indices: tuple[int, ...]) -> np.ndarray:
    selected_indices = set(frame_indices)
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"video stream missing: {path}")
        stream = streams[0]
        if stream.frames and stream.frames < max(selected_indices) + 1:
            raise ValueError(f"CATER video does not cover the requested interval: {path}")
        selected: list[np.ndarray] = []
        for index, frame in enumerate(container.decode(stream)):
            if index in selected_indices:
                selected.append(frame.to_ndarray(format="rgb24"))
            if index >= max(selected_indices):
                break
    if len(selected) != len(selected_indices):
        raise ValueError(f"decoded {len(selected)}/{len(selected_indices)} required frames: {path}")
    return np.stack(selected)


def preprocess(frames: np.ndarray, tf: Any) -> np.ndarray:
    height, width = frames.shape[1:3]
    scale = max(FRAME_SIZE / height, FRAME_SIZE / width)
    new_size = (round(height * scale), round(width * scale))
    resized = tf.image.resize(frames, new_size, method="bilinear")
    cropped = tf.image.resize_with_crop_or_pad(resized, FRAME_SIZE, FRAME_SIZE)
    return np.asarray(cropped, dtype=np.float32) / 255.0


def token_tensor(tokens: np.ndarray, frame_count: int) -> np.ndarray:
    values = np.asarray(tokens)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 768:
        raise ValueError(f"unexpected VideoPrism token shape: {values.shape}")
    result = reshape_videoprism_temporal_tokens(values, frame_count)[0]
    if result.shape != (frame_count, 256, 768) or not np.isfinite(result).all():
        raise ValueError(f"invalid temporal/spatial tokens: {result.shape}")
    if float(np.max(np.abs(result))) > float(np.finfo(np.float16).max):
        raise ValueError("VideoPrism features exceed the finite float16 range")
    return result.astype(np.float16)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_dir = args.manifest_dir.resolve(strict=True)
    media_dir = args.media_dir.resolve(strict=True)
    media_manifest_path = args.media_manifest.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite CATER temporal feature output: {output}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA-256 mismatch")
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != SOURCE_COMMIT:
        raise ValueError(f"unexpected VideoPrism source commit: {source_commit}")

    train_rows = read_jsonl(manifest_dir / "train.jsonl")
    validation_rows = read_jsonl(manifest_dir / "validation.jsonl")
    media_records = read_jsonl(media_manifest_path)
    train, validation = attach_media_records(train_rows, validation_rows, media_records, media_dir)
    unique_by_sample: dict[tuple[str, str], tuple[dict[str, Any], Path, str]] = {}
    for rows in (train, validation):
        for row, media, media_hash in rows:
            unique_by_sample[(row["split"], str(row["id"]))] = (row, media, media_hash)

    sys.path.insert(0, str(source_root))
    import av
    import jax
    import jax.numpy as jnp
    import psutil
    import tensorflow as tf
    from videoprism import models as vp

    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("CATER VideoPrism extraction is pinned to CPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    parameter_count = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(variables))
    if parameter_count != PARAMETER_COUNT:
        raise ValueError(f"unexpected VideoPrism parameter count: {parameter_count}")
    encode = jax.jit(lambda batch: model.apply(variables, batch, train=False))
    metadata = {
        "status": "RUNNING",
        "source_commit": source_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "parameters": parameter_count,
        "device": "CPU",
        "frame_sampling": "13 evenly spaced frames from each sample's inclusive task interval",
        "feature_layout": (
            "[13,256,768] spatial tokens; independently encoded per question interval"
        ),
        "storage_dtype": "float16",
        "manifest_sha256": {
            "train": sha256_file(manifest_dir / "train.jsonl"),
            "validation": sha256_file(manifest_dir / "validation.jsonl"),
            "media_manifest": sha256_file(media_manifest_path),
        },
    }
    if output.exists():
        run_path = output / "run.json"
        if not run_path.is_file() or (output / "result.json").exists():
            raise ValueError("resume requires an incomplete run with run.json and no result.json")
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in metadata.items()):
            raise ValueError("resume inputs/config do not match the existing feature run")
        temporary_files = list(output.rglob("*.tmp"))
        if temporary_files:
            raise ValueError(f"temporary feature files require manual review: {temporary_files[0]}")
    else:
        output.mkdir(parents=True)
        atomic_json(output / "run.json", metadata)
    process = psutil.Process()
    rss_start = process.memory_info().rss
    started = time.perf_counter()
    records = []
    cache_root = output / "features"
    cache_root.mkdir(exist_ok=True)
    for index, ((split, sample_id), (row, media, media_hash)) in enumerate(
        unique_by_sample.items(), 1
    ):
        video_id = str(row["video_id"])
        interval = tuple(map(int, row["frame_interval"]))
        frame_indices = select_interval_frame_indices(*interval)
        relative = Path(split) / f"{sample_id}.npz"
        feature_path = cache_root / relative
        feature_path.parent.mkdir(exist_ok=True)
        if feature_path.exists():
            with np.load(feature_path, allow_pickle=False) as archive:
                features = archive["features"]
                record = json.loads(str(archive["metadata"].item()))
            if (
                record.get("split") != split
                or record.get("video_id") != video_id
                or record.get("sample_id") != sample_id
                or record.get("frame_interval") != list(interval)
                or record.get("media_sha256") != media_hash
                or record.get("feature_sha256")
                != hashlib.sha256(features.tobytes()).hexdigest()
                or features.shape != (13, 256, 768)
                or features.dtype != np.float16
                or not np.isfinite(features).all()
            ):
                raise ValueError(
                    f"existing cached feature failed resume verification: {feature_path}"
                )
            record["feature_file_sha256"] = sha256_file(feature_path)
            record["feature_file_bytes"] = feature_path.stat().st_size
            records.append(record)
            continue
        frames = decode_cater_frames(media, av, frame_indices)
        batch = jnp.asarray(preprocess(frames, tf)[None])
        forward_started = time.perf_counter()
        encoded = encode(batch)
        jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready()
                if hasattr(value, "block_until_ready")
                else value
            ),
            encoded,
        )
        forward_seconds = time.perf_counter() - forward_started
        tokens = encoded[0] if isinstance(encoded, tuple) else encoded
        features = token_tensor(np.asarray(tokens), len(frame_indices))
        record = {
            "split": split,
            "sample_id": sample_id,
            "video_id": video_id,
            "frame_interval": list(interval),
            "frame_indices": list(frame_indices),
            "media_sha256": media_hash,
            "feature_path": str(relative),
            "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "forward_seconds": forward_seconds,
        }
        temporary = feature_path.with_suffix(".npz.tmp")
        with temporary.open("xb") as stream:
            np.savez(
                stream,
                features=features,
                metadata=np.asarray(json.dumps(record, sort_keys=True)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, feature_path)
        record["feature_file_sha256"] = sha256_file(feature_path)
        record["feature_file_bytes"] = feature_path.stat().st_size
        records.append(record)
        if index % 16 == 0:
            print(
                json.dumps(
                    {"completed_samples": index, "total_samples": len(unique_by_sample)}
                ),
                flush=True,
            )

    result = {
        "status": "PASS_CATER_FROZEN_VIDEOPRISM_FEATURES",
        **{key: value for key, value in metadata.items() if key != "status"},
        "examples": len(records),
        "train_example_count": sum(row[0]["split"] == "train" for row in train),
        "validation_example_count": sum(row[0]["split"] == "validation" for row in validation),
        "train_video_count": len({row[0]["video_id"] for row in train}),
        "validation_video_count": len({row[0]["video_id"] for row in validation}),
        "elapsed_seconds": time.perf_counter() - started,
        "rss_start_bytes": rss_start,
        "rss_end_bytes": process.memory_info().rss,
        "records": records,
        "validation_scope": "development-only; no audit data loaded",
    }
    result_path = output / "result.json"
    atomic_json(result_path, result)
    metadata["status"] = result["status"]
    metadata["result_sha256"] = sha256_file(result_path)
    metadata["elapsed_seconds"] = result["elapsed_seconds"]
    atomic_json(output / "run.json", metadata)
    return result


def main() -> None:
    result = run(_args())
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
