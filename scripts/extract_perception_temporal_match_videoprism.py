"""Extract aligned frozen VideoPrism features for the dev-only AV task."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
MODEL_PARAMETERS = 114_365_184


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def media_and_hash(
    split: str,
    video_id: str,
    *,
    old_media: Path,
    added_train_media: Path,
    fresh_valid_media: Path,
    old_hashes: dict[tuple[str, str, str], str],
    new_train_hashes: dict[tuple[str, str, str], str],
    fresh_valid_hashes: dict[tuple[str, str, str], str],
) -> tuple[Path, str]:
    candidates = (
        [(old_media / split / f"{video_id}.mp4", old_hashes)]
        + (
            [(added_train_media / "train" / f"{video_id}.mp4", new_train_hashes)]
            if split == "train"
            else []
        )
        + (
            [(fresh_valid_media / "valid" / f"{video_id}.mp4", fresh_valid_hashes)]
            if split == "valid"
            else []
        )
    )
    for path, hashes in candidates:
        key = (split, video_id, ".mp4")
        if path.is_file() and key in hashes:
            return path, hashes[key]
    raise FileNotFoundError(f"no pinned video asset for {split}:{video_id}")


def load_hash_maps(old_media: Path, train_report: Path, fresh_valid_report: Path):
    old = json.loads((old_media / "acquisition_manifest.json").read_text(encoding="utf-8"))
    train = json.loads(train_report.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_valid_report.read_text(encoding="utf-8"))
    old_hashes = {
        (row["split"], row["video_id"], Path(row["member"]).suffix): row["sha256"]
        for row in old["files"]
    }
    train_hashes = {
        (row["split"], row["video_id"], Path(row["member"]).suffix): row["sha256"]
        for row in train["files"]
    }
    fresh_hashes = {
        ("valid", row["video_id"], ".mp4"): row["video_sha256"]
        for row in fresh["examples_manifest"]
    }
    return old_hashes, train_hashes, fresh_hashes


def closest_frames(
    path: Path, target_times: list[float], av_module: Any
) -> tuple[np.ndarray, list[float], float]:
    targets = np.asarray(target_times, dtype=np.float64)
    best: list[tuple[float, float | None, np.ndarray | None]] = [
        (float("inf"), None, None) for _ in targets
    ]
    with av_module.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"video stream missing: {path}")
        stream = streams[0]
        time_base = float(stream.time_base)
        container.seek(
            max(0, int((float(targets[0]) - 2.0) / time_base)), stream=stream, backward=True
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * frame.time_base)
            if pts > float(targets[-1]) + 0.1:
                break
            if pts < float(targets[0]) - 0.15:
                continue
            distances = np.abs(targets - pts)
            changed = False
            for index, distance in enumerate(distances):
                if distance < best[index][0]:
                    if not changed:
                        rgb = frame.to_ndarray(format="rgb24")
                        changed = True
                    best[index] = (float(distance), pts, rgb)
    if any(row[2] is None for row in best):
        raise ValueError(f"missing target frames in {path}")
    errors = [row[0] for row in best]
    if max(errors) > 0.05:
        raise ValueError(f"frame timestamp error >50ms in {path}: {max(errors)}")
    return np.stack([row[2] for row in best]), [float(row[1]) for row in best], max(errors)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.resolve(strict=True)
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing feature output: {output}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("VideoPrism checkpoint SHA mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        commit != SOURCE_COMMIT
        or subprocess.check_output(
            ["git", "-C", str(source), "diff", "--name-only"], text=True
        ).strip()
    ):
        raise ValueError("VideoPrism source checkout is not pristine at the pinned commit")
    old_hashes, train_hashes, fresh_hashes = load_hash_maps(
        args.old_media, args.train_acquisition, args.fresh_valid_media_report
    )

    import av
    import jax
    import jax.numpy as jnp
    import tensorflow as tf

    sys.path.insert(0, str(source))
    from videoprism import models as vp

    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("VideoPrism extraction must use the pinned CPU runtime")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    parameters = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(variables))
    if parameters != MODEL_PARAMETERS:
        raise ValueError(f"unexpected VideoPrism parameter count: {parameters}")
    encode = jax.jit(lambda batch: model.apply(variables, batch, train=False))
    output.mkdir(parents=True)
    atomic_json(
        output / "run.json",
        {
            "status": "RUNNING",
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_commit": commit,
            "parameters": parameters,
            "device": "CPU",
            "frames": 8,
            "pooling": "mean over VideoPrism tokens",
            "validation_scope": "development-only",
        },
    )

    from scripts.extract_frozen_videoprism_features import _preprocess

    records = []
    started = time.perf_counter()
    total = sum(len(manifest[split]) for split in ("train", "validation"))
    for split_name, media_split in (("train", "train"), ("validation", "valid")):
        for row in manifest[split_name]:
            video_path, expected_media_hash = media_and_hash(
                media_split,
                row["video_id"],
                old_media=args.old_media,
                added_train_media=args.added_train_media,
                fresh_valid_media=args.fresh_valid_media,
                old_hashes=old_hashes,
                new_train_hashes=train_hashes,
                fresh_valid_hashes=fresh_hashes,
            )
            if sha256_file(video_path) != expected_media_hash:
                raise ValueError(f"media SHA mismatch: {video_path}")
            feature_path = output / split_name / f"{row['video_id']}.npy"
            metadata_path = output / split_name / f"{row['video_id']}.json"
            (output / split_name).mkdir(exist_ok=True)
            cache_path = args.old_feature_root / media_split / f"{row['video_id']}.npy"
            cache_meta_path = cache_path.with_suffix(".json")
            if cache_path.is_file() and cache_meta_path.is_file():
                meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                vector = np.load(cache_path, allow_pickle=False)
                cached_times = meta.get("video_target_frame_times_s", [])
                same_times = (
                    len(cached_times) == len(row["video_frame_times_s"])
                    and max(
                        abs(float(a) - float(b))
                        for a, b in zip(cached_times, row["video_frame_times_s"], strict=True)
                    )
                    < 1e-5
                )
                if (
                    meta.get("media_video_sha256") == expected_media_hash
                    and int(meta.get("event_id", -1)) == int(row["video_event_id"])
                    and same_times
                    and vector.shape == (768,)
                    and np.isfinite(vector).all()
                ):
                    record = {
                        "sample_id": row["sample_id"],
                        "split": media_split,
                        "video_id": row["video_id"],
                        "event_id": row["video_event_id"],
                        "media_video_sha256": expected_media_hash,
                        "feature_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                        "feature_dim": 768,
                        "feature_path": str(cache_path),
                        "reused_existing_exact_window": True,
                    }
                    records.append(record)
                    continue

            frames, selected_pts, max_error = closest_frames(
                video_path, row["video_frame_times_s"], av
            )
            batched = _preprocess(frames, tf)[None].astype(np.float32)
            encoded = encode(jnp.asarray(batched))
            jax.tree_util.tree_map(
                lambda value: (
                    value.block_until_ready() if hasattr(value, "block_until_ready") else value
                ),
                encoded,
            )
            tokens = encoded[0] if isinstance(encoded, tuple) else encoded
            vector = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError(f"invalid feature for {row['sample_id']}")
            digest = hashlib.sha256(vector.tobytes()).hexdigest()
            with feature_path.open("xb") as target:
                np.save(target, vector, allow_pickle=False)
            record = {
                "sample_id": row["sample_id"],
                "split": media_split,
                "video_id": row["video_id"],
                "event_id": row["video_event_id"],
                "video_window_s": row["video_window_s"],
                "video_frame_times_s": row["video_frame_times_s"],
                "video_selected_frame_pts_s": selected_pts,
                "max_frame_time_error_s": max_error,
                "media_video_sha256": expected_media_hash,
                "feature_sha256": digest,
                "feature_dim": 768,
                "feature_path": str(feature_path),
                "device": "CPU",
                "reused_existing_exact_window": False,
            }
            atomic_json(metadata_path, record)
            records.append(record)
            if len(records) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "completed": len(records),
                            "total": total,
                            "elapsed_seconds": time.perf_counter() - started,
                            "latest_sample_id": row["sample_id"],
                        }
                    ),
                    flush=True,
                )

    if len(records) != total or len({row["sample_id"] for row in records}) != total:
        raise ValueError("incomplete or duplicate VideoPrism feature records")
    result = {
        "status": "PASS_TEMPORAL_MATCH_VIDEOPRISM_FEATURES",
        "manifest_sha256": sha256_file(manifest_path),
        "examples": total,
        "exact_cached_windows_reused": sum(row["reused_existing_exact_window"] for row in records),
        "new_features_extracted": sum(not row["reused_existing_exact_window"] for row in records),
        "elapsed_seconds": time.perf_counter() - started,
        "model_parameters": parameters,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_commit": commit,
        "records": records,
        "validation_scope": "development-only; not blind evaluation",
    }
    atomic_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--old-media", type=Path, required=True)
    parser.add_argument("--old-feature-root", type=Path, required=True)
    parser.add_argument("--added-train-media", type=Path, required=True)
    parser.add_argument("--fresh-valid-media", type=Path, required=True)
    parser.add_argument("--train-acquisition", type=Path, required=True)
    parser.add_argument("--fresh-valid-media-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
