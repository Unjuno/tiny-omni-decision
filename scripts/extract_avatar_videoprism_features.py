"""Extract timestamp-centered frozen VideoPrism features for AVATAR choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

MODEL_REVISION = "be719a406d563b66f0ac969e7c94bab8e997c81a"
SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
SOURCE_ID = re.compile(r"[A-Za-z0-9_-]{11}_\d{5}/\d{5}\Z")
FRAME_COUNT = 8
WINDOW_SECONDS = 1.0
FRAME_SIZE = 288
BATCH_SIZE = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_choice_assets(train_path: Path, validation_path: Path) -> list[dict[str, Any]]:
    """Build deterministic unique video-frame assets, rejecting timestamp conflicts."""
    by_key: dict[str, float] = {}
    for split, path in (("train", train_path), ("validation", validation_path)):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                raise ValueError(f"{path}:{line_number}: unexpected split marker")
            for option in row.get("options", []):
                key = option.get("video_asset_key")
                timestamp = option.get("video_timestamp_seconds")
                if not isinstance(key, str) or SOURCE_ID.fullmatch(key) is None:
                    raise ValueError(f"{path}:{line_number}: invalid video asset key")
                if not isinstance(timestamp, (int, float)) or timestamp < 0:
                    raise ValueError(f"{path}:{line_number}: invalid video timestamp")
                previous = by_key.setdefault(key, float(timestamp))
                if abs(previous - float(timestamp)) > 1e-9:
                    raise ValueError(f"conflicting timestamp for video asset {key}")
    return [{"asset_key": key, "timestamp_seconds": by_key[key]} for key in sorted(by_key)]


def _decode_window(path: Path, center_seconds: float) -> tuple[np.ndarray, float]:
    import av

    requested = np.linspace(
        center_seconds - WINDOW_SECONDS / 2,
        center_seconds + WINDOW_SECONDS / 2,
        FRAME_COUNT,
    )
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"no video stream: {path}")
        stream = streams[0]
        duration = (
            float(container.duration / av.time_base) if container.duration is not None else None
        )
        if duration is None or duration <= 0:
            raise ValueError(f"video duration unavailable: {path}")
        targets = np.clip(requested, 0.0, max(0.0, duration - 1e-6))
        first_time = float(targets[0])
        seek_offset = max(0, int(first_time / float(stream.time_base)))
        container.seek(seek_offset, stream=stream, backward=True, any_frame=False)
        decoded_times: list[float] = []
        decoded_frames: list[np.ndarray] = []
        last_target = float(targets[-1])
        for frame in container.decode(stream):
            frame_time = (
                float(frame.pts * stream.time_base)
                if frame.pts is not None
                else (decoded_times[-1] if decoded_times else 0.0)
            )
            decoded_times.append(frame_time)
            decoded_frames.append(frame.to_ndarray(format="rgb24"))
            if frame_time > last_target + 0.1:
                break
        if not decoded_frames:
            raise ValueError(f"no frames decoded at requested time: {path} {center_seconds}")
        times = np.asarray(decoded_times)
        indices = np.abs(times[None, :] - targets[:, None]).argmin(axis=1)
        selected = np.stack([decoded_frames[int(index)] for index in indices])
        errors = np.abs(times[indices] - targets)
    if selected.ndim != 4 or selected.shape[0] != FRAME_COUNT:
        raise ValueError(f"unexpected frame selection shape: {selected.shape}")
    return selected, float(errors.max(initial=0.0))


def _preprocess_frames(frames: np.ndarray, jax: Any) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[0] != FRAME_COUNT or frames.shape[-1] != 3:
        raise ValueError(f"unexpected decoded frame shape: {frames.shape}")
    height, width = frames.shape[1:3]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square = frames[:, top : top + side, left : left + side, :]
    resized = jax.image.resize(
        jax.numpy.asarray(square, dtype=jax.numpy.float32) / 255.0,
        (FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3),
        method="cubic",
        antialias=True,
    )
    result = np.asarray(resized, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite preprocessed video frame")
    return result


class VideoPrismEncoder:
    def __init__(self, source_root: Path, checkpoint: Path):
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("VideoPrism checkpoint hash mismatch")
        commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != SOURCE_COMMIT:
            raise ValueError(f"unexpected VideoPrism source commit: {commit}")
        import jax
        import jax.numpy as jnp

        sys.path.insert(0, str(source_root.resolve()))
        from videoprism import models as vp

        self.jax = jax
        self.model = vp.get_model("videoprism_public_v1_base")
        self.variables = vp.load_pretrained_weights(
            "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
        )
        self.forward = jax.jit(lambda clips: self.model.apply(self.variables, clips, train=False))
        self.jnp = jnp
        self.parameter_count = sum(
            int(value.size) for value in jax.tree_util.tree_leaves(self.variables)
        )

    def encode(self, clips: np.ndarray) -> np.ndarray:
        if clips.shape != (BATCH_SIZE, FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3):
            raise ValueError(f"unexpected VideoPrism batch shape: {clips.shape}")
        result = self.forward(self.jnp.asarray(clips))
        self.jax.tree_util.tree_map(
            lambda value: (
                value.block_until_ready() if hasattr(value, "block_until_ready") else value
            ),
            result,
        )
        tokens = result[0] if isinstance(result, tuple) else result
        features = np.asarray(tokens).mean(axis=1).astype(np.float32)
        if features.shape != (BATCH_SIZE, 768) or not np.isfinite(features).all():
            raise ValueError(f"invalid VideoPrism features: {features.shape}")
        return features


def _run(args: argparse.Namespace) -> dict[str, Any]:
    train_path = args.train_input.resolve(strict=True)
    validation_path = args.validation_input.resolve(strict=True)
    media_dir = args.media_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    source_root = args.source_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    media_receipt_path = args.media_receipt.resolve(strict=True)
    if not media_dir.is_dir():
        raise NotADirectoryError(media_dir)
    if args.max_assets is not None and args.max_assets < 1:
        raise ValueError("--max-assets must be positive")
    all_assets = load_choice_assets(train_path, validation_path)
    assets = all_assets[: args.max_assets] if args.max_assets is not None else all_assets
    input_order_hash = canonical_sha256(assets)
    media_receipt = json.loads(media_receipt_path.read_text(encoding="utf-8"))
    if (
        media_receipt.get("status") != "PASS"
        or media_receipt.get("source_archive_sha256")
        != "5ce5be7fa6dda85889750f6c7cc2fad807527063e7d7090c664170a66b078a68"
    ):
        raise ValueError("AVATAR extraction receipt does not match the pinned source archive")
    media_records = {row["name"]: row for row in media_receipt.get("media_sha256", [])}
    required_clips = {row["asset_key"].rsplit("/", 1)[0] + ".mp4" for row in assets}
    if required_clips - set(media_records):
        raise ValueError("media receipt is missing clips referenced by the frozen choice inputs")
    for name in sorted(required_clips):
        path = media_dir / name
        record = media_records[name]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"extracted media failed receipt verification: {path}")
    identity = {
        "repository_head": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        "train_input_sha256": sha256_file(train_path),
        "validation_input_sha256": sha256_file(validation_path),
        "media_receipt_sha256": sha256_file(media_receipt_path),
        "asset_order_sha256": input_order_hash,
        "asset_count": len(assets),
        "max_assets": args.max_assets,
        "media_dir": str(media_dir),
        "video_model": "google/videoprism-base-f16r288",
        "model_revision": MODEL_REVISION,
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "frame_count": FRAME_COUNT,
        "window_seconds": WINDOW_SECONDS,
        "preprocessing": (
            "8 frames over 1.0s centered at annotation time; clamp to media duration; "
            "center square crop; JAX cubic resize to 288; RGB float32 [0,1]"
        ),
        "pooling": "mean over 2048 VideoPrism tokens",
        "batch_size": BATCH_SIZE,
    }
    identity_hash = canonical_sha256(identity)
    shards_dir = output / "shards"
    if output.exists():
        config_path = output / "effective-config.json"
        progress_path = output / "progress.json"
        if not config_path.is_file() or not progress_path.is_file() or not shards_dir.is_dir():
            raise FileExistsError(f"output exists without a valid resumable extraction: {output}")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            saved_config.get("identity") != identity
            or progress.get("identity_sha256") != identity_hash
        ):
            raise ValueError("existing output belongs to a different extraction identity")
        if progress.get("status") == "PASS":
            result_path = output / "result.json"
            if not result_path.is_file() or sha256_file(result_path) != progress.get(
                "result_sha256"
            ):
                raise ValueError("completed extraction result failed its saved hash")
            return json.loads(result_path.read_text(encoding="utf-8"))
        if progress.get("status") != "RUNNING":
            raise ValueError("existing extraction is not resumable")
        completed_assets = int(progress.get("completed_assets", -1))
        records = progress.get("shards", [])
        expected_records = (completed_assets + BATCH_SIZE - 1) // BATCH_SIZE
        if (
            completed_assets < 0
            or completed_assets > len(assets)
            or (completed_assets != len(assets) and completed_assets % BATCH_SIZE != 0)
            or len(records) != expected_records
        ):
            raise ValueError("progress metadata has an inconsistent completed-asset count")
        for index, record in enumerate(records):
            shard = shards_dir / record["name"]
            first = index * BATCH_SIZE
            expected_rows = min(BATCH_SIZE, completed_assets - first)
            if (
                record.get("first_asset_index") != first
                or record.get("asset_count") != expected_rows
                or not shard.is_file()
                or sha256_file(shard) != record.get("sha256")
            ):
                raise ValueError(f"saved extraction shard failed validation: {shard}")
            with np.load(shard, allow_pickle=False) as saved:
                if saved["asset_keys"].tolist() != [
                    row["asset_key"] for row in assets[first : first + expected_rows]
                ]:
                    raise ValueError(f"saved shard IDs do not match the frozen order: {shard}")
        known_shards = {row["name"] for row in records}
        if any(path.name not in known_shards for path in shards_dir.glob("*.npz")):
            raise ValueError("untracked output shard exists; refusing ambiguous resume")
        start_asset = completed_assets
    else:
        output.mkdir(parents=True)
        shards_dir.mkdir()
        write_json_atomic(
            output / "effective-config.json",
            {
                "identity": identity,
                "identity_sha256": identity_hash,
                "started_utc": datetime.now(UTC).isoformat(),
            },
        )
        progress = {
            "status": "RUNNING",
            "identity_sha256": identity_hash,
            "asset_count": len(assets),
            "completed_assets": 0,
            "shards": [],
        }
        write_json_atomic(output / "progress.json", progress)
        start_asset = 0
    extraction_start = time.perf_counter()
    model_start = time.perf_counter()
    encoder = VideoPrismEncoder(source_root, checkpoint)
    model_load_seconds = time.perf_counter() - model_start
    max_time_error = max(
        (float(record.get("max_frame_time_error_seconds", 0.0)) for record in progress["shards"]),
        default=0.0,
    )
    for batch_index, first in enumerate(
        range(start_asset, len(assets), BATCH_SIZE), start=start_asset // BATCH_SIZE
    ):
        rows = assets[first : first + BATCH_SIZE]
        clips = np.zeros((BATCH_SIZE, FRAME_COUNT, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)
        errors = []
        decode_start = time.perf_counter()
        for local_index, row in enumerate(rows):
            key = row["asset_key"]
            clip_name = key.rsplit("/", 1)[0] + ".mp4"
            media_path = media_dir / clip_name
            if not media_path.is_file():
                raise FileNotFoundError(media_path)
            frames, time_error = _decode_window(media_path, row["timestamp_seconds"])
            clips[local_index] = _preprocess_frames(frames, encoder.jax)
            errors.append(time_error)
        decode_seconds = time.perf_counter() - decode_start
        forward_start = time.perf_counter()
        features = encoder.encode(clips)
        forward_seconds = time.perf_counter() - forward_start
        count = len(rows)
        keys = [row["asset_key"] for row in rows]
        shard = shards_dir / f"shard-{batch_index:05d}.npz"
        temporary = shard.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                asset_keys=np.asarray(keys, dtype=np.str_),
                timestamps=np.asarray([row["timestamp_seconds"] for row in rows], dtype=np.float64),
                features=features[:count],
                max_frame_time_error_seconds=np.asarray(errors, dtype=np.float32),
            )
        temporary.replace(shard)
        record = {
            "name": shard.name,
            "sha256": sha256_file(shard),
            "first_asset_index": first,
            "asset_count": count,
            "decode_preprocess_seconds": decode_seconds,
            "model_forward_seconds": forward_seconds,
            "max_frame_time_error_seconds": max(errors, default=0.0),
        }
        progress["shards"].append(record)
        progress["completed_assets"] = first + count
        progress["last_updated_utc"] = datetime.now(UTC).isoformat()
        write_json_atomic(output / "progress.json", progress)
        max_time_error = max(max_time_error, record["max_frame_time_error_seconds"])
        if (batch_index + 1) % 32 == 0 or first + count == len(assets):
            print(
                f"completed {first + count}/{len(assets)} assets; "
                f"decode={decode_seconds:.2f}s forward={forward_seconds:.2f}s",
                flush=True,
            )
    final_keys: list[str] = []
    final_times: list[np.ndarray] = []
    final_features: list[np.ndarray] = []
    final_errors: list[np.ndarray] = []
    for record in progress["shards"]:
        shard = shards_dir / record["name"]
        if sha256_file(shard) != record["sha256"]:
            raise ValueError(f"shard hash mismatch: {shard}")
        with np.load(shard, allow_pickle=False) as values:
            final_keys.extend(values["asset_keys"].tolist())
            final_times.append(values["timestamps"])
            final_features.append(values["features"])
            final_errors.append(values["max_frame_time_error_seconds"])
    if final_keys != [row["asset_key"] for row in assets]:
        raise ValueError("final feature asset order differs from frozen input order")
    final_path = output / "features.npz"
    with final_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            asset_keys=np.asarray(final_keys, dtype=np.str_),
            timestamps=np.concatenate(final_times),
            features=np.concatenate(final_features),
            max_frame_time_error_seconds=np.concatenate(final_errors),
        )
    result = {
        "status": "PASS",
        "identity_sha256": identity_hash,
        "asset_count": len(final_keys),
        "unique_clip_count": len({key.rsplit("/", 1)[0] for key in final_keys}),
        "feature_shape": [len(final_keys), 768],
        "feature_file_sha256": sha256_file(final_path),
        "feature_file_bytes": final_path.stat().st_size,
        "max_frame_time_error_seconds": max_time_error,
        "parameter_count": encoder.parameter_count,
        "device": [str(device) for device in encoder.jax.devices()],
        "model_load_seconds": model_load_seconds,
        "wall_seconds": time.perf_counter() - extraction_start,
        "sealed_audit_loaded": False,
    }
    write_json_atomic(output / "result.json", result)
    progress["status"] = "PASS"
    progress["result_sha256"] = sha256_file(output / "result.json")
    write_json_atomic(output / "progress.json", progress)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--media-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-assets", type=int)
    args = parser.parse_args()
    print(json.dumps(_run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
