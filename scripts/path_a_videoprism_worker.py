"""Persistent JAX/TF CPU worker for Path A Image and VideoPrism encoding."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from tiny_omni_decision.feature_cache import BoundedFeatureCache


def _encode_single_image(
    runtime: Any, image_path: Path, jnp: Any
) -> tuple[np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    image = _decode_image(runtime.tf, image_path)
    height, width = (int(value) for value in runtime.tf.shape(image).numpy()[:2])
    scale = max(288 / height, 288 / width)
    resized = runtime.tf.image.resize(
        image, (round(height * scale), round(width * scale)), method="bilinear"
    )
    cropped = runtime.tf.image.resize_with_crop_or_pad(resized, 288, 288)
    frame = np.asarray(cropped, dtype=np.float32) / 255.0
    frames = np.repeat(frame[None, ...], 8, axis=0)[None, ...]
    preprocess_seconds = time.perf_counter() - started
    forward_started = time.perf_counter()
    encoded = runtime.encode(jnp.asarray(frames))
    runtime.jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        encoded,
    )
    tokens = encoded[0] if isinstance(encoded, tuple) else encoded
    feature = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
    if feature.shape != (768,) or not np.isfinite(feature).all():
        raise ValueError("VideoPrism produced invalid single-image features")
    return feature, {
        "decode_seconds": 0.0,
        "preprocess_seconds": preprocess_seconds,
        "forward_seconds": time.perf_counter() - forward_started,
        "worker_inference_seconds": time.perf_counter() - started,
    }


def _decode_image(tf: Any, image_path: Path) -> Any:
    """Decode supported local image formats without silently accepting others."""
    payload = tf.io.read_file(str(image_path))
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return tf.io.decode_png(payload, channels=3)
    if suffix in {".jpg", ".jpeg"}:
        return tf.io.decode_jpeg(payload, channels=3)
    raise ValueError(f"unsupported image extension: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    assembly = args.assembly.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    image_component = assembly / "components/image"
    runtime_path = image_component / "runtime"
    sys.path.insert(0, str(runtime_path))

    import jax
    import jax.numpy as jnp

    from scripts.extract_frozen_videoprism_features import (
        CHECKPOINT_SHA256,
        MODEL_REVISION,
        _decode_video,
        _preprocess,
        sha256_file,
    )
    from tiny_omni_decision.videoprism_runtime import VideoPrismClevr4Runtime

    started = time.perf_counter()
    process = psutil.Process()
    rss_start = process.memory_info().rss
    with contextlib.redirect_stdout(sys.stderr):
        runtime = VideoPrismClevr4Runtime(image_component, source_root)
    feature_cache_max_entries = 64
    feature_cache = BoundedFeatureCache(feature_cache_max_entries)
    load_seconds = time.perf_counter() - started
    ready = {
        "ready": True,
        "profile": "videoprism_image_video_encoder",
        "process_id": os.getpid(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "tensorflow": runtime.tf.__version__,
        "numpy": np.__version__,
        "devices": [str(device) for device in jax.devices()],
        "video_model_revision": MODEL_REVISION,
        "video_checkpoint_sha256": CHECKPOINT_SHA256,
        "video_encoder_parameters": 114_365_184,
        "feature_cache_max_entries": feature_cache_max_entries,
        "model_load_seconds": load_seconds,
        "rss_start_bytes": rss_start,
        "rss_after_load_bytes": process.memory_info().rss,
    }
    protocol = sys.stdout
    protocol.write(json.dumps(ready, ensure_ascii=False) + "\n")
    protocol.flush()

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                protocol.write('{"stopped":true}\n')
                protocol.flush()
                break
            task = str(request["task"])
            payload = request["payload"]
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                if task in {"image", "encode_image"}:
                    image_path = Path(payload["image_path"]).resolve(strict=True)
                    actual_hash = sha256_file(image_path)
                    expected_hash = payload.get("expected_media_sha256")
                    if expected_hash and actual_hash != expected_hash:
                        raise ValueError("image hash differs from the supplied frozen identity")
                    key = f"image:{actual_hash}"
                    cached = feature_cache.get(key)
                    cache_hit = cached is not None
                    if cached is None:
                        feature, encoder_metrics = _encode_single_image(runtime, image_path, jnp)
                        feature_hash = sha256_file_bytes(feature.tobytes())
                    else:
                        feature, feature_hash = cached
                        encoder_metrics = {
                            "decode_seconds": 0.0,
                            "preprocess_seconds": 0.0,
                            "forward_seconds": 0.0,
                            "worker_inference_seconds": 0.0,
                        }
                    feature_cache.put(key, feature, feature_hash)
                    if task == "encode_image":
                        result = {
                            "image_feature": feature.tolist(),
                            "encoder_metrics": {
                                **encoder_metrics,
                                "cache_hit": cache_hit,
                                "feature_sha256": feature_hash,
                                "cache_entries": len(feature_cache),
                                "rss_after_request_bytes": process.memory_info().rss,
                                "peak_working_set_bytes": getattr(
                                    process.memory_info(), "peak_wset", None
                                ),
                            },
                        }
                    else:
                        decision = runtime.decide_from_features(
                            feature,
                            str(payload["taxonomy"]),
                            tuple(str(item) for item in payload["options"]),
                        )
                        probabilities = decision.probabilities.tolist()
                        logits = decision.logits.tolist()
                        result = {
                            "logits": logits,
                            "probabilities": probabilities,
                            "prediction_index": int(np.argmax(decision.probabilities)),
                            "confidence": float(max(probabilities)),
                            "encoder_metrics": {
                                **encoder_metrics,
                                "cache_hit": cache_hit,
                                "feature_sha256": feature_hash,
                                "cache_entries": len(feature_cache),
                                "rss_after_request_bytes": process.memory_info().rss,
                                "peak_working_set_bytes": getattr(
                                    process.memory_info(), "peak_wset", None
                                ),
                            },
                        }
                elif task == "encode_video":
                    video_path = Path(payload["video_path"]).resolve(strict=True)
                    actual_hash = sha256_file(video_path)
                    expected_hash = payload.get("expected_media_sha256")
                    if expected_hash and actual_hash != expected_hash:
                        raise ValueError("video hash differs from the supplied frozen identity")
                    key = f"video:{actual_hash}"
                    cached = feature_cache.get(key)
                    cache_hit = cached is not None
                    if cached is None:
                        decode_started = time.perf_counter()
                        frames = _decode_video(video_path)
                        decoded_seconds = time.perf_counter() - decode_started
                        preprocess_started = time.perf_counter()
                        processed = _preprocess(frames, runtime.tf)
                        preprocess_seconds = time.perf_counter() - preprocess_started
                        forward_started = time.perf_counter()
                        encoded = runtime.encode(jnp.asarray(processed[None, ...]))
                        jax.tree_util.tree_map(
                            lambda value: (
                                value.block_until_ready()
                                if hasattr(value, "block_until_ready")
                                else value
                            ),
                            encoded,
                        )
                        tokens = encoded[0] if isinstance(encoded, tuple) else encoded
                        feature = np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)
                        forward_seconds = time.perf_counter() - forward_started
                        if feature.shape != (768,) or not np.isfinite(feature).all():
                            raise ValueError("VideoPrism produced invalid video features")
                        feature_hash = sha256_file_bytes(feature.tobytes())
                        encoder_metrics = {
                            "decode_seconds": decoded_seconds,
                            "preprocess_seconds": preprocess_seconds,
                            "forward_seconds": forward_seconds,
                            "worker_inference_seconds": time.perf_counter() - started,
                        }
                    else:
                        feature, feature_hash = cached
                        encoder_metrics = {
                            "decode_seconds": 0.0,
                            "preprocess_seconds": 0.0,
                            "forward_seconds": 0.0,
                            "worker_inference_seconds": 0.0,
                        }
                    feature_cache.put(key, feature, feature_hash)
                    encoder_metrics.update(
                        {
                            "cache_hit": cache_hit,
                            "feature_sha256": feature_hash,
                            "cache_entries": len(feature_cache),
                            "peak_working_set_bytes": getattr(
                                process.memory_info(), "peak_wset", None
                            ),
                            "rss_after_request_bytes": process.memory_info().rss,
                        }
                    )
                    result = {
                        "video_feature": feature.tolist(),
                        "encoder_metrics": encoder_metrics,
                    }
                else:
                    raise ValueError(f"unsupported VideoPrism task: {task}")
            result.update({"request_id": request.get("request_id"), "task": task})
            result["worker_inference_seconds"] = time.perf_counter() - started
            protocol.write(json.dumps(result, ensure_ascii=False) + "\n")
            protocol.flush()
        except Exception as error:
            protocol.write(
                json.dumps(
                    {
                        "request_id": request.get("request_id") if "request" in locals() else None,
                        "error": {"type": type(error).__name__, "message": str(error)},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            protocol.flush()


def sha256_file_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
